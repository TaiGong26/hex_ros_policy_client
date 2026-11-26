################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Callable

import numpy as np

from ...struct.cmd_struct import (
    ARM_CTRL_MODES,
    ArmCommand,
    ControlIntent,
    GripCommand,
)
from ...utils import DEFAULT_DECAY, weight_recency
from .action_process_base import ActionProcessBase, ActionProcessParam
from .action_restore import ACTION_RESTORERS, as_received

# How many restored points are kept by default; the oldest one leaves first.
DEFAULT_QUEUE_CAPACITY = 100

# How far the window reaches by default; the meaning of the two numbers is
# described on ``WindowQueryParam.window_steps``.
DEFAULT_WINDOW_STEPS = (1.0, 1.0)

NS_PER_MS = 1_000_000

_CHUNK_KEYS = ("stamp_ns", "chunk_step_interval_ms", "action_mode",
               "arm_ctrl_mode", "arm", "grip")

# Class of the command of each group; the group names are also the fields of
# the ``ControlIntent`` the query assembles, so the two cannot drift apart.
_COMMANDS = {"arm": ArmCommand, "grip": GripCommand}

_GROUPS = tuple(_COMMANDS)


@dataclass(frozen=True)
class WindowQueryParam(ActionProcessParam):
    """Parameters of the window query.

    Attributes:
        window_steps (tuple[float, float]): How far the window reaches around
            the query instant, as multiples of one step, ``(past, future)``;
            neither one may be negative, and both ends are closed. The two
            reaches trade smoothness against sharpness: a bigger one blends
            more points and smooths the motion, a smaller one follows the
            newest point more sharply.
        queue_capacity (int): How many restored points are kept; the oldest
            one leaves first.
        decay (float): Weight of the newer samples of one device; zero
            weights every sample the same.
    """

    window_steps: tuple[float, float] = DEFAULT_WINDOW_STEPS
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY
    decay: float = DEFAULT_DECAY

    def __post_init__(self) -> None:
        """Reject a window or a limit that cannot describe a query.

        Raises:
            ValueError: If a reach is not finite or is negative, if the
                capacity is not positive, or if the decay is not finite.
        """
        super().__post_init__()
        past_steps, future_steps = self.window_steps
        if not (np.isfinite(past_steps) and np.isfinite(future_steps)):
            raise ValueError(
                f"window_steps must be finite, got {self.window_steps!r}")
        if past_steps < 0.0 or future_steps < 0.0:
            raise ValueError(
                "window_steps must not be negative, "
                f"got {self.window_steps!r}")
        if self.queue_capacity <= 0:
            raise ValueError(
                f"queue_capacity must be positive, "
                f"got {self.queue_capacity}")
        if not np.isfinite(self.decay):
            raise ValueError(f"decay must be finite, got {self.decay!r}")


@dataclass(frozen=True)
class _Point:
    """One restored point of the action queue.

    Attributes:
        t_ns (int): Absolute time of the point in nanoseconds.
        control (dict): Group name to the device name to the absolute values
            of this point.
    """

    t_ns: int
    control: dict = field(default_factory=dict)


class ActionWindowQuery(ActionProcessBase):
    """Action processor that answers a time window with one weighted target.

    A chunk is restored once, on arrival, and cut into one point per step.
    A query takes the points that fall inside a window around the current
    instant and blends them, so no time grid is kept and no step is
    interpolated here.
    """

    def __init__(self, params: WindowQueryParam) -> None:
        """Create a processor with an empty queue.

        Args:
            params (WindowQueryParam): Query parameters, already checked.
        """
        super().__init__(params=params)
        self.__params = params
        self.__queue: deque[_Point] = deque(maxlen=params.queue_capacity)
        self.__latched_step_ns = 0
        self.__last_query_ns = 0

    def push_chunk(self, chunk: dict) -> None:
        """Accept one decoded chunk and store its points.

        The steps are restored to absolute values on arrival; an unusable
        chunk is logged and dropped, never raised.

        Args:
            chunk (dict): Decoded chunk of the protocol.
        """
        try:
            points = self.__chunk_to_points(chunk=chunk)
        except (ValueError, TypeError) as error:
            self._logger.logw(
                "[chunk.push/schema_invalid] action chunk dropped: "
                f"invalid structure: {error}")
            return
        if points is None:
            return
        self.__store_points(points=points)

    def get_target(
        self,
        query_offset_ns: int | None = None,
    ) -> ControlIntent | None:
        """Return the target that applies at the target query instant.

        Args:
            query_offset_ns (int | None): Query time offset in nanoseconds;
                defaults to the action chunk step interval if None.

        Returns:
            ControlIntent | None: Commands keyed by ``arm`` and ``grip``;
            ``None`` when no stored point falls in the window.
        """
        now_ns = self._now_ns_fn()
        self.__warn_clock_backwards(now_ns=now_ns)
        self.__last_query_ns = now_ns
        offset_ns = (
            int(query_offset_ns)
            if query_offset_ns is not None
            else self.__latched_step_ns
        )
        target_time_ns = now_ns + offset_ns
        points = self.__points_in_window(now_ns=target_time_ns)
        if not points:
            self._logger.logd(
                "[chunk.target/queue_empty] no target this tick: no point "
                "falls in the window, holding the measured snapshot")
            return None
        samples = self.__gather_samples(points=points)
        blended = self.__blend_samples(samples=samples)
        return self.__build_commands(
            blended=blended,
            stamp_ns=target_time_ns,
        )

    def __gather_samples(
            self,
            points: list[_Point]) -> dict[str, dict[str, list[np.ndarray]]]:
        """Collect the samples each device offers inside the window.

        Args:
            points (list[_Point]): Points inside the window, oldest first.

        Returns:
            dict[str, dict[str, list[np.ndarray]]]: Group name to device name
            to its samples, oldest first; devices with no point are absent.
        """
        samples: dict[str, dict[str, list[np.ndarray]]] = {
            group: {} for group in _COMMANDS}
        for point in points:
            for group in _COMMANDS:
                for name, sample in point.control.get(group, {}).items():
                    samples[group].setdefault(name, []).append(sample)
        return samples

    def __blend_samples(
            self,
            samples: dict[str, dict[str, list[np.ndarray]]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Blend the samples of each device into one value.

        Args:
            samples (dict[str, dict[str, list[np.ndarray]]]): Group name to
                the device name to its samples, oldest first.

        Returns:
            dict[str, dict[str, np.ndarray]]: Group name to device name to
            its blended value.
        """
        return {
            group: {name: weight_recency(samples=device_samples,
                                         weight=self.__params.decay)
                    for name, device_samples in per_device.items()}
            for group, per_device in samples.items()}

    def __build_commands(
        self,
        blended: dict[str, dict[str, np.ndarray]],
        stamp_ns: int | None = None,
    ) -> ControlIntent:
        """Build the commands of one tick out of the blended values.

        Args:
            blended (dict[str, dict[str, np.ndarray]]): Group name to device
                name to its blended value.
            stamp_ns (int | None): Target time in nanoseconds for this intent.

        Returns:
            ControlIntent: Commands keyed by ``arm`` and ``grip``; a group
            without a device stays empty.
        """
        commands: dict = {group: {} for group in _COMMANDS}
        for group, command_class in _COMMANDS.items():
            for name, value in blended[group].items():
                commands[group][name] = command_class(pos=value)
        return ControlIntent(stamp_ns=stamp_ns, **commands)

    def reset(self) -> None:
        """Drop every stored point, the latched step and the references."""
        super().reset()
        self.__queue.clear()
        self.__latched_step_ns = 0
        self.__last_query_ns = 0

    def __chunk_to_points(self, chunk: dict) -> list[_Point] | None:
        """Check one chunk and turn it into absolute points.

        Args:
            chunk (dict): Chunk as delivered by the middleware.

        Returns:
            list[_Point] | None: One point per step, oldest first; ``None``
            when the chunk is dropped, which is logged.

        Raises:
            ValueError: If a required key is missing, or if a group, a device
                or a value cannot be restored.
        """
        missing = [key for key in _CHUNK_KEYS if key not in chunk]
        if missing:
            raise ValueError(f"missing fields {missing}")
        arm_mode = chunk["arm_ctrl_mode"]
        action_mode = chunk["action_mode"]
        restorer = ACTION_RESTORERS.get((arm_mode, action_mode))
        
        if restorer is None:
            self._logger.logw(
                "[chunk.mode/unsupported_mode] action chunk dropped: "
                f"arm_ctrl_mode={arm_mode} action_mode={action_mode} "
                "has no restorer")
            return None
        
        ## The current arm mode only supports "pos"
        if arm_mode != ARM_CTRL_MODES[0]:
            self._logger.logw(
                "[chunk.mode/unsupported_mode] action chunk dropped: "
                f"arm_ctrl_mode={arm_mode} is not dispatched at this stage")
            return None
        
        declared_ms = chunk["chunk_step_interval_ms"]
        step_ns = self.__latch_step(declared_ms=declared_ms)
        if step_ns is None:
            return None
        
        stamp_ns = int(chunk["stamp_ns"])
        anchor = None
        if restorer is not as_received:
            anchor = self._anchor_at(
                header_ns=stamp_ns,
                arm_ctrl_mode=arm_mode,
            )
            if anchor is None:
                return None
        
        matrices = self.__restore_matrices(
            chunk=chunk,
            anchor=anchor,
            restorer=restorer,
        )

        return self.__matrices_to_points(
            stamp_ns=stamp_ns,
            step_ns=step_ns,
            matrices=matrices)

    def __restore_matrices(
            self,
            chunk: dict,
            anchor: dict | None,
            restorer: Callable[..., np.ndarray],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Turn the received values of one chunk into absolute matrices.

        Args:
            chunk (dict): Chunk as delivered by the middleware; ``arm`` and
                ``grip`` hold one step list per device.
            anchor (dict | None): Absolute values the increments start from,
                keyed by group and device name; ``None`` for absolute chunks.
            restorer (Callable[..., np.ndarray]): Conversion chosen by the
                modes of the chunk.

        Returns:
            dict[str, dict[str, np.ndarray]]: Group name to device name to
            its matrix of shape ``(steps, width)``.

        Raises:
            ValueError: If a group is not a mapping or is empty, if a device
                has no reference of the right width, if a matrix is empty or
                not two dimensional, or if a value is not finite.
        """
        matrices: dict[str, dict[str, np.ndarray]] = {}
        for group in _GROUPS:
            devices = chunk[group]
            if not isinstance(devices, dict):
                raise ValueError(f"{group} is not a mapping")
            matrices[group] = {}

            for name, rows in devices.items():
                matrix = np.array(rows, dtype=np.float64)
                if matrix.ndim != 2 or matrix.shape[0] == 0:
                    raise ValueError(f"{group}.{name} has an invalid shape")
                reference = None
                if anchor is not None:
                    reference = anchor.get(group, {}).get(name)
                    if reference is None:
                        raise ValueError(
                            f"{group}.{name} has no reference in the anchor")
                    if np.shape(reference) != (matrix.shape[1],):
                        raise ValueError(
                            f"{group}.{name} is {matrix.shape[1]} wide but "
                            f"its reference is {np.shape(reference)}")
                restored = restorer(received=matrix, anchor=reference)
                if not np.isfinite(restored).all():
                    raise ValueError(
                        f"{group}.{name} has a value that is not finite")
                matrices[group][name] = restored
            if not matrices[group]:
                raise ValueError(f"{group} is empty")
        return matrices

    def __matrices_to_points(
            self,
            stamp_ns: int,
            step_ns: int,
            matrices: dict[str, dict[str, np.ndarray]],
    ) -> list[_Point]:
        """Cut the restored matrices into one point per step.

        Devices may carry different numbers of steps; a shorter one stops
        contributing past its end.

        Args:
            stamp_ns (int): Time of the first step in nanoseconds.
            step_ns (int): Duration of one step in nanoseconds.
            matrices (dict[str, dict[str, np.ndarray]]): Group name to
                device name to its matrix of shape ``(steps, width)``.

        Returns:
            list[_Point]: One point per step of the longest device, oldest
            first.
        """
        rows = max(matrix.shape[0]
                   for group in _GROUPS
                   for matrix in matrices[group].values())
        points = []
        for index in range(rows):
            control = {}
            for group in _GROUPS:
                control[group] = {
                    name: matrix[index]
                    for name, matrix in matrices[group].items()
                    if matrix.shape[0] > index
                }
            points.append(_Point(
                t_ns=stamp_ns + index * step_ns,
                control=control,
            ))
        return points

    def __latch_step(self, declared_ms: int | None) -> int | None:
        """Latch the step of the first chunk and return the one to use.

        A later chunk with another step is reported and ignored.

        Args:
            declared_ms (int | None): ``chunk_step_interval_ms`` as received;
                ``None`` when the field was empty.

        Returns:
            int | None: Step in nanoseconds; ``None`` when the declared value
            is not positive, which is logged.
        """
        if declared_ms is None or int(declared_ms) <= 0:
            self._logger.logw(
                "[chunk.push/step_missing] action chunk dropped: "
                f"chunk_step_interval_ms={declared_ms!r} is not a positive "
                "number of milliseconds")
            return None
        declared_ns = int(declared_ms) * NS_PER_MS
        if not self.__latched_step_ns:
            self.__latched_step_ns = declared_ns
            return declared_ns

        if declared_ns != self.__latched_step_ns:
            self._logger.logw(
                "[chunk.push/step_mismatch] step ignored: "
                f"expected_ns={self.__latched_step_ns} "
                f"received_ns={declared_ns}, "
                "the first step of the task is kept")

        return self.__latched_step_ns

    def __store_points(self, points: list[_Point]) -> None:
        """Store the points of one chunk in time order.

        A point of an instant already stored replaces it; past the capacity
        the oldest point leaves.

        Args:
            points (list[_Point]): Points of one chunk, in time order.
        """
        if not points:
            return
        merged = list(self.__queue)
        merged.extend(points)
        merged.sort(key=attrgetter("t_ns"))
        deduped = [merged[0]]
        for point in merged[1:]:
            if point.t_ns == deduped[-1].t_ns:
                deduped[-1] = point
            else:
                deduped.append(point)
        self.__queue = deque(deduped, maxlen=self.__params.queue_capacity)

    def __points_in_window(self, now_ns: int) -> list[_Point]:
        """Return the stored points inside the window of this instant.

        Args:
            now_ns (int): Query time in nanoseconds.

        Returns:
            list[_Point]: Points inside the window, oldest first; both ends
            are included.
        """
        start_ns, end_ns = self.__window_edges(now_ns=now_ns)
        stored = list(self.__queue)
        stamps = [point.t_ns for point in stored]
        first = bisect_left(stamps, start_ns)
        last = bisect_left(stamps, end_ns, lo=first)
        if last < len(stored) and stored[last].t_ns == end_ns:
            last += 1
        return stored[first:last]

    def __window_edges(self, now_ns: int) -> tuple[int, int]:
        """Return the two instants that bound the window.

        Args:
            now_ns (int): Query time in nanoseconds.

        Returns:
            tuple[int, int]: First and last instant in nanoseconds; both are
            included.
        """
        past_steps, future_steps = self.__params.window_steps
        return (now_ns - int(round(past_steps * self.__latched_step_ns)),
                now_ns + int(round(future_steps * self.__latched_step_ns)))

    def __warn_clock_backwards(self, now_ns: int) -> None:
        """Warn when the clock went backwards, keeping the stored points.

        A rollback is the node's business; here it is only reported.

        Args:
            now_ns (int): Time of this tick in nanoseconds.
        """
        if self.__last_query_ns and now_ns < self.__last_query_ns:
            self._logger.logw(
                "[chunk.queue/clock_backwards] clock went backwards: "
                f"now_ns={now_ns} is earlier than the last query "
                f"{self.__last_query_ns}, the stored points are kept")
