################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ...struct.cmd_struct import ControlIntent
from ...struct.device_struct import DeviceObservationFrame
from ...struct.geom_struct import Pose

# How many observations are kept, so the increments have a reference to use.
DEFAULT_ANCHOR_CAPACITY = 100


@dataclass(frozen=True)
class ActionProcessParam:
    """Parameters every action process takes.

    Attributes:
        arm_names (tuple[str, ...]): Arm names of the topology.
        now_fn (Callable[[], int]): Clock injected by the node, returning the
            current time in nanoseconds.
        anchor_tolerance_ns (int): Largest distance in nanoseconds between a
            chunk stamp and the observation used as its reference; beyond it
            the chunk has no usable reference.
        anchor_capacity (int): How many observations are kept; the oldest
            one is dropped first.
        logger (Any | None): Logger used for drop lines.
    """

    arm_names: tuple[str, ...]
    now_fn: Callable[[], int]
    anchor_tolerance_ns: int
    anchor_capacity: int = DEFAULT_ANCHOR_CAPACITY
    logger: Any | None = None

    def __post_init__(self) -> None:
        """Reject a limit that cannot describe a history.

        Raises:
            ValueError: If the tolerance or the capacity is not positive.
        """
        if self.anchor_tolerance_ns <= 0:
            raise ValueError(
                "anchor_tolerance_ns must be positive, "
                f"got {self.anchor_tolerance_ns}")
        if self.anchor_capacity <= 0:
            raise ValueError(
                "anchor_capacity must be positive, "
                f"got {self.anchor_capacity}")


class ActionProcessBase(ABC):
    """Base class of the action chunk processors.

    The node calls :meth:`push_chunk` when a chunk arrives, :meth:`push_obs`
    for every observation it publishes, :meth:`get_target` on every control
    tick and :meth:`reset` when a task starts. The calls interleave, so no
    order between them is assumed.
    """

    def __init__(self, params: ActionProcessParam) -> None:
        """Store the topology, the clock and the reference policy.

        Args:
            params (ActionProcessParam): Parameters of the processor,
                already checked when the parameter object was built.
        """
        self._arm_names = tuple(params.arm_names)
        self._now_ns_fn = params.now_fn
        self._logger = params.logger
        self.__anchor_tolerance_ns = int(params.anchor_tolerance_ns)
        # Stamps with their values, oldest first, bounded by the capacity:
        # the search bisects them, so the order has to hold.
        self.__anchors: deque[tuple[int, dict]] = deque(
            maxlen=int(params.anchor_capacity))

    def push_obs(self, observation: DeviceObservationFrame) -> None:
        """Keep one observation as the reference of the increment modes.

        Both arm values are extracted here, because the control mode of a
        future chunk is unknown and each mode needs its own value.

        Args:
            observation (DeviceObservationFrame): Assembled frame; only its arm
                samples carry a reference, a sample whose arm state is
                ``None`` is skipped, and a frame without a stamp is ignored.
        """
        stamp_ns = observation.get("stamp_ns")

        if stamp_ns is None:
            return
        reference = {"arm": {}, "grip": {}}

        for name, sample in observation.get("arm", {}).items():
            state = sample.arm_state
            if state is None:
                continue

            joints = np.array(state.arm_pos, dtype=np.float64).reshape(-1)
            pose = np.asarray(
                _pose_values(pose=state.ee_base), dtype=np.float64)

            reference["arm"][name] = {"pos": joints, "ee": pose}

            grip = sample.grip_state
            if grip is not None:
                reference["grip"][name] = np.array(
                    grip.grip_pos, dtype=np.float64).reshape(-1)

        if not reference["arm"] and not reference["grip"]:
            return

        entry = (int(stamp_ns), reference)
        if not self.__anchors or entry[0] >= self.__anchors[-1][0]:
            # In stamp order, so a full deque drops its oldest stamp itself.
            self.__anchors.append(entry)
            return

        # A frame that arrives late is placed where its stamp belongs. Only
        # the stamp is compared: the value holds arrays, and comparing two
        # equal stamps by value would compare them.
        index = bisect_left(self.__anchors, (entry[0],))
        if len(self.__anchors) == self.__anchors.maxlen:
            if index == 0:
                # Older than every kept stamp: this one would be dropped.
                return
            self.__anchors.popleft()
            index -= 1
        self.__anchors.insert(index, entry)

    def _anchor_at(self, header_ns: int, arm_ctrl_mode: str) -> dict | None:
        """Return the absolute values that a chunk's increments are added to.

        ``header_ns`` comes from the model side and the stamps come from the
        observation layer, so the two need not agree: the answer is the entry
        whose stamp is the closest one, and no conversion is done. The history
        is kept in stamp order, so only the entries on either side of the
        insertion point can be the closest; the older one wins a tie.

        Args:
            header_ns (int): Reference time the chunk declares, in
                nanoseconds.
            arm_ctrl_mode (str): Mode of the arm value to pick, one of
                ``ARM_CTRL_MODES``; the gripper is not affected by the mode.

        Returns:
            dict | None: ``arm`` mapping of device name to the value of the
            requested mode, ``grip`` mapping of device name to the gripper
            position, ``None`` when the history is empty or nothing lies
            within ``anchor_tolerance_ns`` of ``header_ns``.
        """
        if not self.__anchors:
            self._logger.logw(
                "[chunk.anchor/missing] action chunk dropped: no "
                f"observation received yet, header_ns={header_ns}")
            return None
        
        index = bisect_left(self.__anchors, (header_ns,))
        index = min(index, len(self.__anchors) - 1)
        if index and header_ns - self.__anchors[index - 1][0] <= (
                self.__anchors[index][0] - header_ns):
            index -= 1

        stamp_ns, values = self.__anchors[index]
        
        if abs(stamp_ns - header_ns) > self.__anchor_tolerance_ns:
            limit_ms = self.__anchor_tolerance_ns / 1e6
            self._logger.logw(
                "[chunk.anchor/missing] action chunk dropped: no observation "
                f"within {limit_ms:.1f}ms of header_ns={header_ns}, "
                f"history={len(self.__anchors)}")
            return None
        
        anchor = {"arm": {},
                  "grip": {name: value.copy()
                           for name, value in values["grip"].items()}}
        
        for name, per_mode in values["arm"].items():
            chosen = per_mode.get(arm_ctrl_mode)
            if chosen is not None:
                anchor["arm"][name] = chosen.copy()

        return anchor

    @abstractmethod
    def push_chunk(self, chunk: dict) -> None:
        """Accept one chunk received from the model side.

        A chunk that cannot be used is dropped with a warning rather than
        raised, so the caller keeps pushing at the rate the model sends.

        Args:
            chunk (dict): Decoded chunk with ``stamp_ns``,
                ``chunk_step_interval_ms``, ``action_mode``, ``arm_ctrl_mode``
                and one step list per device under ``arm`` and ``grip``, each
                shaped ``steps x dof`` with ``dof`` 1 for a gripper, keyed by
                the name the model uses; the implementation turns those lists
                into arrays, the protocol helper does not.
        """

    @abstractmethod
    def get_target(
        self,
        query_offset_ns: int | None = None,
    ) -> ControlIntent | None:
        """Return the target that applies at the target query instant.

        Args:
            query_offset_ns (int | None): Query time offset in nanoseconds;
                defaults to the action chunk step interval if None.

        Returns:
            ControlIntent | None: Commands keyed by ``arm`` and ``grip``,
            ``None`` when no stored chunk applies to the target instant.
        """

    def reset(self) -> None:
        """Drop every kept reference so a new task starts empty.

        A subclass stores its own state and calls this one first, so no
        reference of the finished task can be the anchor of the next one.
        """
        self.__anchors.clear()


def _pose_values(pose: Pose) -> list[float]:
    """Flatten one pose into the seven values the wire carries.

    Args:
        pose (Pose): Position and orientation of the end effector.

    Returns:
        list[float]: Seven values as ``(x, y, z, qw, qx, qy, qz)``.
    """
    position = pose.position
    orientation = pose.orientation
    return [position.x, position.y, position.z,
            orientation.w, orientation.x, orientation.y, orientation.z]
