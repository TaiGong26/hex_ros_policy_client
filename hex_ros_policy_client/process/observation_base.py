################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Abstract observation assembler.

The assembler takes the grouped dictionary produced by ``Robot.read_obs()``,
checks that every required source is present and fresh, and hands one nested
dictionary with a single frame stamp to the middleware.

The required sources are not hard coded: they are derived from the two name
lists of the device hub, so adding or removing a device changes the checks
without touching this file. Depth is not required, because a camera may be
declared without a depth stream.

Attributes:
    SYNC_STRATEGIES: Registered time sync strategies. Only the fixed step
        one is registered; the other two exist as signatures without a body
        and are intentionally not selectable by a parameter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..utils import drop

SYNC_FIXED_STEP = "fixed_step"

SOURCE_CAMERA = "camera"
SOURCE_ARM = "arm"
SOURCE_GRIP = "grip"


def source_table(camera_names: tuple[str, ...],
                 arm_names: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """Build the required source table of one topology.

    Args:
        camera_names (tuple): Camera names from the device hub.
        arm_names (tuple): Arm names from the device hub.

    Returns:
        dict: Label to ``(kind, device name)``; the label is what the drop log
        names, for example ``right_wrist_rgb``.
    """
    table: dict[str, tuple[str, str]] = {}
    for name in camera_names:
        table[f"{name}_rgb"] = (SOURCE_CAMERA, name)
    for name in arm_names:
        table[f"{name}_arm"] = (SOURCE_ARM, name)
        table[f"{name}_grip"] = (SOURCE_GRIP, name)
    return table


def source_stamp(observation: dict, kind: str,
                 name: str) -> int | None:
    """Read the stamp of one required source.

    Args:
        observation (dict): Grouped dictionary of the robot.
        kind (str): ``SOURCE_CAMERA``, ``SOURCE_ARM`` or ``SOURCE_GRIP``.
        name (str): Device name.

    Returns:
        int | None: Stamp of the source, ``None`` when the source is missing.
    """
    if kind == SOURCE_CAMERA:
        camera = observation["camera"].get(name)
        if camera is None:
            return None
        return camera.rgb.ts_ns
    arm = observation["arm"].get(name)
    if arm is None or arm.stamp_ns is None:
        return None
    if kind == SOURCE_ARM:
        return None if arm.arm_state is None else arm.stamp_ns
    return None if arm.grip_state is None else arm.stamp_ns


def sync_fixed_step(state: dict, observation: dict, now_ns: int) -> int:
    """Advance the frame stamp by exactly one observation period.

    The first frame uses the current time, every later frame is the previous
    frame plus one period. Only successful frames call this function, so the
    spacing never drifts because of dropped frames.

    Args:
        state (dict): Sync state that holds the last frame stamp.
        observation (dict): Grouped observation; unused here.
        now_ns (int): Current time.

    Returns:
        int: Frame stamp in nanoseconds.
    """
    if state.get("frame_ns") is None:
        state["frame_ns"] = now_ns
    else:
        state["frame_ns"] += state["period_ns"]
    return state["frame_ns"]


def sync_by_primary(state: dict, observation: dict,
                    now_ns: int) -> int:
    """Placeholder for the primary source strategy.

    Args:
        state (dict): Sync state.
        observation (dict): Grouped observation.
        now_ns (int): Current time.

    Raises:
        NotImplementedError: Always; the strategy is not implemented yet.
    """
    raise NotImplementedError("not implemented yet")


def sync_by_weighted_average(state: dict, observation: dict,
                             now_ns: int) -> int:
    """Placeholder for the weighted average strategy.

    Args:
        state (dict): Sync state.
        observation (dict): Grouped observation.
        now_ns (int): Current time.

    Raises:
        NotImplementedError: Always; the strategy is not implemented yet.
    """
    raise NotImplementedError("not implemented yet")


SYNC_STRATEGIES: dict[str, Callable[[dict, dict, int], int]] = {
    SYNC_FIXED_STEP: sync_fixed_step,
}


class ObservationBase(ABC):
    """Base class of the observation assemblers.

    Attributes:
        camera_names (tuple): Camera names of the topology.
        arm_names (tuple): Arm names of the topology.
    """

    def __init__(self,
                 camera_names: tuple[str, ...],
                 arm_names: tuple[str, ...],
                 freshness_ns: int,
                 period_ns: int,
                 now_fn: Callable[[], int],
                 sync_fn: Callable[[dict, dict, int], int],
                 logger=None) -> None:
        """Store the source lists and the injected clock.

        Args:
            camera_names (tuple): Camera names of the topology.
            arm_names (tuple): Arm names of the topology.
            freshness_ns (int): Age above which a source counts as stale.
            period_ns (int): Observation period used by the sync strategy.
            now_fn (Callable): Clock injected by the node.
            sync_fn (Callable): Time sync strategy from the table.
            logger (Any | None): Logger used for drop lines.
        """
        self.camera_names = tuple(camera_names)
        self.arm_names = tuple(arm_names)
        self.__freshness_ns = int(freshness_ns)
        self.__now_fn = now_fn
        self.__logger = logger
        self.__sources = source_table(
            camera_names=self.camera_names,
            arm_names=self.arm_names,
        )
        self.__sync_fn = sync_fn
        self.__sync_state: dict = {"frame_ns": None, "period_ns": period_ns}
        self.__dropped = 0

    @property
    def sources(self) -> dict[str, tuple[str, str]]:
        """Required sources of this topology.

        Returns:
            dict: Label to ``(kind, device name)``.
        """
        return dict(self.__sources)

    @abstractmethod
    def push_obs(self, observation: dict) -> None:
        """Store the newest grouped observation.

        Args:
            observation (dict): Output of ``Robot.read_obs()``.
        """

    @abstractmethod
    def get_obs(self) -> dict | None:
        """Assemble one frame.

        Returns:
            dict | None: ``{"stamp_ns", "camera", "arm"}`` or ``None`` when a
            required source is missing or too old.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget the stored samples and the frame stamp."""

    def _bad_sources(self, observation: dict) -> list[str]:
        """Describe every source that is missing or too old.

        Args:
            observation (dict): Grouped observation.

        Returns:
            list: One sentence per unusable source.
        """
        now_ns = self.__now_fn()
        limit_ms = self.__freshness_ns / 1e6
        problems = []
        for label, (kind, name) in self.__sources.items():
            stamp_ns = source_stamp(
                observation=observation,
                kind=kind,
                name=name,
            )
            if stamp_ns is None:
                problems.append(f"{label} has no sample")
            elif now_ns - stamp_ns > self.__freshness_ns:
                age_ms = (now_ns - stamp_ns) / 1e6
                problems.append(
                    f"{label} is stale: {age_ms:.1f}ms > {limit_ms:.1f}ms")
        return problems

    def _report_drop(self, problems: list[str]) -> None:
        """Log one line that lists every unusable source.

        Args:
            problems (list): Sentences produced by :meth:`_bad_sources`.
        """
        self.__dropped += 1
        total = len(self.__sources)
        cause = (f"{len(problems)} of {total} sources unusable - "
                 + "；".join(problems))
        drop(
            logger=self.__logger,
            point="obs.assemble",
            obj="whole observation dropped",
            cause=cause,
            reason="multi",
            count=self.__dropped,
        )

    def _next_stamp(self, observation: dict) -> int:
        """Ask the sync strategy for the stamp of this frame.

        Args:
            observation (dict): Grouped observation.

        Returns:
            int: Frame stamp in nanoseconds.
        """
        return self.__sync_fn(self.__sync_state, observation,
                              self.__now_fn())

    def _reset_sync(self) -> None:
        """Forget the frame stamp, so the next frame restarts the sequence."""
        self.__sync_state["frame_ns"] = None


# The two unimplemented strategies stay reachable by name but are not part of
# the selectable table, so no parameter can turn them on by accident.
UNREGISTERED_SYNC = (sync_by_primary, sync_by_weighted_average)
