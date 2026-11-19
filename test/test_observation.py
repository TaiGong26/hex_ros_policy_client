################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the observation assembler."""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.process.history_observation import (
    HistoryObservation,
)
from hex_ros_policy_client.process.observation_base import (
    SYNC_FIXED_STEP,
    SYNC_STRATEGIES,
    UNREGISTERED_SYNC,
    source_table,
)
from hex_ros_policy_client.struct.device_struct import (
    ArmObservation,
    CameraObservation,
    ImageFrame,
)
from hex_ros_policy_client.utils.common_util.logkit import (
    _reset_state as reset_log,
)

from .fakes import Capture

NS_PER_MS = 1_000_000
PERIOD_NS = 33 * NS_PER_MS
FRESHNESS_NS = 40 * NS_PER_MS


class Clock:
    """Clock stand-in driven by the test.

    Attributes:
        now_ns (int): Current time.
    """

    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        """Create a clock at a known instant.

        Args:
            now_ns (int): Initial time.
        """
        self.now_ns = now_ns

    def __call__(self) -> int:
        """Return the current time.

        Returns:
            int: Current time in nanoseconds.
        """
        return self.now_ns


def camera(name: str, stamp_ns: int) -> CameraObservation:
    """Build a colour only camera observation.

    Args:
        name (str): Unused label kept for readability.
        stamp_ns (int): Capture time.

    Returns:
        CameraObservation: Observation with a one pixel frame.
    """
    frame = ImageFrame(ts_ns=stamp_ns, encoding="jpeg", width=1, height=1,
                       data=np.zeros(1, dtype=np.uint8))
    return CameraObservation(rgb=frame)


def arm(stamp_ns: int) -> ArmObservation:
    """Build an arm observation with both parts filled.

    Args:
        stamp_ns (int): Stamp shared by arm and gripper.

    Returns:
        ArmObservation: Observation with a fake arm and gripper state.
    """
    return ArmObservation(stamp_ns=stamp_ns, arm_state=_arm_state(),
                          grip_state=_grip_state())


def _arm_state():
    """Build a minimal arm state.

    Returns:
        ArmState: State with two joints.
    """
    from hex_ros_policy_client.struct.device_struct import ArmState
    from hex_ros_policy_client.struct.geom_struct import Pose
    return ArmState(arm_pos=np.zeros(2), arm_vel=np.zeros(2),
                    arm_eff=np.zeros(2), ee_base=Pose())


def _grip_state():
    """Build a minimal gripper state.

    Returns:
        GripState: State with one joint.
    """
    from hex_ros_policy_client.struct.device_struct import GripState
    return GripState(grip_pos=np.zeros(1), grip_vel=np.zeros(1),
                     grip_eff=np.zeros(1))


def observation(stamp_ns: int, arms=("left",), cameras=("fixed",)) -> dict:
    """Build a grouped observation dictionary.

    Args:
        stamp_ns (int): Stamp used for every source.
        arms (tuple): Arm names to fill.
        cameras (tuple): Camera names to fill.

    Returns:
        dict: Grouped observation.
    """
    return {
        "camera": {name: camera(name, stamp_ns) for name in cameras},
        "arm": {name: arm(stamp_ns) for name in arms},
    }


def build(tmp_path, clock: Clock, arms=("left",), cameras=("fixed",),
          logger=None) -> HistoryObservation:
    """Create an assembler for a small topology.

    Args:
        tmp_path: Unused path kept for symmetry with other builders.
        clock (Clock): Injected clock.
        arms (tuple): Arm names.
        cameras (tuple): Camera names.
        logger (Capture | None): Logger used for drop lines.

    Returns:
        HistoryObservation: Assembler under test.
    """
    return HistoryObservation(cameras, arms, FRESHNESS_NS, PERIOD_NS, clock,
                              SYNC_STRATEGIES[SYNC_FIXED_STEP],
                              logger or Capture())


def test_source_table_follows_the_topology() -> None:
    """Required sources are derived from the name lists."""
    table = source_table(("fixed",), ("left",))
    assert set(table) == {"fixed_rgb", "left_arm", "left_grip"}
    assert table["left_grip"] == ("grip", "left")


def test_no_observation_yields_none() -> None:
    """Before the first sample nothing is assembled."""
    assert build(None, Clock()).get_obs() is None


def test_all_sources_present() -> None:
    """A complete observation produces a frame with a stamp."""
    clock = Clock()
    assembler = build(None, clock)
    assembler.push_obs(observation(clock.now_ns))
    frame = assembler.get_obs()
    assert frame is not None
    assert frame["stamp_ns"] == clock.now_ns
    assert set(frame) == {"stamp_ns", "camera", "arm"}


def test_stale_source_discards_the_frame() -> None:
    """One old source discards the whole frame and names the source."""
    reset_log()
    clock = Clock()
    logger = Capture()
    assembler = build(None, clock, logger=logger)
    assembler.push_obs(observation(clock.now_ns))
    clock.now_ns += 86 * NS_PER_MS
    assert assembler.get_obs() is None
    line = logger.messages()[0]
    assert line.startswith("[obs.assemble/multi] ")
    assert "fixed_rgb is stale: 86.0ms > 40.0ms" in line
    assert "left_arm is stale: 86.0ms > 40.0ms" in line


def test_missing_source_discards_the_frame() -> None:
    """A source that never arrived is reported separately from a stale one."""
    reset_log()
    clock = Clock()
    logger = Capture()
    assembler = build(None, clock, logger=logger)
    incomplete = observation(clock.now_ns)
    incomplete["arm"]["left"] = ArmObservation()
    assembler.push_obs(incomplete)
    assert assembler.get_obs() is None
    line = logger.messages()[0]
    assert "left_arm has no sample" in line
    assert "left_grip has no sample" in line
    assert "fixed_rgb" not in line


def test_depth_is_not_required() -> None:
    """A camera without depth still counts as a usable source."""
    clock = Clock()
    assembler = build(None, clock)
    assembler.push_obs(observation(clock.now_ns))
    assert assembler.get_obs() is not None


def test_frame_stamp_steps_by_one_period() -> None:
    """Consecutive frames are exactly one period apart."""
    clock = Clock()
    assembler = build(None, clock)
    stamps = []
    for _ in range(3):
        assembler.push_obs(observation(clock.now_ns))
        stamps.append(assembler.get_obs()["stamp_ns"])
        clock.now_ns += 7 * NS_PER_MS
    assert stamps[1] - stamps[0] == PERIOD_NS
    assert stamps[2] - stamps[1] == PERIOD_NS


def test_dropped_frame_does_not_advance_the_stamp() -> None:
    """A discarded frame leaves the sequence untouched."""
    clock = Clock()
    assembler = build(None, clock)
    assembler.push_obs(observation(clock.now_ns))
    first = assembler.get_obs()["stamp_ns"]
    clock.now_ns += 500 * NS_PER_MS
    assert assembler.get_obs() is None
    assembler.push_obs(observation(clock.now_ns))
    assert assembler.get_obs()["stamp_ns"] == first + PERIOD_NS


def test_reset_forgets_sample_and_stamp() -> None:
    """Reset restarts the sequence with the next frame."""
    clock = Clock()
    assembler = build(None, clock)
    assembler.push_obs(observation(clock.now_ns))
    assembler.get_obs()
    assembler.reset()
    assert assembler.get_obs() is None
    assembler.push_obs(observation(clock.now_ns))
    assert assembler.get_obs()["stamp_ns"] == clock.now_ns


def test_unimplemented_strategies_are_not_selectable() -> None:
    """Only the fixed step strategy can be selected by configuration."""
    assert set(SYNC_STRATEGIES) == {SYNC_FIXED_STEP}
    for strategy in UNREGISTERED_SYNC:
        with pytest.raises(NotImplementedError):
            strategy({}, {}, 0)
