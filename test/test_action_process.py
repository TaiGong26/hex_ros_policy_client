################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the action chunk processor."""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.process.time_integration_chunk import (
    TimeIntegrationChunk,
)
from hex_ros_policy_client.struct.cmd_struct import TARGET_ARM, TARGET_GRIP
from hex_ros_policy_client.utils.common_util.logkit import (
    _reset_state as reset_log,
)
from hex_ros_policy_client.utils import limit_to_step, linear_interpolate

from .fakes import Capture

NS_PER_MS = 1_000_000
STEP_MS = 33


class Clock:
    """Clock stand-in driven by the test.

    Attributes:
        now_ns (int): Current time.
    """

    def __init__(self, now_ns: int = 0) -> None:
        """Create a clock at the given instant.

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


def chunk(stamp_ns: int, values: list, step_ms: int | None = STEP_MS,
          action_mode: str = "joint") -> dict:
    """Build one decoded chunk for a single arm and gripper.

    Args:
        stamp_ns (int): Stamp of the chunk.
        values (list): One value per step.
        step_ms (int | None): Step of the chunk, ``None`` to use the default.
        action_mode (str): Mode of the steps, ``joint`` by default.

    Returns:
        dict: Chunk as the codec delivers it.
    """
    return {
        "stamp_ns": stamp_ns,
        "chunk_step_interval_ms": step_ms,
        "action_mode": action_mode,
        "arm_ctrl_mode": "pos",
        "arm": {"left": [[value, value] for value in values]},
        "grip": {"left": [[value] for value in values]},
    }


def build(clock: Clock, logger=None, interpolator=linear_interpolate,
          capacity: int = 100,
          average_strategy: str = "none") -> TimeIntegrationChunk:
    """Create a processor for one arm.

    Args:
        clock (Clock): Injected clock.
        logger (Capture | None): Logger for drop lines.
        interpolator (Any): Injected interpolation function.
        capacity (int): Queue capacity.

    Returns:
        TimeIntegrationChunk: Processor under test.
    """
    return TimeIntegrationChunk(("left",), clock, interpolator,
                                capacity=capacity,
                                average_strategy=average_strategy,
                                logger=logger or Capture())


def test_empty_queue_returns_none_at_debug_level() -> None:
    """No chunk means no target, which is normal and not a warning."""
    reset_log()
    logger = Capture()
    processor = build(Clock(), logger)
    assert processor.get_target() is None
    level, line = logger.records[0]
    assert level < 30
    assert line.startswith("[chunk.target/queue_empty] ")


def test_target_interpolates_inside_a_step() -> None:
    """Halfway through a step the target is halfway between two samples."""
    clock = Clock(10_000_000_000)
    processor = build(clock)
    processor.push_chunk(chunk(clock.now_ns, [0.0, 1.0, 2.0]))
    clock.now_ns += STEP_MS * NS_PER_MS // 2
    target = processor.get_target()
    assert target[TARGET_ARM]["left"].pos == pytest.approx([0.5, 0.5])
    assert target[TARGET_GRIP]["left"].pos == pytest.approx([0.5])


def test_target_uses_the_last_step_when_the_chunk_ends() -> None:
    """The final sample stays valid until the chunk is exhausted."""
    clock = Clock(10_000_000_000)
    processor = build(clock)
    processor.push_chunk(chunk(clock.now_ns, [0.0, 1.0]))
    clock.now_ns += 2 * STEP_MS * NS_PER_MS - 1
    target = processor.get_target()
    assert target[TARGET_ARM]["left"].pos == pytest.approx([1.0, 1.0])


def test_exhausted_chunk_yields_none() -> None:
    """After the last step the chunk covers nothing."""
    clock = Clock(10_000_000_000)
    processor = build(clock)
    processor.push_chunk(chunk(clock.now_ns, [0.0, 1.0]))
    clock.now_ns += 3 * STEP_MS * NS_PER_MS
    assert processor.get_target() is None


def test_overlap_is_weighted_towards_the_newest_chunk() -> None:
    """Two overlapping chunks average with the configured decay."""
    clock = Clock(10_000_000_000)
    processor = build(clock, average_strategy="exponential")
    processor.push_chunk(chunk(clock.now_ns, [1.0, 1.0, 1.0]))
    clock.now_ns += STEP_MS * NS_PER_MS
    processor.push_chunk(chunk(clock.now_ns, [2.0, 2.0, 2.0]))
    target = processor.get_target()
    expected = (0.5 * 1.0 + 1.0 * 2.0) / 1.5
    assert target[TARGET_ARM]["left"].pos == pytest.approx([expected] * 2)
    assert processor.last_weights == {1: 0.5, 2: 1.0}


def test_default_step_is_used_when_the_frame_omits_it() -> None:
    """A frame without a step is accepted instead of rejected."""
    clock = Clock(10_000_000_000)
    processor = build(clock)
    processor.push_chunk(chunk(clock.now_ns, [0.0, 1.0], step_ms=None))
    clock.now_ns += STEP_MS * NS_PER_MS
    target = processor.get_target()
    assert target[TARGET_ARM]["left"].pos == pytest.approx([1.0, 1.0])


def test_queue_overflow_drops_the_oldest() -> None:
    """The queue keeps its capacity and reports the dropped sequence."""
    reset_log()
    logger = Capture()
    clock = Clock(10_000_000_000)
    processor = build(clock, logger, capacity=2)
    for seq in range(1, 4):
        processor.push_chunk(chunk(clock.now_ns, [float(seq)]))
    assert "chunk.queue/full" in logger.messages()[-1]
    assert "seq=1" in logger.messages()[-1]


def test_bad_chunk_is_dropped() -> None:
    """A chunk without the arm group is rejected without raising."""
    reset_log()
    logger = Capture()
    processor = build(Clock(1), logger)
    processor.push_chunk({"stamp_ns": 0, "grip": {}})
    assert "chunk.push/schema_invalid" in logger.messages()[0]


def test_unknown_arm_mode_is_dropped() -> None:
    """A mode the action layer does not know is dropped, never raised."""
    reset_log()
    logger = Capture()
    processor = build(Clock(1), logger)
    bad = chunk(0, [0.0])
    bad["arm_ctrl_mode"] = "bogus"
    processor.push_chunk(bad)
    assert "chunk.push/schema_invalid" in logger.messages()[0]


def test_ee_arm_mode_is_dropped_as_unsupported() -> None:
    """A known mode this stage cannot dispatch is dropped with its reason."""
    reset_log()
    logger = Capture()
    processor = build(Clock(1), logger)
    ee = chunk(0, [0.0])
    ee["arm_ctrl_mode"] = "ee"
    processor.push_chunk(ee)
    assert "chunk.mode/unsupported_mode" in logger.messages()[0]


def test_injected_interpolator_changes_the_behaviour() -> None:
    """Swapping the interpolation function swaps the result."""
    clock = Clock(10_000_000_000)
    processor = build(clock, interpolator=lambda start, target, elapsed,
                      duration: np.asarray(target, dtype=np.float64))
    processor.push_chunk(chunk(clock.now_ns, [0.0, 4.0]))
    clock.now_ns += STEP_MS * NS_PER_MS // 2
    target = processor.get_target()
    assert target[TARGET_ARM]["left"].pos == pytest.approx([4.0, 4.0])


def test_reset_clears_the_queue() -> None:
    """Reset drops every stored chunk."""
    clock = Clock(10_000_000_000)
    processor = build(clock)
    processor.push_chunk(chunk(clock.now_ns, [0.0, 1.0]))
    processor.reset()
    assert processor.get_target() is None
    assert processor.last_weights == {}


def test_rejects_bad_construction_arguments() -> None:
    """Capacity and decay are validated at construction."""
    with pytest.raises(ValueError):
        TimeIntegrationChunk(("left",), Clock(), linear_interpolate,
                             capacity=0)
    with pytest.raises(ValueError):
        TimeIntegrationChunk(("left",), Clock(), linear_interpolate,
                             decay=0.0)


def test_limited_target_never_jumps() -> None:
    """The step limit keeps the increment inside the threshold."""
    target = np.array([1.0, 1.0])
    measured = np.zeros(2)
    limited = limit_to_step(target, measured, 0.1)
    assert np.max(np.abs(limited - measured)) == pytest.approx(0.1)
