"""Unit tests for ActionWindowQuery target timestamp and query offset."""

import numpy as np
import pytest

from hex_ros_policy_client.process.action.action_window_query import (
    ActionWindowQuery,
    WindowQueryParam,
)
from hex_ros_policy_client.struct.cmd_struct import (
    ArmCommand,
    ControlIntent,
    GripCommand,
)

BASE_NS = 1_000_000_000
STEP_MS = 20
STEP_NS = STEP_MS * 1_000_000


class Clock:
    """Deterministic nanosecond clock for tests."""

    def __init__(self, start_ns: int = BASE_NS) -> None:
        """Initialize clock with a starting timestamp.

        Args:
            start_ns (int): Starting time in nanoseconds.
        """
        self.now_ns = start_ns

    def __call__(self) -> int:
        """Return current clock time in nanoseconds.

        Returns:
            int: Current timestamp in nanoseconds.
        """
        return self.now_ns


class LoggerStub:
    """Logger test double recording debug and warning messages."""

    def __init__(self) -> None:
        """Initialize message storage."""
        self.debugs: list[str] = []
        self.warnings: list[str] = []

    def logd(self, msg: str) -> None:
        """Record debug message.

        Args:
            msg (str): Message text.
        """
        self.debugs.append(msg)

    def logw(self, msg: str) -> None:
        """Record warning message.

        Args:
            msg (str): Message text.
        """
        self.warnings.append(msg)


def build_processor(clock: Clock) -> ActionWindowQuery:
    """Construct an ActionWindowQuery processor with standard test params.

    Args:
        clock (Clock): Clock provider for now_fn.

    Returns:
        ActionWindowQuery: Initialized processor.
    """
    param = WindowQueryParam(
        arm_names=("left",),
        now_fn=clock,
        anchor_tolerance_ns=50_000_000,
        logger=LoggerStub(),
    )
    return ActionWindowQuery(params=param)


def make_chunk(
    stamp_ns: int = BASE_NS,
    step_ms: int = STEP_MS,
    steps: int = 5,
) -> dict:
    """Construct a minimal valid action chunk dictionary.

    Args:
        stamp_ns (int): Base timestamp in nanoseconds.
        step_ms (int): Step interval in milliseconds.
        steps (int): Number of steps to generate.

    Returns:
        dict: Standard chunk payload.
    """
    return {
        "stamp_ns": stamp_ns,
        "chunk_step_interval_ms": step_ms,
        "action_mode": "abs",
        "arm_ctrl_mode": "pos",
        "arm": {
            "left": [[float(i), float(i)] for i in range(steps)],
        },
        "grip": {
            "left": [[float(i)] for i in range(steps)],
        },
    }


def test_control_intent_stamp_ns_field() -> None:
    """ControlIntent supports optional stamp_ns with default None."""
    intent_default = ControlIntent()
    assert intent_default.stamp_ns is None
    intent_stamped = ControlIntent(stamp_ns=123456789)
    assert intent_stamped.stamp_ns == 123456789


def test_get_target_empty_queue_returns_none() -> None:
    """Empty processor returns None before any chunk is pushed."""
    clock = Clock(BASE_NS)
    processor = build_processor(clock)
    assert processor.get_target() is None


def test_get_target_default_offset_applies_chunk_step_interval() -> None:
    """When query_offset_ns is None, default query is now_ns + step_ns."""
    clock = Clock(BASE_NS)
    processor = build_processor(clock)
    processor.push_chunk(make_chunk(stamp_ns=BASE_NS, step_ms=STEP_MS))

    # now_ns = BASE_NS.
    # Default query_offset_ns=None -> query_ns = BASE_NS + STEP_NS
    # At query_ns = BASE_NS + STEP_NS, window is [BASE_NS, BASE_NS + 2*STEP_NS]
    # Points 0, 1, 2 fall in window.
    target = processor.get_target()
    assert target is not None
    assert target.stamp_ns == BASE_NS + STEP_NS
    assert isinstance(target.arm["left"], ArmCommand)
    assert isinstance(target.grip["left"], GripCommand)


def test_get_target_custom_query_offset_applied() -> None:
    """Custom query_offset_ns specifies explicit offset from now_ns."""
    clock = Clock(BASE_NS)
    processor = build_processor(clock)
    processor.push_chunk(make_chunk(stamp_ns=BASE_NS, step_ms=STEP_MS))

    custom_offset_ns = 2 * STEP_NS
    target = processor.get_target(query_offset_ns=custom_offset_ns)
    assert target is not None
    assert target.stamp_ns == BASE_NS + custom_offset_ns


def test_get_target_zero_query_offset_queries_at_current_time() -> None:
    """When query_offset_ns=0, query is evaluated at now_ns."""
    clock = Clock(BASE_NS)
    processor = build_processor(clock)
    processor.push_chunk(make_chunk(stamp_ns=BASE_NS, step_ms=STEP_MS))

    target = processor.get_target(query_offset_ns=0)
    assert target is not None
    assert target.stamp_ns == BASE_NS


def test_control_intent_preserves_stamp_ns_across_copies() -> None:
    """ControlIntent stamp_ns is copied when duplicating intents."""
    original = ControlIntent(
        arm={"left": ArmCommand(ctrl_mode="pos", pos=[0.1])},
        grip={"left": GripCommand(ctrl_mode="pos", pos=[0.5])},
        stamp_ns=BASE_NS + 12345,
    )
    duplicate = ControlIntent(
        arm=dict(original.arm),
        grip=dict(original.grip),
        stamp_ns=original.stamp_ns,
    )
    assert duplicate.stamp_ns == BASE_NS + 12345
    assert duplicate.arm["left"].pos == [0.1]
    assert duplicate.grip["left"].pos == [0.5]

