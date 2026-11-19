################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the drop-log helper."""

from __future__ import annotations

import importlib.util
import logging
import pathlib
import shutil
import sys
import tempfile

import pytest

from hex_ros_policy_client.utils.common_util import logkit


class Capture:
    """Logger stand-in that keeps every emitted line.

    Attributes:
        lines (list): Pairs of (level, message).
    """

    def __init__(self) -> None:
        """Create an empty capture."""
        self.lines: list[tuple[int, str]] = []

    def log(self, level: int, message: str) -> None:
        """Record one line.

        Args:
            level (int): Log level.
            message (str): Rendered message.
        """
        self.lines.append((level, message))


class Clock:
    """Monotonic clock stand-in driven by the test.

    Attributes:
        now_s (float): Current reading in seconds.
    """

    def __init__(self) -> None:
        """Create a clock at zero."""
        self.now_s = 0.0

    def __call__(self) -> float:
        """Return the current reading.

        Returns:
            float: Current reading in seconds.
        """
        return self.now_s


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Install a controllable clock and clear the counters.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch the module.

    Returns:
        Clock: Installed clock.
    """
    logkit._reset_state()
    fake = Clock()
    monkeypatch.setattr(logkit, "_monotonic", fake)
    return fake


def _drop(capture: Capture, **kwargs) -> None:
    """Emit one drop line with the default key of the tests.

    Args:
        capture (Capture): Destination logger.
        **kwargs: Overrides forwarded to :func:`logkit.drop`.
    """
    args = {
        "logger": capture,
        "point": "ctrl.dispatch",
        "obj": "no command sent this tick",
        "cause": "observation is stale: 95.2ms > 40.0ms",
        "reason": "obs_stale",
        "count": 8,
    }
    args.update(kwargs)
    logkit.drop(**args)


def test_grammar(clock: Clock) -> None:
    """A line starts with the prefix and carries the object and the cause."""
    capture = Capture()
    _drop(capture)
    assert len(capture.lines) == 1
    level, line = capture.lines[0]
    assert level == logging.WARNING
    assert line.startswith("[ctrl.dispatch/obs_stale] ")
    assert "no command sent this tick" in line
    assert "observation is stale: 95.2ms > 40.0ms" in line
    assert line.endswith("(total 8)")


def test_count_hidden_when_below_two(clock: Clock) -> None:
    """A single drop does not print a counter."""
    capture = Capture()
    _drop(capture, count=1)
    assert capture.lines[0][1].endswith("40.0ms")


def test_rate_limit_suppresses_and_counts(clock: Clock) -> None:
    """Lines inside the window are suppressed but counted."""
    capture = Capture()
    _drop(capture, count=2)
    for _ in range(5):
        clock.now_s += 0.1
        _drop(capture, count=2)
    assert len(capture.lines) == 1
    assert logkit.suppressed("ctrl.dispatch", "obs_stale") == 5


def test_rate_limit_merges_suppressed_on_next_emit(clock: Clock) -> None:
    """The next emitted line reports how many lines were throttled away."""
    capture = Capture()
    _drop(capture, count=2)
    for _ in range(5):
        clock.now_s += 0.1
        _drop(capture, count=2)
    clock.now_s += logkit.RATE_LIMIT_S
    _drop(capture, count=3)
    assert len(capture.lines) == 2
    assert capture.lines[1][1].endswith("(total 3, 5 suppressed)")
    assert logkit.suppressed("ctrl.dispatch", "obs_stale") == 0


def test_different_reason_is_not_throttled(clock: Clock) -> None:
    """Throttling is per (point, reason) pair."""
    capture = Capture()
    _drop(capture)
    _drop(capture, reason="state_waiting")
    assert len(capture.lines) == 2


def test_long_line_is_folded(clock: Clock) -> None:
    """A long cause is folded instead of wrapped."""
    capture = Capture()
    _drop(capture, cause="stale " + "right wrist " * 40, count=7)
    line = capture.lines[0][1]
    assert len(line) <= logkit.MAX_LINE_WIDTH
    assert line.startswith("[ctrl.dispatch/obs_stale] ")
    assert "no command sent this tick" in line
    assert "chars folded" in line


def test_folded_count_matches_removed_characters(clock: Clock) -> None:
    """The folded marker reports the number of removed characters."""
    capture = Capture()
    cause = "observation is stale: " + "x" * 400
    _drop(capture, cause=cause)
    line = capture.lines[0][1]
    kept = line.split("：", 1)[1].split("…", 1)[0]
    folded = int(line.split(" ... (", 1)[1].split(" chars", 1)[0])
    assert len(kept) + folded == len(cause)


@pytest.mark.parametrize("reason", ["Obs_stale", "obs-stale", "", "a" * 33])
def test_bad_reason_shape_rejected(clock: Clock, reason: str) -> None:
    """Reasons that break the shape raise instead of being logged."""
    capture = Capture()
    with pytest.raises(ValueError):
        _drop(capture, reason=reason)
    assert capture.lines == []


def test_debug_level_is_passed_through(clock: Clock) -> None:
    """A normal-state drop keeps the DEBUG level given by the caller."""
    capture = Capture()
    _drop(capture, level=logging.DEBUG, reason="queue_empty", count=0)
    assert capture.lines[0][0] == logging.DEBUG


def test_file_is_portable() -> None:
    """The file works on its own after being copied away."""
    source = pathlib.Path(logkit.__file__)
    with tempfile.TemporaryDirectory() as tmp:
        copied = pathlib.Path(tmp) / "logkit.py"
        shutil.copyfile(source, copied)
        spec = importlib.util.spec_from_file_location("standalone_logkit",
                                                      copied)
        module = importlib.util.module_from_spec(spec)
        sys.modules["standalone_logkit"] = module
        try:
            spec.loader.exec_module(module)
            capture = Capture()
            module.drop(capture, "arm.left", "command dropped",
                        "ctrl_mode=ee is not in the mode table",
                        reason="unsupported_mode")
            assert capture.lines[0][1].startswith(
                "[arm.left/unsupported_mode] command dropped: ")
        finally:
            sys.modules.pop("standalone_logkit", None)
