################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Drop-log grammar, rate limiting and counters.

Every line produced here has the same shape::

    [<module.point>/<reason>] <object><action>: <cause> (<counter>)

The module is a single file with no dependency on the rest of the package, on
ROS or on the environment, so it can be copied into another project as is. It
holds no business vocabulary: ``reason`` is only checked for shape, its value
set belongs to the caller.

The caller passes the object together with its action as ``obj``, so the
library only concatenates and never invents text.

Attributes:
    RATE_LIMIT_S: Minimum interval between two lines of the same key.
    MAX_LINE_WIDTH: Maximum width of one line, in characters.
    REASON_PATTERN: Shape of a valid ``reason`` token.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

RATE_LIMIT_S = 1.0
MAX_LINE_WIDTH = 160
REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_FOLD = " ... ({} chars folded)"

# Replaced by the tests to make the throttling deterministic.
_monotonic = time.monotonic


class _Entry:
    """Throttling state of one (point, reason) pair.

    Attributes:
        last_emit_s (float | None): Clock reading of the last emitted line.
        suppressed (int): Lines suppressed since the last emitted line.
    """

    __slots__ = ("last_emit_s", "suppressed")

    def __init__(self) -> None:
        """Create an entry that has never emitted anything."""
        self.last_emit_s: float | None = None
        self.suppressed = 0


_entries: dict[tuple[str, str], _Entry] = {}


def drop(logger: Any,
         point: str,
         obj: str,
         cause: str,
         *,
         reason: str,
         level: int = logging.WARNING,
         count: int = 0) -> None:
    """Emit one drop log line, throttled per ``(point, reason)``.

    Args:
        logger: Logger-like object exposing ``log(level, message)``.
        point (str): Drop point of the form ``<module>.<point>``.
        obj (str): Data object and what happened to it, supplied by the
            caller in the project language.
        cause (str): One checkable sentence with paired values and units.
        reason (str): Why the data was dropped, lower snake case, at most 32
            characters.
        level (int): Log level, by default a warning.
        count (int): Cumulative drops reported by the caller; only shown when
            it reached 2.

    Raises:
        ValueError: If ``reason`` does not match the required shape or if one
            of the text arguments is empty.
    """
    _validate(
        point=point,
        obj=obj,
        cause=cause,
        reason=reason,
    )
    key = (point, reason)
    entry = _entries.get(key)
    if entry is None:
        entry = _Entry()
        _entries[key] = entry
    now = _monotonic()
    if entry.last_emit_s is not None:
        if now - entry.last_emit_s < RATE_LIMIT_S:
            entry.suppressed += 1
            return
    entry.last_emit_s = now
    suppressed = entry.suppressed
    entry.suppressed = 0
    _emit(
        logger=logger,
        level=level,
        message=_compose(
            point=point,
            reason=reason,
            obj=obj,
            cause=cause,
            count=count,
            suppressed_count=suppressed,
        ),
    )


def suppressed(point: str, reason: str) -> int:
    """Return how many lines of one key are currently throttled away.

    Args:
        point (str): Drop point passed to :func:`drop`.
        reason (str): Reason passed to :func:`drop`.

    Returns:
        int: Number of suppressed lines since the last emitted line.
    """
    entry = _entries.get((point, reason))
    return 0 if entry is None else entry.suppressed


def _emit(logger: Any, level: int, message: str) -> None:
    """Emit one line through either logger flavour.

    The standard library orders the arguments as ``(level, message)``, while
    the ROS logger orders them as ``(message, severity)``. Both are accepted
    here so the same line reaches a test logger and a live node.

    Args:
        logger (Any): Standard library logger or ROS logger.
        level (int): Standard library level; the ROS logger shares the values.
        message (str): Composed line.
    """
    if type(logger).__module__.startswith("rclpy"):
        logger.log(message, level)
        return
    logger.log(level, message)


def _validate(point: str, obj: str, cause: str, reason: str) -> None:
    """Check the shape of the arguments before anything is emitted.

    Args:
        point (str): Drop point.
        obj (str): Data object text.
        cause (str): Cause sentence.
        reason (str): Reason token.

    Raises:
        ValueError: If the reason shape is wrong or a text is empty.
    """
    if not REASON_PATTERN.match(reason):
        raise ValueError(f"invalid reason token: {reason!r}")
    for name, value in (("point", point), ("obj", obj), ("cause", cause)):
        if not value or not value.strip():
            raise ValueError(f"empty {name}")


def _compose(point: str,
             reason: str,
             obj: str,
             cause: str,
             count: int,
             suppressed_count: int) -> str:
    """Build the final line, folding the cause when it gets too long.

    Args:
        point (str): Drop point.
        reason (str): Reason token.
        obj (str): Data object text.
        cause (str): Cause sentence.
        count (int): Cumulative drops reported by the caller.
        suppressed_count (int): Lines throttled away since the last line.

    Returns:
        str: One line that follows the grammar and fits the width limit when
        the fixed part allows it.
    """
    prefix = f"[{point}/{reason}] "
    notes = []
    if count >= 2:
        notes.append(f"total {count}")
    if suppressed_count >= 1:
        notes.append(f"{suppressed_count} suppressed")
    tail = f" ({', '.join(notes)})" if notes else ""
    budget = MAX_LINE_WIDTH - len(prefix) - len(obj) - 1 - len(tail)
    return prefix + obj + ": " + _fold_cause(
        budget=budget,
        cause=cause,
    ) + tail


def _fold_cause(budget: int, cause: str) -> str:
    """Truncate a cause so that the line stays inside the width limit.

    Args:
        budget (int): Characters available for the cause.
        cause (str): Cause sentence.

    Returns:
        str: The cause itself, or its head followed by the folded count.
    """
    if len(cause) <= budget:
        return cause
    omitted = len(cause)
    while True:
        marker = _FOLD.format(omitted)
        keep = max(budget - len(marker), 0)
        actual = len(cause) - keep
        if actual == omitted or keep == 0:
            return cause[:keep] + _FOLD.format(actual)
        omitted = actual


def _reset_state() -> None:
    """Forget every throttling counter; used by the tests."""
    _entries.clear()
