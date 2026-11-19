################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Interpolation helpers shared by the action layer and the node.

The interpolator is injected at construction time instead of being hard coded
in the action processor, so switching the interpolation only changes the table
lookup. The same functions serve the return-to-stable behaviour: both places
interpolate an array between two values, only the meaning of the array differs.

Attributes:
    INTERPOLATORS: Name to function table; adding one entry is the only change
        needed to make a new interpolation selectable by parameter.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

EPSILON = 1e-9


def linear_interpolate(start: np.ndarray,
                       target: np.ndarray,
                       elapsed_ns: int,
                       duration_ns: int) -> np.ndarray:
    """Interpolate element wise between two vectors.

    Args:
        start (np.ndarray): Value at ``elapsed_ns <= 0``.
        target (np.ndarray): Value at ``elapsed_ns >= duration_ns``.
        elapsed_ns (int): Time since the interpolation started.
        duration_ns (int): Duration of the whole interpolation.

    Returns:
        np.ndarray: Interpolated value as float64.
    """
    ratio = _ratio(
        elapsed_ns=elapsed_ns,
        duration_ns=duration_ns,
    )
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(target, dtype=np.float64)
    return first + (last - first) * ratio


def slerp_interpolate(start: np.ndarray,
                      target: np.ndarray,
                      elapsed_ns: int,
                      duration_ns: int) -> np.ndarray:
    """Interpolate two quaternions along the shortest arc.

    Args:
        start (np.ndarray): Quaternion (x, y, z, w) at the start.
        target (np.ndarray): Quaternion (x, y, z, w) at the end.
        elapsed_ns (int): Time since the interpolation started.
        duration_ns (int): Duration of the whole interpolation.

    Returns:
        np.ndarray: Interpolated unit quaternion (x, y, z, w).

    Raises:
        ValueError: If either input is not a four element quaternion.
    """
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(target, dtype=np.float64)
    if first.shape != (4,) or last.shape != (4,):
        raise ValueError("slerp expects quaternions of shape (4,)")
    return _slerp(
        start=first,
        target=last,
        ratio=_ratio(
        elapsed_ns=elapsed_ns,
        duration_ns=duration_ns,
    ),
    )


def pose_interpolate(start: np.ndarray,
                     target: np.ndarray,
                     elapsed_ns: int,
                     duration_ns: int) -> np.ndarray:
    """Interpolate a pose: linear position, spherical orientation.

    Args:
        start (np.ndarray): Pose (px, py, pz, qx, qy, qz, qw) at the start.
        target (np.ndarray): Pose with the same layout at the end.
        elapsed_ns (int): Time since the interpolation started.
        duration_ns (int): Duration of the whole interpolation.

    Returns:
        np.ndarray: Interpolated pose with the same layout.

    Raises:
        ValueError: If either input is not a seven element pose.
    """
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(target, dtype=np.float64)
    if first.shape != (7,) or last.shape != (7,):
        raise ValueError("pose interpolation expects poses of shape (7,)")
    ratio = _ratio(
        elapsed_ns=elapsed_ns,
        duration_ns=duration_ns,
    )
    return np.concatenate([
        first[:3] + (last[:3] - first[:3]) * ratio,
        _slerp(
            start=first[3:],
            target=last[3:],
            ratio=ratio,
        ),
    ])


INTERPOLATORS: dict[str, Callable[..., np.ndarray]] = {
    "linear": linear_interpolate,
    "slerp": slerp_interpolate,
    "pose": pose_interpolate,
}


def get_interpolator(name: str) -> Callable[..., np.ndarray]:
    """Look up an interpolator by parameter value.

    Args:
        name (str): Key of :data:`INTERPOLATORS`.

    Returns:
        Callable: Interpolation function.

    Raises:
        ValueError: If the name is not registered in the table.
    """
    try:
        return INTERPOLATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown interpolation {name!r}, "
            f"expected one of {sorted(INTERPOLATORS)}") from None


def limit_to_step(target: np.ndarray,
                  measured: np.ndarray,
                  threshold: float) -> np.ndarray:
    """Limit one step of a target so that no joint jumps too far.

    Every element shares the same ratio, so the direction of the increment is
    preserved and the result is a scaled step instead of a per joint clamp.

    Args:
        target (np.ndarray): Target of this tick.
        measured (np.ndarray): Measured value read in this tick.
        threshold (float): Largest allowed increment, in the unit of the
            target.

    Returns:
        np.ndarray: Target to send, equal to ``target`` when the error is
        already small enough.

    Raises:
        ValueError: If the threshold is not positive.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    wanted = np.asarray(target, dtype=np.float64)
    current = np.asarray(measured, dtype=np.float64)
    error = wanted - current
    max_error = float(np.max(np.abs(error)))
    if max_error < threshold:
        return wanted
    return current + error * (threshold / max_error)


def _ratio(elapsed_ns: int, duration_ns: int) -> float:
    """Map elapsed time to a bounded interpolation ratio.

    Args:
        elapsed_ns (int): Time since the interpolation started.
        duration_ns (int): Duration of the whole interpolation.

    Returns:
        float: Ratio in ``[0.0, 1.0]``.
    """
    if elapsed_ns <= 0:
        return 0.0
    if duration_ns <= 0 or elapsed_ns >= duration_ns:
        return 1.0
    return elapsed_ns / duration_ns


def _slerp(start: np.ndarray, target: np.ndarray, ratio: float) -> np.ndarray:
    """Spherically interpolate two quaternions.

    Args:
        start (np.ndarray): Start quaternion (x, y, z, w).
        target (np.ndarray): End quaternion (x, y, z, w).
        ratio (float): Interpolation ratio in ``[0.0, 1.0]``.

    Returns:
        np.ndarray: Normalised quaternion (x, y, z, w).
    """
    first = start / max(float(np.linalg.norm(start)), EPSILON)
    last = target / max(float(np.linalg.norm(target)), EPSILON)
    dot = float(np.dot(first, last))
    if dot < 0.0:
        last = -last
        dot = -dot
    if dot > 0.9995:
        return _normalise(quaternion=first + (last - first) * ratio)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    sine = np.sin(angle)
    return (_np_sin(value=(1.0 - ratio) * angle) / sine * first
            + _np_sin(value=ratio * angle) / sine * last)


def _np_sin(value: float) -> float:
    """Sine of an angle in radians, kept separate for readability.

    Args:
        value (float): Angle in radians.

    Returns:
        float: Sine of the angle.
    """
    return float(np.sin(value))


def _normalise(quaternion: np.ndarray) -> np.ndarray:
    """Normalise a quaternion.

    Args:
        quaternion (np.ndarray): Quaternion (x, y, z, w).

    Returns:
        np.ndarray: Unit quaternion.
    """
    return quaternion / max(float(np.linalg.norm(quaternion)), EPSILON)
