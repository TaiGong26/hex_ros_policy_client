################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from typing import Callable

import numpy as np

from hex_util_ros import interp_joint, quat_slerp


def linear_interpolate(start: np.ndarray,
                       target: np.ndarray,
                       elapsed_ns: int,
                       duration_ns: int) -> np.ndarray:
    """Blend two vectors in proportion to the elapsed time.

    Args:
        start (np.ndarray): Vector at the start of the segment, same shape as
            ``target``.
        target (np.ndarray): Vector at the end of the segment.
        elapsed_ns (int): Nanoseconds elapsed since the start of the segment;
            values outside ``[0, duration_ns]`` are clamped.
        duration_ns (int): Length of the segment in nanoseconds.

    Returns:
        np.ndarray: Vector of the shape of ``start``; ``start`` when
        ``elapsed_ns <= 0``, ``target`` when ``elapsed_ns >= duration_ns`` or
        when ``duration_ns`` is not positive.
    """
    ratio = 1.0 if duration_ns <= 0 else float(np.clip(
        a=elapsed_ns / duration_ns,
        a_min=0.0,
        a_max=1.0,
    ))
    return start + (target - start) * ratio


def slerp_interpolate(start: np.ndarray,
                      target: np.ndarray,
                      elapsed_ns: int,
                      duration_ns: int) -> np.ndarray:
    """Blend two quaternions along the shortest arc.

    Args:
        start (np.ndarray): Unit quaternion ``(w, x, y, z)`` at the start of
            the segment, shape ``(4,)``.
        target (np.ndarray): Unit quaternion ``(w, x, y, z)`` at the end of the
            segment, shape ``(4,)``.
        elapsed_ns (int): Nanoseconds elapsed since the start of the segment;
            values outside ``[0, duration_ns]`` are clamped.
        duration_ns (int): Length of the segment in nanoseconds.

    Returns:
        np.ndarray: Unit quaternion ``(w, x, y, z)`` of shape ``(4,)``.

    Raises:
        ValueError: If either input does not have shape ``(4,)``.
    """
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(target, dtype=np.float64)
    if first.shape != (4,) or last.shape != (4,):
        raise ValueError("slerp expects quaternions of shape (4,)")
    ratio = 1.0 if duration_ns <= 0 else float(np.clip(
        a=elapsed_ns / duration_ns,
        a_min=0.0,
        a_max=1.0,
    ))
    return quat_slerp(
        q1=first,
        q2=last,
        t=ratio,
    )


def pose_interpolate(start: np.ndarray,
                     target: np.ndarray,
                     elapsed_ns: int,
                     duration_ns: int) -> np.ndarray:
    """Blend two poses, position in proportion and orientation along the arc.

    Args:
        start (np.ndarray): Pose ``(px, py, pz, qw, qx, qy, qz)`` at the start
            of the segment, shape ``(7,)``.
        target (np.ndarray): Pose with the same layout at the end of the
            segment, shape ``(7,)``.
        elapsed_ns (int): Nanoseconds elapsed since the start of the segment;
            values outside ``[0, duration_ns]`` are clamped.
        duration_ns (int): Length of the segment in nanoseconds.

    Returns:
        np.ndarray: Pose ``(px, py, pz, qw, qx, qy, qz)`` of shape ``(7,)``;
        ``start`` when ``elapsed_ns <= 0``, ``target`` when
        ``elapsed_ns >= duration_ns`` or when ``duration_ns`` is not positive.

    Raises:
        ValueError: If either input does not have shape ``(7,)``.
    """
    if np.shape(start) != (7,) or np.shape(target) != (7,):
        raise ValueError("pose interpolation expects poses of shape (7,)")
    return np.concatenate([
        linear_interpolate(
            start=np.asarray(start, dtype=np.float64)[:3],
            target=np.asarray(target, dtype=np.float64)[:3],
            elapsed_ns=elapsed_ns,
            duration_ns=duration_ns,
        ),
        slerp_interpolate(
            start=np.asarray(start, dtype=np.float64)[3:],
            target=np.asarray(target, dtype=np.float64)[3:],
            elapsed_ns=elapsed_ns,
            duration_ns=duration_ns,
        ),
    ])


def pos_lim(target: np.ndarray,
            curr: np.ndarray,
            threshold: float) -> np.ndarray:
    """Scale the increment of a position down to a threshold.

    Args:
        target (np.ndarray): Position wanted this tick, same shape as ``curr``.
        curr (np.ndarray): Position measured this tick.
        threshold (float): Largest allowed increment, positive, in the unit of
            the position.

    Returns:
        np.ndarray: Position to send, shape of ``curr``; equal to ``target``
        when every element moves less than ``threshold``, otherwise the
        largest element moves exactly ``threshold``.
    """
    return np.asarray(
        interp_joint(
            cur_q=curr,
            tar_joint=target,
            err_limit=threshold,
        ),
        dtype=np.float64,
    )


INTERPOLATORS: dict[str, Callable[..., np.ndarray]] = {
    "linear": linear_interpolate,
    "slerp": slerp_interpolate,
    "pose": pose_interpolate,
    "pos_lim": pos_lim,
}


def get_interpolator_fn(name: str) -> Callable[..., np.ndarray]:
    """Return the function registered under one name.

    Args:
        name (str): Key of ``INTERPOLATORS``.

    Returns:
        Callable[..., np.ndarray]: Function registered under ``name``.

    Raises:
        ValueError: If ``name`` is not a key of ``INTERPOLATORS``.
    """
    try:
        return INTERPOLATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown interpolation {name!r}, "
            f"expected one of {sorted(INTERPOLATORS)}") from None
