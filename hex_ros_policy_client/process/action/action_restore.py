################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

import numpy as np

from hex_util_ros import quat_mul, quat_norm


EE_XYZ = slice(0, 3)
EE_QUAT = slice(3, 7)


def as_received(received: np.ndarray,
                anchor: np.ndarray | None = None) -> np.ndarray:
    """Return the received values unchanged.

    Args:
        received (np.ndarray): Values of shape ``(steps, dof)``, already
            absolute, where ``dof`` is the number of joints or dimensions.
        anchor (np.ndarray | None): Ignored, the values need no reference;
            kept so every restorer takes the same arguments.

    Returns:
        np.ndarray: The same array, not a copy and not converted, so a caller
        that needs float64 converts it itself.
    """
    return received


def pos_delta(received: np.ndarray,
              anchor: np.ndarray | None = None) -> np.ndarray:
    """Turn per step increments into absolute joint positions.

    Args:
        received (np.ndarray): Increments of shape ``(steps, dof)``, row 0
            taken from ``anchor`` and row i from row i-1, where ``dof`` is the
            number of joints.
        anchor (np.ndarray | None): Absolute joint positions of the chunk
            start, shape ``(dof,)``; required.

    Returns:
        np.ndarray: Absolute joint positions of shape ``(steps, dof)``,
        float64, the running sum added to ``anchor``.

    Raises:
        ValueError: If ``anchor`` is not given.
    """
    if anchor is None:
        raise ValueError("anchor is required to restore per step increments")
    return np.asarray(anchor, dtype=np.float64) + np.cumsum(received, axis=0)


def pos_diff(received: np.ndarray,
             anchor: np.ndarray | None = None) -> np.ndarray:
    """Turn per row differences into absolute joint positions.

    Args:
        received (np.ndarray): Differences of shape ``(steps, dof)``, every row
            taken from ``anchor``, where ``dof`` is the number of joints.
        anchor (np.ndarray | None): Absolute joint positions of the chunk
            start, shape ``(dof,)``; required.

    Returns:
        np.ndarray: Absolute joint positions of shape ``(steps, dof)``,
        float64, ``anchor`` added to every row.

    Raises:
        ValueError: If ``anchor`` is not given.
    """
    if anchor is None:
        raise ValueError("anchor is required to restore per row differences")
    return np.asarray(anchor, dtype=np.float64) + received


def ee_delta(received: np.ndarray,
             anchor: np.ndarray | None = None) -> np.ndarray:
    """Turn per step pose increments into absolute poses.

    Args:
        received (np.ndarray): Increments of shape ``(steps, 7)`` as
            ``(dx, dy, dz, dqw, dqx, dqy, dqz)``, row 0 taken from ``anchor``
            and row i from row i-1.
        anchor (np.ndarray | None): Absolute pose of the chunk start, shape
            ``(7,)`` as ``(x, y, z, qw, qx, qy, qz)``; required.

    Returns:
        np.ndarray: Absolute poses of shape ``(steps, 7)``, float64; the
        Cartesian position is a running sum from ``anchor``, the orientation
        is each increment multiplied onto the previous pose, normalized with
        ``qw >= 0``.

    Raises:
        ValueError: If ``anchor`` is not given, if a row is not seven wide, if
            ``anchor`` is not seven long, or if a row carries a zero
            quaternion, which names no rotation.
    """
    if anchor is None:
        raise ValueError("anchor is required to restore per step increments")

    poses = np.asarray(received, dtype=np.float64)

    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(
            f"ee steps must have shape (steps, 7), got {poses.shape}")
    anchor_pose = np.asarray(anchor, dtype=np.float64).reshape(-1)

    if anchor_pose.shape != (7,):
        raise ValueError(
            f"ee anchor must have shape (7,), got {anchor_pose.shape}")

    result = np.empty_like(poses)

    # Cartesian position, a plain running sum
    result[:, EE_XYZ] = anchor_pose[EE_XYZ] + np.cumsum(
        poses[:, EE_XYZ], axis=0)

    increments = poses[:, EE_QUAT]
    zero_rows = np.flatnonzero(np.linalg.norm(increments, axis=1) == 0.0)
    if zero_rows.size:
        raise ValueError(
            f"ee increments must carry a rotation, rows {zero_rows.tolist()} "
            "are zero, use (1, 0, 0, 0) for no rotation")

    orientations = np.empty((poses.shape[0], 4), dtype=np.float64)
    current = anchor_pose[EE_QUAT]
    for index, increment in enumerate(increments):
        current = quat_mul(increment, current)
        orientations[index] = current

    result[:, EE_QUAT] = quat_norm(orientations)
    return result


ACTION_RESTORERS = {
    ("pos", "abs"): as_received,
    ("pos", "delta"): pos_delta,
    ("pos", "diff"): pos_diff,
    ("ee", "abs"): as_received,
    ("ee", "delta"): ee_delta,
}


def received_to_absolute(arm_ctrl_mode: str,
                         action_mode: str,
                         received: np.ndarray,
                         anchor: np.ndarray | None = None) -> np.ndarray:
    """Convert one matrix of received values into absolute values.

    Args:
        arm_ctrl_mode (str): ``pos`` for joint positions or ``ee`` for an end
            effector pose.
        action_mode (str): ``abs`` when the values are already absolute,
            ``delta`` for per step increments and ``diff`` for per row
            differences.
        received (np.ndarray): Values of shape ``(steps, dof)`` as the model
            sent them.
        anchor (np.ndarray | None): Absolute values of the chunk start, shape
            ``(dof,)``; required unless ``action_mode`` is ``abs``.

    Returns:
        np.ndarray: Absolute values of shape ``(steps, dof)``; float64 for
        every mode except ``abs``, which returns ``received`` unconverted.

    Raises:
        ValueError: If the pair is not in ``ACTION_RESTORERS``, if a mode is
            not a hashable value, or if the selected restorer is called
            without ``anchor``.
    """
    key = (arm_ctrl_mode, action_mode)
    try:
        restorer = ACTION_RESTORERS.get(key)
    except TypeError as exc:
        raise ValueError(
            f"modes must be hashable strings, got {key!r}") from exc
    if restorer is None:
        raise ValueError(
            f"no restorer for arm_ctrl_mode={arm_ctrl_mode!r} "
            f"and action_mode={action_mode!r}, "
            f"expected one of {sorted(ACTION_RESTORERS)}")
    return restorer(received, anchor)
