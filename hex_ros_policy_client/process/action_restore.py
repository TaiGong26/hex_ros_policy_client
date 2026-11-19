################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Restoring the encoding of an action chunk.

The model side may send absolute positions or increments; the action processor
only ever works with absolute positions, so every chunk is restored on entry.

Each restorer has the same signature, ``(steps, dof)`` in and ``(steps, dof)``
out, and the anchor is the first row of the array. Replacing a mode therefore
means adding one entry to :data:`ACTION_RESTORERS`, and nothing else in the
package changes.

Attributes:
    ACTION_RESTORERS: Restorer per ``action_mode`` value.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Restorer = Callable[[np.ndarray], np.ndarray]


def restore_joint(steps: np.ndarray) -> np.ndarray:
    """Keep absolute positions unchanged.

    Args:
        steps (np.ndarray): Steps of shape ``(steps, dof)``.

    Returns:
        np.ndarray: The same steps.
    """
    return steps


def restore_delta(steps: np.ndarray) -> np.ndarray:
    """Restore increments by prefix sum, anchored on the first row.

    Args:
        steps (np.ndarray): Increments of shape ``(steps, dof)``.

    Returns:
        np.ndarray: ``cumsum(steps) + steps[0]``, same shape.
    """
    return np.cumsum(steps, axis=0) + steps[0]


def restore_joint_diff(steps: np.ndarray) -> np.ndarray:
    """Restore increments by adding the first row to every row.

    Args:
        steps (np.ndarray): Offsets of shape ``(steps, dof)``.

    Returns:
        np.ndarray: ``steps + steps[0]``, same shape.
    """
    return steps + steps[0]


ACTION_RESTORERS = {
    "joint": restore_joint,
    "delta": restore_delta,
    "joint_diff": restore_joint_diff,
}


def restore(action_mode: str, steps: np.ndarray) -> np.ndarray:
    """Restore one group of steps with the restorer of its mode.

    Args:
        action_mode (str): One of :data:`ACTION_RESTORERS`.
        steps (np.ndarray): Steps of shape ``(steps, dof)``.

    Returns:
        np.ndarray: Absolute positions, same shape.

    Raises:
        ValueError: If the mode has no restorer.
    """
    restorer = ACTION_RESTORERS.get(action_mode)
    if restorer is None:
        raise ValueError(f"no restorer for action_mode={action_mode!r}")
    return restorer(steps)
