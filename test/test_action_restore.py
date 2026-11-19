################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the action chunk restorer table."""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.process.action_restore import (
    ACTION_RESTORERS,
    restore,
)
from hex_ros_policy_client.struct.cmd_struct import ACTION_MODES


def steps(values: list[list[float]]) -> np.ndarray:
    """Build a steps matrix.

    Args:
        values (list): Rows of the matrix.

    Returns:
        np.ndarray: Matrix of shape ``(steps, dof)``.
    """
    return np.asarray(values, dtype=np.float64)


def test_joint_is_unchanged() -> None:
    """Absolute positions pass through untouched."""
    matrix = steps([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(restore("joint", matrix), matrix)


def test_delta_is_prefix_sum_anchored_on_the_first_row() -> None:
    """Increments become positions by cumsum plus the first row."""
    matrix = steps([[1.0], [1.0], [2.0]])
    restored = restore("delta", matrix)
    assert restored[:, 0].tolist() == [2.0, 3.0, 5.0]


def test_joint_diff_adds_the_first_row_to_every_row() -> None:
    """Offsets become positions by adding the first row."""
    matrix = steps([[1.0], [1.0], [2.0]])
    restored = restore("joint_diff", matrix)
    assert restored[:, 0].tolist() == [2.0, 2.0, 3.0]


def test_restorers_keep_the_shape() -> None:
    """Every mode returns the shape it was given."""
    matrix = np.zeros((4, 7))
    for mode in ACTION_MODES:
        assert restore(mode, matrix).shape == matrix.shape


def test_every_protocol_mode_has_a_restorer() -> None:
    """The mode table and the restorer table stay in step."""
    assert set(ACTION_RESTORERS) == set(ACTION_MODES)


def test_unknown_mode_is_rejected() -> None:
    """A mode outside the table raises instead of passing through."""
    with pytest.raises(ValueError):
        restore("absolute", steps([[1.0]]))
