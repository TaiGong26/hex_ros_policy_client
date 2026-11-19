################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the interpolation and limiting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.utils import (
    get_interpolator,
    limit_to_step,
    linear_interpolate,
    pose_interpolate,
    slerp_interpolate,
)


@pytest.mark.parametrize("elapsed_ns, expected", [
    (-1, [0.0, 0.0]),
    (0, [0.0, 0.0]),
    (500, [5.0, 10.0]),
    (1000, [10.0, 20.0]),
    (2000, [10.0, 20.0]),
])
def test_linear_boundaries(elapsed_ns: int, expected: list) -> None:
    """Elapsed time clamps to the two ends of the """
    result = linear_interpolate(np.zeros(2), np.array([10.0, 20.0]),
                                          elapsed_ns, 1000)
    assert np.allclose(result, np.array(expected))


def test_get_interpolator_returns_table_entry() -> None:
    """A registered name resolves to its function."""
    assert get_interpolator("linear") is linear_interpolate


def test_get_interpolator_rejects_unknown_name() -> None:
    """An unregistered name is a configuration error."""
    with pytest.raises(ValueError):
        get_interpolator("cubic")


def test_slerp_rotates_quarter_turn() -> None:
    """Half of a quarter turn is a 45 degree rotation."""
    start = np.array([0.0, 0.0, 0.0, 1.0])
    target = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    result = slerp_interpolate(start, target, 500, 1000)
    assert np.allclose(result, [0.0, 0.0, np.sin(np.pi / 8),
                                np.cos(np.pi / 8)], atol=1e-9)


def test_slerp_rejects_wrong_shape() -> None:
    """A non quaternion input is rejected."""
    with pytest.raises(ValueError):
        slerp_interpolate(np.zeros(3), np.zeros(3), 0, 1)


def test_pose_interpolates_position_and_orientation() -> None:
    """A pose interpolates its two halves with their own rules."""
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    target = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    result = pose_interpolate(start, target, 500, 1000)
    assert np.allclose(result[:3], [0.5, 0.5, 0.5])
    assert np.allclose(result[3:], [0.0, 0.0, 0.0, 1.0], atol=1e-9)


def test_pose_rejects_wrong_shape() -> None:
    """A pose without seven elements is rejected."""
    with pytest.raises(ValueError):
        pose_interpolate(np.zeros(4), np.zeros(4), 0, 1)


def test_limit_keeps_target_below_threshold() -> None:
    """An error inside the threshold is sent unchanged."""
    measured = np.zeros(3)
    target = np.array([0.05, -0.05, 0.0])
    result = limit_to_step(target, measured, 0.1)
    assert np.allclose(result, target)


def test_limit_scales_above_threshold() -> None:
    """An error above the threshold is scaled on every element equally."""
    measured = np.zeros(2)
    target = np.array([0.5, 0.1])
    result = limit_to_step(target, measured, 0.1)
    ratio = 0.1 / 0.5
    assert np.allclose(result, target * ratio)
    assert np.allclose(np.sign(result), np.sign(target))


def test_limit_rejects_bad_threshold() -> None:
    """A non positive threshold is a configuration error."""
    with pytest.raises(ValueError):
        limit_to_step(np.zeros(1), np.zeros(1), 0.0)
