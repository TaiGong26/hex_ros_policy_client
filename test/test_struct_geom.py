################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the geometric primitives."""

from __future__ import annotations

import dataclasses

import pytest

from hex_ros_policy_client.struct.geom_struct import Pose, Quaternion, Vector3


def test_defaults() -> None:
    """Defaults describe the origin and the identity rotation."""
    vector = Vector3()
    quaternion = Quaternion()
    pose = Pose()
    assert (vector.x, vector.y, vector.z) == (0.0, 0.0, 0.0)
    assert (quaternion.x, quaternion.y, quaternion.z) == (0.0, 0.0, 0.0)
    assert quaternion.w == 1.0
    assert pose.position == vector
    assert pose.orientation == quaternion


def test_frozen_assignment_rejected() -> None:
    """A frozen instance cannot be modified after construction."""
    vector = Vector3()
    with pytest.raises(dataclasses.FrozenInstanceError):
        vector.x = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        Quaternion().w = 0.5


def test_default_factory_makes_independent_poses() -> None:
    """Two default poses never share the same nested objects."""
    first = Pose()
    second = Pose()
    assert first.position is not second.position
    assert first.orientation is not second.orientation
