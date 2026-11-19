################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the device level structures."""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.struct.device_struct import (
    ArmObservation,
    ArmState,
    CameraObservation,
    GripState,
    ImageFrame,
)
from hex_ros_policy_client.struct.geom_struct import Pose


def _frame() -> ImageFrame:
    """Build a minimal colour frame.

    Returns:
        ImageFrame: Frame with a 1x1 pixel buffer.
    """
    return ImageFrame(ts_ns=1, encoding="rgb8", width=1, height=1,
                      data=np.zeros((1, 1, 3), dtype=np.uint8))


def test_arm_state_requires_every_field() -> None:
    """Arm state has no optional field."""
    with pytest.raises(TypeError):
        ArmState(arm_pos=np.zeros(7))
    state = ArmState(arm_pos=np.zeros(7), arm_vel=np.zeros(7),
                     arm_eff=np.zeros(7), ee_base=Pose())
    assert state.arm_pos.shape == (7,)


def test_arm_observation_all_none_is_valid() -> None:
    """An arm that never received data is still a valid observation."""
    observation = ArmObservation(stamp_ns=None, arm_state=None,
                                 grip_state=None)
    assert observation.stamp_ns is None
    assert observation.arm_state is None
    assert observation.grip_state is None


def test_arm_observation_shares_one_stamp() -> None:
    """Arm and gripper do not keep separate clocks."""
    observation = ArmObservation(
        stamp_ns=42,
        arm_state=ArmState(np.zeros(2), np.zeros(2), np.zeros(2), Pose()),
        grip_state=GripState(np.zeros(1), np.zeros(1), np.zeros(1)),
    )
    assert observation.stamp_ns == 42
    assert not hasattr(observation, "grip_stamp_ns")


def test_camera_observation_depth_is_optional() -> None:
    """A colour only camera leaves depth empty."""
    color_only = CameraObservation(rgb=_frame())
    assert color_only.depth is None
    with_depth = CameraObservation(rgb=_frame(), depth=_frame())
    assert with_depth.depth is not None


def test_image_frame_fields_are_mandatory() -> None:
    """An image frame without pixels is rejected."""
    with pytest.raises(TypeError):
        ImageFrame(ts_ns=1, encoding="rgb8", width=1, height=1)
