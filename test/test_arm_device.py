################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the arm device."""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.struct.cmd_struct import (
    ARM_CTRL_MODES,
    GRIP_CTRL_MODES,
    ArmCommand,
    GripCommand,
)
from hex_ros_policy_client.struct.geom_struct import Pose
from hex_ros_policy_client.device.arm_device import (
    ARM_MODE_TABLE,
    GRIP_MODE_TABLE,
    ArmDevice,
)
from hex_ros_policy_client.utils.common_util.logkit import (
    _reset_state as reset_log,
)

from .fakes import FakeNode

hex_ros_msgs = pytest.importorskip("hex_ros_msgs.msg")

ARM_PARAMS = {
    "type": "arm",
    "state_topic": "left/manip_state",
    "ctrl_topic": "left/manip_ctrl",
    "grip_limit": [0.0, 0.75],
    "lim_vel": 0.5,
    "lim_acc": 50.0,
    "grav": [0.0, 0.0, 0.0],
}


def build_device() -> tuple[ArmDevice, FakeNode]:
    """Create an arm device bound to a fake node.

    Returns:
        tuple: Device and the node it uses.
    """
    node = FakeNode()
    return ArmDevice(node, "left", ARM_PARAMS), node


def state_message() -> object:
    """Build one manip_state message with known contents.

    Returns:
        object: Filled ``HexRosRoboManipStateStamped``.
    """
    message = hex_ros_msgs.HexRosRoboManipStateStamped()
    message.header.stamp.sec = 3
    message.header.stamp.nanosec = 5
    arm = message.manip_state.arm_state
    arm.jnt.position = [0.1 * index for index in range(7)]
    arm.jnt.velocity = [0.01 * index for index in range(7)]
    arm.jnt.effort = [0.001 * index for index in range(7)]
    arm.pose.position.x = 1.0
    arm.pose.orientation.w = 0.5
    grip = message.manip_state.grip_state
    grip.jnt.position = [0.25]
    grip.jnt.velocity = [0.0]
    grip.jnt.effort = [0.0]
    return message


def test_observation_before_any_message() -> None:
    """A fresh device reports nothing instead of a fake zero sample."""
    device, _ = build_device()
    assert device.get_obs() is None


def test_on_states_maps_every_field() -> None:
    """The message is converted into the data layer structure."""
    device, node = build_device()
    node.subscription("left/manip_state").callback(state_message())
    observation = device.get_obs()
    assert observation.stamp_ns == 3 * 1_000_000_000 + 5
    assert observation.arm_state.arm_pos.shape == (7,)
    assert observation.arm_state.ee_base.position.x == 1.0
    assert observation.grip_state.grip_pos[0] == 0.25


def test_send_arm_command() -> None:
    """A joint command fills the arm half of the control message."""
    device, node = build_device()
    device.send_ctrl(ArmCommand(
                                pos=np.zeros(7)))
    sent = node.publishers["left/manip_ctrl"].sent
    assert len(sent) == 1
    arm_ctrl = sent[0].manip_ctrl.arm_ctrl
    assert arm_ctrl.ctrl_mode == hex_ros_msgs.HexRosRoboArmCtrl.CTRL_MODE_JNT
    assert list(arm_ctrl.jnt.pos) == [0.0] * 7
    assert list(arm_ctrl.jnt.lim_vel) == [0.5]
    assert list(arm_ctrl.jnt.lim_acc) == [50.0]
    assert arm_ctrl.grav.x == 0.0


def test_control_message_carries_no_stamp() -> None:
    """Dispatch means execute now, so the header stays empty."""
    device, node = build_device()
    device.send_ctrl(ArmCommand(pos=np.zeros(7)))
    stamp = node.publishers["left/manip_ctrl"].sent[0].header.stamp
    assert (stamp.sec, stamp.nanosec) == (0, 0)


@pytest.mark.parametrize("value, expected", [
    (-1.0, 0.0),
    (0.0, 0.375),
    (1.0, 0.75),
    (2.0, 0.75),
    (-3.0, 0.0),
])
def test_grip_mapping(value: float, expected: float) -> None:
    """The normalised gripper target is mapped onto the travel range."""
    device, node = build_device()
    device.send_ctrl(GripCommand(pos=[value]))
    sent = node.publishers["left/manip_ctrl"].sent[0]
    assert sent.manip_ctrl.grip_ctrl.jnt.pos[0] == pytest.approx(expected)


def test_unsupported_mode_is_dropped_not_raised() -> None:
    """End effector mode is not registered yet, so it is dropped with a log."""
    reset_log()
    device, node = build_device()
    device.send_ctrl(ArmCommand(ctrl_mode=ARM_CTRL_MODES[1], pose=Pose()))
    assert node.publishers["left/manip_ctrl"].sent == []
    lines = node.logger.messages()
    assert len(lines) == 1
    assert lines[0].startswith("[arm.left/unsupported_mode] ")
    assert "ctrl_mode=ee" in lines[0]


def test_mode_tables() -> None:
    """Only joint mode is registered for arms, both modes for grippers."""
    assert set(ARM_MODE_TABLE) == {ARM_CTRL_MODES[0]}
    assert set(GRIP_MODE_TABLE) == set(GRIP_CTRL_MODES)


def test_unknown_command_type_is_dropped() -> None:
    """A foreign object is reported instead of crashing the tick."""
    reset_log()
    device, node = build_device()
    device.send_ctrl(object())
    assert node.publishers["left/manip_ctrl"].sent == []
    assert "unknown command type" in node.logger.messages()[0]


def test_release_destroys_both_endpoints() -> None:
    """Releasing the device drops its subscription and publisher."""
    device, node = build_device()
    subscription = node.subscription("left/manip_state")
    device._release()
    assert subscription.destroyed is True
    assert len(node.destroyed) == 2
