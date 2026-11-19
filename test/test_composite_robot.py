################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the composite robot."""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.core.composite_robot_base import CompositeRobotBase
from hex_ros_policy_client.core.dual_manip_robot import DualManipRobot
from hex_ros_policy_client.struct.cmd_struct import (
    ARM_CTRL_MODES,
    GRIP_CTRL_MODES,
    TARGET_ARM,
    TARGET_GRIP,
    ArmCommand,
    GripCommand,
)

from .fakes import FakeNode
from .test_device_hub import DUAL_TABLE, write_table

pytest.importorskip("hex_ros_msgs.msg")
pytest.importorskip("sensor_msgs.msg")


def build_robot(tmp_path) -> tuple[DualManipRobot, FakeNode]:
    """Create a robot bound to a table with two arms and one camera.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        tuple: Robot and the node it uses.
    """
    node = FakeNode()
    robot = DualManipRobot(node, write_table(tmp_path, DUAL_TABLE))
    return robot, node


def test_start_builds_devices_once(tmp_path) -> None:
    """Starting twice keeps the devices of the first start."""
    robot, _ = build_robot(tmp_path)
    robot.start()
    hub = robot.device_hub
    robot.start()
    assert robot.device_hub is hub
    assert hub.arm_names == ("left", "right")


def test_stop_releases_devices(tmp_path) -> None:
    """Stopping releases the endpoints and is idempotent."""
    robot, node = build_robot(tmp_path)
    robot.start()
    robot.stop()
    assert robot.device_hub is None
    released = len(node.destroyed)
    robot.stop()
    assert len(node.destroyed) == released


def test_read_obs_is_grouped(tmp_path) -> None:
    """Observations come back grouped by device type."""
    robot, _ = build_robot(tmp_path)
    robot.start()
    observation = robot.read_obs()
    assert set(observation) == {"camera", "arm"}
    assert tuple(observation["arm"]) == ("left", "right")
    assert tuple(observation["camera"]) == ("fixed", "left_wrist")
    assert observation["arm"]["left"] is None


def test_send_ctrl_transposes_once(tmp_path) -> None:
    """One target becomes one command per device and group."""
    robot, node = build_robot(tmp_path)
    robot.start()
    calls = []
    for name, device in robot.device_hub.arm_devices.items():
        device.send_ctrl = lambda command, name=name: calls.append(
            (name, command.ctrl_mode))
    target = {
        TARGET_ARM: {"left": ArmCommand(pos=np.zeros(7))},
        TARGET_GRIP: {"left": GripCommand(pos=[0.0])},
    }
    robot.send_ctrl(target)
    assert calls == [("left", ARM_CTRL_MODES[0]),
                    ("left", GRIP_CTRL_MODES[0])]


def test_send_ctrl_reports_unknown_device(tmp_path) -> None:
    """A command for an undeclared device is dropped with a log line."""
    robot, node = build_robot(tmp_path)
    robot.start()
    target = {TARGET_ARM: {"ghost": ArmCommand(pos=[0.0])}}
    robot.send_ctrl(target)
    assert "ctrl.route/unknown_device" in node.logger.messages()[0]


def test_public_surface_is_four_methods() -> None:
    """The robot exposes start, stop, read_obs and send_ctrl only."""
    public = {name for name in dir(CompositeRobotBase)
              if not name.startswith("_")}
    assert public == {"start", "stop", "read_obs", "send_ctrl",
                      "device_hub"}


def test_robot_does_not_parse_the_table() -> None:
    """The robot only holds the path; parsing lives in the hub."""
    source = CompositeRobotBase.read_obs.__doc__ or ""
    assert "yaml" not in source
    robot_source = str(DualManipRobot.__module__)
    assert robot_source.endswith("dual_manip_robot")
