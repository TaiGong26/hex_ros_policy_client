################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the device hub."""

from __future__ import annotations

import pytest

from hex_ros_policy_client.device.device_hub import DeviceHub

from .fakes import FakeNode

pytest.importorskip("hex_ros_msgs.msg")
pytest.importorskip("sensor_msgs.msg")

DUAL_TABLE = """
cameras:
  fixed:
    type: camera
    color_topic: fixed/image_color_raw
  left_wrist:
    type: camera
    color_topic: left_wrist/image_color_raw
    depth_topic: left_wrist/image_depth_raw
arms:
  left:
    type: arm
    state_topic: left/manip_state
    ctrl_topic: left/manip_ctrl
    grip_limit: [0.0, 0.75]
  right:
    type: arm
    state_topic: right/manip_state
    ctrl_topic: right/manip_ctrl
    grip_limit: [0.0, 0.75]
"""

SINGLE_TABLE = """
cameras:
  main_camera:
    type: camera
    color_topic: main/image_color_raw
arms:
  main:
    type: arm
    state_topic: main/manip_state
    ctrl_topic: main/manip_ctrl
    grip_limit: [0.0, 1.0]
"""


def write_table(tmp_path, text: str) -> str:
    """Write a device table and return its path.

    Args:
        tmp_path: Pytest temporary directory.
        text (str): YAML contents.

    Returns:
        str: Path of the written table.
    """
    path = tmp_path / "devices.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_dual_table_builds_every_device(tmp_path) -> None:
    """The keys of the hub match the keys of the table."""
    hub = DeviceHub.from_table(FakeNode(), write_table(tmp_path, DUAL_TABLE))
    assert hub.camera_names == ("fixed", "left_wrist")
    assert hub.arm_names == ("left", "right")


def test_single_arm_table(tmp_path) -> None:
    """A single arm table needs no code change."""
    hub = DeviceHub.from_table(FakeNode(), write_table(tmp_path,
                                                       SINGLE_TABLE))
    assert hub.arm_names == ("main",)
    assert hub.camera_names == ("main_camera",)


def test_missing_file_is_reported() -> None:
    """A missing table stops the start up."""
    with pytest.raises(FileNotFoundError):
        DeviceHub.from_table(FakeNode(), "/nonexistent/devices.yaml")


def test_empty_table_is_rejected(tmp_path) -> None:
    """A table without devices is a configuration error."""
    with pytest.raises(ValueError):
        table = write_table(tmp_path, "cameras: {}\n")
        DeviceHub.from_table(FakeNode(), table)


def test_unknown_type_is_rejected(tmp_path) -> None:
    """An unknown device type is a configuration error."""
    text = """
arms:
  left:
    type: gripper
    state_topic: left/manip_state
    ctrl_topic: left/manip_ctrl
    grip_limit: [0.0, 1.0]
"""
    with pytest.raises(ValueError):
        DeviceHub.from_table(FakeNode(), write_table(tmp_path, text))


def test_missing_required_key_is_rejected(tmp_path) -> None:
    """A camera without its colour topic is a configuration error."""
    text = """
cameras:
  fixed:
    type: camera
"""
    with pytest.raises(ValueError):
        DeviceHub.from_table(FakeNode(), write_table(tmp_path, text))


def test_duplicate_name_is_rejected(tmp_path) -> None:
    """The same name in two sections is ambiguous and therefore refused."""
    text = """
cameras:
  shared:
    type: camera
    color_topic: shared/image_color_raw
arms:
  shared:
    type: arm
    state_topic: shared/manip_state
    ctrl_topic: shared/manip_ctrl
    grip_limit: [0.0, 1.0]
"""
    with pytest.raises(ValueError):
        DeviceHub.from_table(FakeNode(), write_table(tmp_path, text))


def test_hub_has_no_behaviour() -> None:
    """The hub only knows the factory and its attributes."""
    public = {name for name in dir(DeviceHub) if not name.startswith("_")}
    assert public == {"from_table", "camera_devices", "arm_devices",
                      "camera_names", "arm_names"}
    for forbidden in ("read_obs", "send_ctrl", "start", "stop"):
        assert not hasattr(DeviceHub, forbidden)
