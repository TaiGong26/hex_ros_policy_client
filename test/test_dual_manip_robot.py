################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Unit tests of ``DualManipRobot``.

One block per behaviour: the initial state, start and stop, the device table
guards, and the input and output format of ``read_obs`` and ``send_ctrl`` on
both sides of the class.
"""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.core.dual_manip_robot import (
    SECTION_ARMS,
    SECTION_CAMERAS,
    DualManipRobot,
)
from hex_ros_policy_client.struct.cmd_struct import (
    ArmCommand,
    ControlIntent,
    GripCommand,
)
from hex_ros_policy_client.struct.device_struct import (
    ArmObservation,
    CameraObservation,
)

from .fakes import DUAL_TABLE, SINGLE_TABLE, FakeInterface, write_table

pytest.importorskip("hex_ros_msgs.msg")
pytest.importorskip("sensor_msgs.msg")

from hex_ros_msgs.msg import (  # noqa: E402
    HexRosRoboManipStateStamped,
)
from sensor_msgs.msg import Image  # noqa: E402

NS_PER_S = 1_000_000_000
CTRL_SUFFIX = "manip_ctrl"
ARM_DOF = 7
STATE_TOPIC = "left/manip_state"
RIGHT_STATE_TOPIC = "right/manip_state"
COLOR_TOPIC = "fixed/image_color_raw"


def _robot(interface: FakeInterface) -> DualManipRobot:
    """Create a robot wired to an interface.

    Args:
        interface (FakeInterface): Interface that creates the endpoints.

    Returns:
        DualManipRobot: Robot bound to that interface.
    """
    return DualManipRobot(
        device_path=interface.device_path,
        logger=interface.getlogger(),
        create_subscription=interface.create_subscription,
        create_publisher=interface.create_publisher,
    )


def build_robot(tmp_path) -> tuple[DualManipRobot, FakeInterface]:
    """Create a robot over the two arm, two camera table.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        tuple: Robot and the interface it uses.
    """
    interface = FakeInterface(
        device_path=write_table(tmp_path, DUAL_TABLE))
    return _robot(interface), interface


def started_robot(tmp_path) -> tuple[DualManipRobot, FakeInterface]:
    """Create a robot that already built its devices.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        tuple: Started robot and the interface it uses.
    """
    robot, interface = build_robot(tmp_path)
    robot.start()
    return robot, interface


def published(interface: FakeInterface, device: str = "") -> list:
    """Return the messages sent to the control endpoints.

    Args:
        interface (FakeInterface): Interface the robot created endpoints on.
        device (str): Device whose endpoint is read, every device when empty.

    Returns:
        list: Published messages, in order, empty when none was sent.
    """
    messages = []
    for topic, publisher in interface.node.publishers.items():
        if not topic.endswith(CTRL_SUFFIX):
            continue
        if device and f"{device}/" not in topic:
            continue
        messages.extend(publisher.sent)
    return messages


def state_message(stamp_ns: int, joints: list) -> HexRosRoboManipStateStamped:
    """Build one manipulation state message.

    Args:
        stamp_ns (int): Stamp of the message, in nanoseconds.
        joints (list): Measured joint positions.

    Returns:
        HexRosRoboManipStateStamped: Filled state message.
    """
    message = HexRosRoboManipStateStamped()
    message.header.stamp.sec = stamp_ns // NS_PER_S
    message.header.stamp.nanosec = stamp_ns % NS_PER_S
    arm_jnt = message.manip_state.arm_state.jnt
    arm_jnt.position = list(joints)
    arm_jnt.velocity = [0.0] * len(joints)
    arm_jnt.effort = [0.0] * len(joints)
    grip_jnt = message.manip_state.grip_state.jnt
    grip_jnt.position = [0.5]
    grip_jnt.velocity = [0.0]
    grip_jnt.effort = [0.0]
    return message


def image_message(stamp_ns: int) -> Image:
    """Build one mono image message.

    Args:
        stamp_ns (int): Stamp of the message, in nanoseconds.

    Returns:
        Image: Two by two mono image.
    """
    message = Image()
    message.header.stamp.sec = stamp_ns // NS_PER_S
    message.header.stamp.nanosec = stamp_ns % NS_PER_S
    message.width = 2
    message.height = 2
    message.encoding = "mono8"
    message.step = 2
    message.data = bytes(4)
    return message


def deliver(interface: FakeInterface, topic: str, message) -> None:
    """Deliver one message to the subscription of a device.

    Args:
        interface (FakeInterface): Interface the robot created endpoints on.
        topic (str): Topic the device subscribed to.
        message: Message delivered to the handler.
    """
    interface.node.subscription(topic).callback(message)


def robot_from(tmp_path, text: str) -> tuple[DualManipRobot, FakeInterface]:
    """Build a robot over a table given as text.

    Args:
        tmp_path: Pytest temporary directory.
        text (str): YAML contents.

    Returns:
        tuple: Robot and the interface it uses.
    """
    interface = FakeInterface(device_path=write_table(tmp_path, text))
    return _robot(interface), interface


# ── 1. initial state ──────────────────────────────────────────────
def test_initial_state_holds_no_device() -> None:
    """A robot that was never started holds no device and no name."""
    interface = FakeInterface()
    robot = _robot(interface)
    assert robot.device_names == {}
    assert robot.read_obs() == {"camera": {}, "arm": {}}


def test_send_ctrl_before_start_is_dropped(tmp_path) -> None:
    """A command sent before start is refused with a log line."""
    robot, interface = build_robot(tmp_path)
    robot.send_ctrl(ControlIntent(arm={"left": ArmCommand(pos=[0.0])}))
    assert published(interface) == []
    messages = interface.logger.messages()
    assert "[ctrl.route/not_started]" in messages[0]


# ── 2. start ──────────────────────────────────────────────────────
def test_start_builds_every_device(tmp_path) -> None:
    """The names follow the table, section by section and in order."""
    robot, _ = started_robot(tmp_path)
    assert robot.device_names[SECTION_ARMS] == ("left", "right")
    assert robot.device_names[SECTION_CAMERAS] == ("fixed", "left_wrist")


def test_start_is_idempotent(tmp_path) -> None:
    """Starting twice keeps the names of the first start."""
    robot, _ = started_robot(tmp_path)
    names = robot.device_names
    robot.start()
    assert robot.device_names is names


def test_single_arm_table_needs_no_code_change(tmp_path) -> None:
    """A table with one arm and one camera builds the same way."""
    interface = FakeInterface(
        device_path=write_table(tmp_path, SINGLE_TABLE))
    robot = _robot(interface)
    robot.start()
    assert robot.device_names[SECTION_ARMS] == ("main",)
    assert robot.device_names[SECTION_CAMERAS] == ("main_camera",)


def test_start_opens_one_endpoint_per_topic(tmp_path) -> None:
    """Every subscription and publisher of the table is created once."""
    _, interface = started_robot(tmp_path)
    topics = sorted(item.topic for item in interface.node.subscriptions)
    assert topics == [COLOR_TOPIC, STATE_TOPIC,
                      "left_wrist/image_color_raw",
                      "left_wrist/image_depth_raw", RIGHT_STATE_TOPIC]
    assert sorted(interface.node.publishers) == ["left/manip_ctrl",
                                                "right/manip_ctrl"]


# ── 3. stop ───────────────────────────────────────────────────────
def test_stop_releases_the_devices(tmp_path) -> None:
    """A stopped robot holds no device again."""
    robot, _ = started_robot(tmp_path)
    robot.stop()
    assert robot.device_names == {}
    assert robot.read_obs() == {"camera": {}, "arm": {}}


def test_stop_is_idempotent(tmp_path) -> None:
    """Stopping twice is allowed and keeps the released state."""
    robot, _ = started_robot(tmp_path)
    robot.stop()
    robot.stop()
    assert robot.device_names == {}


def test_start_after_stop_builds_again(tmp_path) -> None:
    """A restart after a stop builds a fresh set of devices."""
    robot, _ = started_robot(tmp_path)
    names = robot.device_names
    robot.stop()
    robot.start()
    assert robot.device_names is not names
    assert robot.device_names[SECTION_ARMS] == ("left", "right")


def test_send_ctrl_after_stop_is_dropped(tmp_path) -> None:
    """A command sent after stop is refused with a log line."""
    robot, interface = started_robot(tmp_path)
    robot.stop()
    robot.send_ctrl(ControlIntent(arm={"left": ArmCommand(pos=[0.0])}))
    assert published(interface) == []
    messages = interface.logger.messages()
    assert "[ctrl.route/not_started]" in messages[0]


# ── 4. table guards ───────────────────────────────────────────────
def test_missing_table_is_reported(tmp_path) -> None:
    """A missing table stops the start up."""
    interface = FakeInterface(device_path="/nonexistent/devices.yaml")
    with pytest.raises(FileNotFoundError, match="device table not found"):
        _robot(interface).start()


def test_non_mapping_table_is_rejected(tmp_path) -> None:
    """A table that is not a mapping is a configuration error."""
    robot, _ = robot_from(tmp_path, "- a\n- b\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        robot.start()


def test_unknown_section_is_rejected(tmp_path) -> None:
    """A section the robot does not know is a configuration error."""
    robot, _ = robot_from(tmp_path, "grippers:\n  left: {}\n")
    with pytest.raises(ValueError, match="unknown sections"):
        robot.start()


def test_table_without_devices_is_rejected(tmp_path) -> None:
    """A table without devices is a configuration error."""
    robot, _ = robot_from(tmp_path, "cameras: {}\n")
    with pytest.raises(ValueError, match="declares no device"):
        robot.start()


def test_non_mapping_section_is_rejected(tmp_path) -> None:
    """A section that is not a mapping is a configuration error."""
    robot, _ = robot_from(tmp_path, "arms:\n  - left\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        robot.start()


def test_non_mapping_entry_is_rejected(tmp_path) -> None:
    """An entry that is not a mapping is a configuration error."""
    robot, _ = robot_from(tmp_path, "arms:\n  left: 3\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        robot.start()


def test_missing_required_key_is_rejected(tmp_path) -> None:
    """An arm without its configuration sections is a configuration error."""
    text = """
arms:
  left:
    ros_cfg:
      namespace: left
      subscribers:
        state_topic: manip_state
"""
    robot, _ = robot_from(tmp_path, text)
    with pytest.raises(ValueError, match="missing required keys"):
        robot.start()


def test_duplicate_name_is_rejected(tmp_path) -> None:
    """The same name in two sections is ambiguous and therefore refused."""
    text = """
cameras:
  shared:
    ros_cfg:
      namespace: shared
      subscribers:
        color_topic: image_color_raw
arms:
  shared:
    ros_cfg:
      namespace: shared
      subscribers:
        state_topic: manip_state
      publishers:
        ctrl_topic: manip_ctrl
    arm_cfg: {}
    grip_cfg: {}
"""
    robot, _ = robot_from(tmp_path, text)
    with pytest.raises(ValueError, match="appears in both"):
        robot.start()


def test_device_configuration_error_propagates(tmp_path) -> None:
    """A device that refuses its own entry stops the start up."""
    text = """
cameras:
  fixed:
    ros_cfg:
      namespace: fixed
      subscribers:
        color_topic: image_color_raw
    crop_mode: stretch
"""
    robot, _ = robot_from(tmp_path, text)
    with pytest.raises(ValueError, match="unsupported crop_mode"):
        robot.start()


def test_a_failed_start_leaves_the_robot_empty(tmp_path) -> None:
    """A refused table does not leave a half built robot behind."""
    robot, _ = robot_from(tmp_path, "cameras: {}\n")
    with pytest.raises(ValueError):
        robot.start()
    assert robot.device_names == {}
    assert robot.read_obs() == {"camera": {}, "arm": {}}


# ── 5. read_obs output ────────────────────────────────────────────
def test_read_obs_groups_every_device(tmp_path) -> None:
    """The output is grouped by device type and keyed by the table names."""
    robot, _ = started_robot(tmp_path)
    observation = robot.read_obs()
    assert set(observation) == {"camera", "arm"}
    assert tuple(observation["arm"]) == ("left", "right")
    assert tuple(observation["camera"]) == ("fixed", "left_wrist")
    assert observation["arm"]["left"] is None
    assert observation["camera"]["fixed"] is None


def test_read_obs_returns_the_arm_sample(tmp_path) -> None:
    """A delivered state comes back as an ArmObservation of that arm."""
    robot, interface = started_robot(tmp_path)
    joints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    deliver(interface, STATE_TOPIC, state_message(NS_PER_S, joints))
    observation = robot.read_obs()
    sample = observation["arm"]["left"]
    assert isinstance(sample, ArmObservation)
    assert sample.stamp_ns == NS_PER_S
    assert list(sample.arm_state.arm_pos) == joints
    assert observation["arm"]["right"] is None


def test_read_obs_returns_the_camera_sample(tmp_path) -> None:
    """A delivered image comes back as a CameraObservation of that camera."""
    robot, interface = started_robot(tmp_path)
    deliver(interface, COLOR_TOPIC, image_message(NS_PER_S))
    observation = robot.read_obs()
    sample = observation["camera"]["fixed"]
    assert isinstance(sample, CameraObservation)
    assert sample.rgb is not None
    assert sample.depth is None


def test_read_obs_keeps_the_device_queue(tmp_path) -> None:
    """A sample stays readable, because the assembly needs a pair of them."""
    robot, interface = started_robot(tmp_path)
    deliver(interface, STATE_TOPIC, state_message(NS_PER_S, [0.0] * ARM_DOF))
    first = robot.read_obs()["arm"]["left"]
    second = robot.read_obs()["arm"]["left"]
    assert first is not None
    assert second is not None
    assert first.stamp_ns == second.stamp_ns


# ── 6. send_ctrl input ────────────────────────────────────────────
def test_send_ctrl_publishes_arm_and_gripper_together(tmp_path) -> None:
    """One target becomes one message carrying both commands."""
    robot, interface = started_robot(tmp_path)
    robot.send_ctrl(ControlIntent(
        arm={"left": ArmCommand(pos=np.zeros(ARM_DOF))},
        grip={"left": GripCommand(pos=[0.0])},
        stamp_ns=NS_PER_S + 12345,
    ))
    messages = published(interface, "left")
    assert len(messages) == 1
    message = messages[0]
    assert list(message.manip_ctrl.arm_ctrl.jnt.pos) == [0.0] * ARM_DOF
    assert message.manip_ctrl.grip_ctrl.jnt.pos
    assert message.header.stamp.sec == 1
    assert message.header.stamp.nanosec == 12345


def test_send_ctrl_reaches_one_device_per_name(tmp_path) -> None:
    """A target with two arms becomes one message per arm."""
    robot, interface = started_robot(tmp_path)
    robot.send_ctrl(ControlIntent(
        arm={"left": ArmCommand(pos=np.zeros(ARM_DOF)),
             "right": ArmCommand(pos=np.ones(ARM_DOF))},
        grip={"left": GripCommand(pos=[0.0]),
              "right": GripCommand(pos=[1.0])},
    ))
    left = published(interface, "left")
    right = published(interface, "right")
    assert len(left) == 1
    assert len(right) == 1
    assert list(left[0].manip_ctrl.arm_ctrl.jnt.pos) == [0.0] * ARM_DOF
    assert list(right[0].manip_ctrl.arm_ctrl.jnt.pos) == [1.0] * ARM_DOF


def test_send_ctrl_without_a_target_time_keeps_the_stamp_zero(
        tmp_path) -> None:
    """A target without a stamp leaves the header at zero."""
    robot, interface = started_robot(tmp_path)
    robot.send_ctrl(ControlIntent(
        arm={"left": ArmCommand(pos=np.zeros(ARM_DOF))},
        stamp_ns=None,
    ))
    message = published(interface, "left")[0]
    assert message.header.stamp.sec == 0
    assert message.header.stamp.nanosec == 0


def test_send_ctrl_without_a_gripper_command_still_publishes(
        tmp_path) -> None:
    """An arm command alone is published with the gripper at mode none."""
    robot, interface = started_robot(tmp_path)
    robot.send_ctrl(ControlIntent(arm={"left": ArmCommand(pos=[0.0])}))
    message = published(interface, "left")[0]
    assert message.manip_ctrl.grip_ctrl.ctrl_mode == 0


def test_send_ctrl_accepts_an_empty_target(tmp_path) -> None:
    """A target that addresses nothing publishes nothing."""
    robot, interface = started_robot(tmp_path)
    robot.send_ctrl(ControlIntent())
    assert published(interface) == []
    assert interface.logger.messages() == []


def test_send_ctrl_reports_an_unknown_device(tmp_path) -> None:
    """A command for an undeclared device is dropped with a log line."""
    robot, interface = started_robot(tmp_path)
    robot.send_ctrl(ControlIntent(arm={"ghost": ArmCommand(pos=[0.0])}))
    assert published(interface) == []
    messages = interface.logger.messages()
    assert "[ctrl.route/unknown_device]" in messages[0]


def test_send_ctrl_reports_a_camera_name(tmp_path) -> None:
    """A command addressed to a camera is dropped, cameras take no command."""
    robot, interface = started_robot(tmp_path)
    robot.send_ctrl(ControlIntent(arm={"fixed": ArmCommand(pos=[0.0])}))
    assert published(interface) == []
    messages = interface.logger.messages()
    assert "[ctrl.route/unknown_device]" in messages[0]


def test_send_ctrl_drops_a_gripper_only_command(tmp_path) -> None:
    """A gripper command without an arm command cannot be published."""
    robot, interface = started_robot(tmp_path)
    robot.send_ctrl(ControlIntent(grip={"left": GripCommand(pos=[0.0])}))
    assert published(interface) == []
    messages = interface.logger.messages()
    assert "[ctrl.route/no_arm_command]" in messages[0]
