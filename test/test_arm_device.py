################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-24
################################################################
"""Unit tests for the rewritten HexArmDevice."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

# Fallback stubs for ROS message types if run outside full ROS 2 environment
try:
    from geometry_msgs.msg import Vector3 as Vector3Msg
    from hex_ros_msgs.msg import (
        HexRosRoboArmCtrl,
        HexRosRoboGripCtrl,
        HexRosRoboManipCtrlStamped,
        HexRosRoboManipStateStamped,
    )
except ImportError:
    class _MockVector3:
        def __init__(self, x: float = 0.0, y: float = 0.0,
                     z: float = 0.0) -> None:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class _MockQuaternion:
        def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0,
                     z: float = 0.0) -> None:
            self.w = float(w)
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class _MockPose:
        def __init__(self) -> None:
            self.position = _MockVector3()
            self.orientation = _MockQuaternion()

    class _MockJnt:
        def __init__(self) -> None:
            self.pos: list[float] = []
            self.eff: list[float] = []
            self.lim_vel: list[float] = []
            self.lim_acc: list[float] = []
            self.position: list[float] = []
            self.velocity: list[float] = []
            self.effort: list[float] = []

    class HexRosRoboArmCtrl:
        CTRL_MODE_NONE = 0
        CTRL_MODE_MIT = 1
        CTRL_MODE_JNT = 2
        CTRL_MODE_EE = 3

        def __init__(self) -> None:
            self.ctrl_mode = self.CTRL_MODE_NONE
            self.jnt = _MockJnt()
            self.pose = _MockPose()
            self.grav = _MockVector3()

    class HexRosRoboGripCtrl:
        CTRL_MODE_NONE = 0
        CTRL_MODE_MIT = 1
        CTRL_MODE_JNT = 2
        CTRL_MODE_TAU = 3

        def __init__(self) -> None:
            self.ctrl_mode = self.CTRL_MODE_NONE
            self.jnt = _MockJnt()

    class _MockManipCtrl:
        def __init__(self) -> None:
            self.arm_ctrl = HexRosRoboArmCtrl()
            self.grip_ctrl = HexRosRoboGripCtrl()

    class _MockHeader:
        def __init__(self) -> None:
            self.stamp = types.SimpleNamespace(sec=0, nanosec=0)

    class HexRosRoboManipCtrlStamped:
        def __init__(self) -> None:
            self.header = _MockHeader()
            self.manip_ctrl = _MockManipCtrl()

    class _MockArmState:
        def __init__(self) -> None:
            self.jnt = _MockJnt()
            self.pose = _MockPose()

    class _MockGripState:
        def __init__(self) -> None:
            self.jnt = _MockJnt()

    class _MockManipState:
        def __init__(self) -> None:
            self.arm_state = _MockArmState()
            self.grip_state = _MockGripState()

    class HexRosRoboManipStateStamped:
        def __init__(self) -> None:
            self.header = _MockHeader()
            self.manip_state = _MockManipState()

    Vector3Msg = _MockVector3

    _geom_mod = types.ModuleType("geometry_msgs")
    _geom_msg_mod = types.ModuleType("geometry_msgs.msg")
    _geom_msg_mod.Vector3 = Vector3Msg
    _geom_mod.msg = _geom_msg_mod
    sys.modules.setdefault("geometry_msgs", _geom_mod)
    sys.modules.setdefault("geometry_msgs.msg", _geom_msg_mod)

    _hex_mod = types.ModuleType("hex_ros_msgs")
    _hex_msg_mod = types.ModuleType("hex_ros_msgs.msg")
    _hex_msg_mod.HexRosRoboArmCtrl = HexRosRoboArmCtrl
    _hex_msg_mod.HexRosRoboGripCtrl = HexRosRoboGripCtrl
    _hex_msg_mod.HexRosRoboManipCtrlStamped = HexRosRoboManipCtrlStamped
    _hex_msg_mod.HexRosRoboManipStateStamped = HexRosRoboManipStateStamped
    _hex_mod.msg = _hex_msg_mod
    sys.modules.setdefault("hex_ros_msgs", _hex_mod)
    sys.modules.setdefault("hex_ros_msgs.msg", _hex_msg_mod)

from hex_ros_policy_client.device.arm_device import (
    HexArmDevice,
)
from hex_ros_policy_client.struct.cmd_struct import (
    ArmCommand,
    GripCommand,
)
from hex_ros_policy_client.struct.geom_struct import (
    Pose,
    Quaternion,
    Vector3,
)


class MockPublisher:
    """Mock publisher storing sent messages."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.published: list[Any] = []

    def publish(self, msg: Any) -> None:
        self.published.append(msg)


class EndpointRecorder:
    """Record endpoint creation calls and store callable handles."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.handles: dict[str, Any] = {}
        self.publishers: dict[str, MockPublisher] = {}

    def create_subscription(
        self,
        msg_type: Any,
        topic: str,
        callback: Any,
    ) -> str:
        self.calls.append(("sub", msg_type, topic, callback.__name__))
        handle = f"sub:{topic}"
        self.handles[handle] = callback
        return handle

    def create_publisher(self, msg_type: Any, topic: str) -> MockPublisher:
        self.calls.append(("pub", msg_type, topic))
        pub = MockPublisher(topic=topic)
        self.publishers[topic] = pub
        return pub


class CaptureLogger:
    """Capture logger warning calls."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def logw(self, message: str) -> None:
        self.warning(message)


def build_device(
    name: str = "left",
    params: dict | None = None,
    now_fn: Callable[[], int] | None = None,
) -> tuple[HexArmDevice, EndpointRecorder, CaptureLogger]:
    """Helper to instantiate a HexArmDevice with recorded endpoints."""
    default_params = {
        "ros_cfg": {
            "namespace": name,
            "subscribers": {"state_topic": "manip_state"},
            "publishers": {"ctrl_topic": "manip_ctrl"},
        },
        "arm_cfg": {
            "lim_vel": 0.5,
            "lim_acc": 50.0,
            "grav": [0.0, 0.0, -9.81],
            "stable_position": [0.0, -1.5, 3.0, 0.0, 0.0, 0.0],
            "kp": [50.0, 50.0, 50.0, 50.0, 20.0, 20.0],
            "kd": [2.5, 2.5, 2.5, 2.5, 1.0, 1.0],
        },
        "grip_cfg": {
            "lim_pos": [0.0, 0.70],
            "lim_vel": 1.0,
            "lim_eff": 1.0,
        },
    }
    cfg = params if params is not None else default_params
    recorder = EndpointRecorder()
    logger = CaptureLogger()
    device = HexArmDevice(
        name=name,
        params=cfg,
        create_subscription=recorder.create_subscription,
        create_publisher=recorder.create_publisher,
        logger=logger,
        now_fn=now_fn,
    )
    return device, recorder, logger


def make_state_msg(
    stamp: tuple[int, int] = (1, 0),
    arm_pos: list[float] | None = None,
    grip_pos: list[float] | None = None,
) -> HexRosRoboManipStateStamped:
    """Build a mock state message with deterministic joint values."""
    msg = HexRosRoboManipStateStamped()
    msg.header.stamp.sec = stamp[0]
    msg.header.stamp.nanosec = stamp[1]

    arm = msg.manip_state.arm_state
    arm.jnt.position = arm_pos if arm_pos is not None else [
        0.1 * i for i in range(7)]
    arm.jnt.velocity = [0.01 * i for i in range(7)]
    arm.jnt.effort = [0.001 * i for i in range(7)]
    arm.pose.position.x = 1.0
    arm.pose.position.y = 2.0
    arm.pose.position.z = 3.0
    arm.pose.orientation.w = 1.0
    arm.pose.orientation.x = 0.0
    arm.pose.orientation.y = 0.0
    arm.pose.orientation.z = 0.0

    grip = msg.manip_state.grip_state
    grip.jnt.position = grip_pos if grip_pos is not None else [0.35]
    grip.jnt.velocity = [0.0]
    grip.jnt.effort = [0.0]
    return msg


# =====================================================================
# 1. 构造与配置解析
# =====================================================================

def test_init_creates_state_sub_and_ctrl_pub() -> None:
    """HexArmDevice creates state subscriber and control publisher."""
    device, recorder, _ = build_device("left")
    assert device.name == "left"
    assert "state_topic" in device.subscribers
    assert "ctrl_topic" in device.publishers
    assert len(recorder.calls) == 2
    assert recorder.calls[0] == (
        "sub", HexRosRoboManipStateStamped, "left/manip_state", "on_states")
    assert recorder.calls[1] == (
        "pub", HexRosRoboManipCtrlStamped, "left/manip_ctrl")


def test_init_default_params_fallback() -> None:
    """Missing optional fields fallback to defined defaults."""
    params = {
        "ros_cfg": {
            "namespace": "arm",
            "subscribers": {"state_topic": "st"},
            "publishers": {"ctrl_topic": "ct"},
        },
        "arm_cfg": {},
        "grip_cfg": {},
    }
    device, _, _ = build_device(params=params)
    stored = device._HexArmDevice__params
    assert stored["arm"]["lim_vel"] == 0.5
    assert stored["arm"]["lim_acc"] == 50.0
    assert stored["arm"]["grav"] == (0.0, 0.0, -9.81)
    assert stored["arm"]["stable_position"] is None
    assert stored["grip"]["lim_pos"] == (0.0, 1.0)
    assert stored["grip"]["lim_vel"] == 1.0
    assert stored["grip"]["lim_eff"] == 1.0


def test_init_rejects_missing_required_keys() -> None:
    """Device missing any of REQUIRED_KEYS fails fast."""
    with pytest.raises(ValueError) as caught:
        build_device(params={"grip_cfg": {}})
    assert "missing required keys" in str(caught.value)


# =====================================================================
# 2. 队列机制与二分单调性
# =====================================================================

def test_enqueue_maintains_monotonic_stamp_ordering() -> None:
    """OutOfOrder arriving state messages are placed in sorted order."""
    device, recorder, _ = build_device()
    cb = recorder.handles["sub:left/manip_state"]
    cb(make_state_msg(stamp=(10, 0)))
    cb(make_state_msg(stamp=(30, 0)))
    cb(make_state_msg(stamp=(20, 0)))

    queue = device._HexArmDevice__queue
    stamps = [item[0] for item in queue]
    assert stamps == [10_000_000_000, 20_000_000_000, 30_000_000_000]


def test_enqueue_respects_queue_maxlen() -> None:
    """Queue length does not exceed maxlen, evicting oldest states."""
    device, recorder, _ = build_device()
    cb = recorder.handles["sub:left/manip_state"]
    for sec in range(35):
        cb(make_state_msg(stamp=(sec, 0)))

    queue = device._HexArmDevice__queue
    assert len(queue) == 30
    assert queue[0][0] == 5 * 1_000_000_000
    assert queue[-1][0] == 34 * 1_000_000_000


# =====================================================================
# 3. get_obs(drain) 语义与空队列保护
# =====================================================================

def test_get_obs_empty_queue_returns_none() -> None:
    """Empty queue safely returns None without raising IndexError."""
    device, _, _ = build_device()
    assert device.get_obs() is None


def test_get_obs_drain_true_clears_queue() -> None:
    """get_obs(drain=True) clears the queue upon retrieval."""
    device, recorder, _ = build_device()
    cb = recorder.handles["sub:left/manip_state"]
    cb(make_state_msg(stamp=(10, 0)))
    obs = device.get_obs(drain=True)
    assert obs is not None
    assert obs.stamp_ns == 10_000_000_000
    assert len(device._HexArmDevice__queue) == 0
    assert device.get_obs() is None


def test_get_obs_drain_false_retains_queue() -> None:
    """get_obs(drain=False) retains the queue for multiple reads."""
    device, recorder, _ = build_device()
    cb = recorder.handles["sub:left/manip_state"]
    cb(make_state_msg(stamp=(10, 0)))
    obs1 = device.get_obs(drain=False)
    assert obs1 is not None
    assert len(device._HexArmDevice__queue) == 1
    obs2 = device.get_obs(drain=False)
    assert obs2 is not None


# =====================================================================
# 4. 控制下发 send_ctrl 多模式与联合打包
# =====================================================================

def test_send_ctrl_joint_pos_mode() -> None:
    """Arm pos command maps to CTRL_MODE_JNT with limits and gravity."""
    device, recorder, _ = build_device()
    arm_cmd = ArmCommand(ctrl_mode="pos", pos=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    device.send_ctrl(arm_command=arm_cmd)

    pub = recorder.publishers["left/manip_ctrl"]
    assert len(pub.published) == 1
    sent = pub.published[0]
    arm_ctrl = sent.manip_ctrl.arm_ctrl
    assert arm_ctrl.ctrl_mode == HexRosRoboArmCtrl.CTRL_MODE_JNT
    assert list(arm_ctrl.jnt.pos) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert list(arm_ctrl.jnt.lim_vel) == [0.5]
    assert list(arm_ctrl.jnt.lim_acc) == [50.0]
    assert (arm_ctrl.grav.x, arm_ctrl.grav.y, arm_ctrl.grav.z) == (
        0.0, 0.0, -9.81)
    assert sent.manip_ctrl.grip_ctrl.ctrl_mode == (
        HexRosRoboGripCtrl.CTRL_MODE_NONE)


def test_send_ctrl_cartesian_ee_mode() -> None:
    """Arm ee command maps to CTRL_MODE_EE with target Pose."""
    device, recorder, _ = build_device()
    ee_pose = Pose(
        position=Vector3(x=0.5, y=-0.2, z=0.3),
        orientation=Quaternion(w=0.707, x=0.0, y=0.707, z=0.0),
    )
    arm_cmd = ArmCommand(ctrl_mode="ee", pose=ee_pose)
    device.send_ctrl(arm_command=arm_cmd)

    pub = recorder.publishers["left/manip_ctrl"]
    assert len(pub.published) == 1
    sent = pub.published[0]
    arm_ctrl = sent.manip_ctrl.arm_ctrl
    assert arm_ctrl.ctrl_mode == HexRosRoboArmCtrl.CTRL_MODE_EE
    assert (arm_ctrl.pose.position.x, arm_ctrl.pose.position.y) == (0.5, -0.2)
    assert arm_ctrl.pose.orientation.w == 0.707


def test_send_ctrl_mit_mode_applies_param_gains_by_default() -> None:
    """Arm mit command applies configured kp and kd parameters by default."""
    device, recorder, _ = build_device()
    arm_cmd = ArmCommand(
        ctrl_mode="mit",
        pos=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    device.send_ctrl(arm_command=arm_cmd)

    pub = recorder.publishers["left/manip_ctrl"]
    assert len(pub.published) == 1
    sent = pub.published[0]
    arm_ctrl = sent.manip_ctrl.arm_ctrl
    assert arm_ctrl.ctrl_mode == HexRosRoboArmCtrl.CTRL_MODE_MIT
    assert list(arm_ctrl.jnt.pos) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert list(arm_ctrl.jnt.vel) == [0.0] * 6
    assert list(arm_ctrl.jnt.eff) == [0.0] * 6
    assert list(arm_ctrl.jnt.kp) == [50.0, 50.0, 50.0, 50.0, 20.0, 20.0]
    assert list(arm_ctrl.jnt.kd) == [2.5, 2.5, 2.5, 2.5, 1.0, 1.0]


def test_send_ctrl_mit_mode_custom_vel_eff_and_gains() -> None:
    """Arm mit command respects per-command vel, eff, kp, and kd."""
    device, recorder, _ = build_device()
    arm_cmd = ArmCommand(
        ctrl_mode="mit",
        pos=[0.0] * 6,
        vel=[0.1] * 6,
        eff=[0.5] * 6,
        kp=[100.0] * 6,
        kd=[5.0] * 6,
    )
    device.send_ctrl(arm_command=arm_cmd)

    pub = recorder.publishers["left/manip_ctrl"]
    assert len(pub.published) == 1
    sent = pub.published[0]
    arm_ctrl = sent.manip_ctrl.arm_ctrl
    assert arm_ctrl.ctrl_mode == HexRosRoboArmCtrl.CTRL_MODE_MIT
    assert list(arm_ctrl.jnt.pos) == [0.0] * 6
    assert list(arm_ctrl.jnt.vel) == [0.1] * 6
    assert list(arm_ctrl.jnt.eff) == [0.5] * 6
    assert list(arm_ctrl.jnt.kp) == [100.0] * 6
    assert list(arm_ctrl.jnt.kd) == [5.0] * 6


def test_send_ctrl_grip_pos_mode() -> None:
    """Grip pos command maps to physical travel inside grip_limit."""
    device, recorder, _ = build_device()
    arm_cmd = ArmCommand(ctrl_mode="pos", pos=[0.0, 0.0, 0.0])
    # grip_limit: [0.0, 0.70]. -1.0 -> 0.0, 0.0 -> 0.35, 1.0 -> 0.70
    grip_cmd = GripCommand(ctrl_mode="pos", pos=[0.0])
    device.send_ctrl(arm_command=arm_cmd, grip_command=grip_cmd)

    pub = recorder.publishers["left/manip_ctrl"]
    assert len(pub.published) == 1
    sent = pub.published[0]
    grip_ctrl = sent.manip_ctrl.grip_ctrl
    assert grip_ctrl.ctrl_mode == HexRosRoboGripCtrl.CTRL_MODE_JNT
    assert list(grip_ctrl.jnt.lim_vel) == [1.0]
    assert list(grip_ctrl.jnt.eff) == [1.0]
    assert pytest.approx(grip_ctrl.jnt.pos[0]) == 0.35


def test_send_ctrl_grip_eff_mode() -> None:
    """Grip eff command maps to CTRL_MODE_TAU with effort array."""
    device, recorder, _ = build_device()
    arm_cmd = ArmCommand(ctrl_mode="pos", pos=[0.0, 0.0, 0.0])
    grip_cmd = GripCommand(ctrl_mode="eff", eff=[1.5])
    device.send_ctrl(arm_command=arm_cmd, grip_command=grip_cmd)

    pub = recorder.publishers["left/manip_ctrl"]
    assert len(pub.published) == 1
    sent = pub.published[0]
    grip_ctrl = sent.manip_ctrl.grip_ctrl
    assert grip_ctrl.ctrl_mode == HexRosRoboGripCtrl.CTRL_MODE_TAU
    assert list(grip_ctrl.jnt.eff) == [1.5]


def test_send_ctrl_combined_arm_and_grip_published_together() -> None:
    """Arm and grip commands are packed and published in a single message."""
    device, recorder, _ = build_device()
    arm_cmd = ArmCommand(ctrl_mode="pos", pos=[0.1, 0.2, 0.3])
    grip_cmd = GripCommand(ctrl_mode="pos", pos=[1.0])
    device.send_ctrl(arm_command=arm_cmd, grip_command=grip_cmd)

    pub = recorder.publishers["left/manip_ctrl"]
    # Only ONE message published
    assert len(pub.published) == 1
    sent = pub.published[0]
    assert sent.manip_ctrl.arm_ctrl.ctrl_mode == (
        HexRosRoboArmCtrl.CTRL_MODE_JNT)
    assert sent.manip_ctrl.grip_ctrl.ctrl_mode == (
        HexRosRoboGripCtrl.CTRL_MODE_JNT)
    assert pytest.approx(sent.manip_ctrl.grip_ctrl.jnt.pos[0]) == 0.70


def test_send_ctrl_raises_value_error_when_arm_command_is_none() -> None:
    """Invoking send_ctrl with arm_command=None raises ValueError."""
    device, _, _ = build_device()
    with pytest.raises(ValueError) as caught:
        device.send_ctrl(arm_command=None)  # type: ignore[arg-type]
    assert "arm_command cannot be None" in str(caught.value)


def test_send_ctrl_unsupported_mode_drops_and_warns() -> None:
    """Invalid command mode logs a warning and is dropped."""
    device, recorder, logger = build_device()
    # Construct an invalid mode command object bypassing post_init
    invalid_arm = object.__new__(ArmCommand)
    object.__setattr__(invalid_arm, "ctrl_mode", "unsupported")
    object.__setattr__(invalid_arm, "pos", None)
    object.__setattr__(invalid_arm, "pose", None)

    device.send_ctrl(arm_command=invalid_arm)
    pub = recorder.publishers["left/manip_ctrl"]
    assert len(pub.published) == 0
    assert len(logger.warnings) == 1
    assert "unsupported_mode" in logger.warnings[0]


# =====================================================================
# 5. 内存隔离与克隆（Clone 安全）
# =====================================================================

def test_clone_arm_state_arrays_isolated_from_message() -> None:
    """Observation arm joint arrays do not share memory with source message."""
    device, recorder, _ = build_device()
    cb = recorder.handles["sub:left/manip_state"]
    msg = make_state_msg(arm_pos=[0.1, 0.2, 0.3])
    cb(msg)

    obs = device.get_obs(drain=False)
    assert obs is not None
    msg_raw = np.array(msg.manip_state.arm_state.jnt.position,
                       dtype=np.float64)
    assert np.shares_memory(obs.arm_state.arm_pos, msg_raw) is False


def test_clone_grip_state_arrays_isolated_from_message() -> None:
    """Observation grip joint arrays do not share memory with message."""
    device, recorder, _ = build_device()
    cb = recorder.handles["sub:left/manip_state"]
    msg = make_state_msg(grip_pos=[0.5])
    cb(msg)

    obs = device.get_obs(drain=False)
    assert obs is not None
    msg_raw = np.array(msg.manip_state.grip_state.jnt.position,
                       dtype=np.float64)
    assert np.shares_memory(obs.grip_state.grip_pos, msg_raw) is False


def test_clone_consecutive_get_obs_isolated() -> None:
    """Multiple get_obs calls produce completely independent data arrays."""
    device, recorder, _ = build_device()
    cb = recorder.handles["sub:left/manip_state"]
    cb(make_state_msg())

    obs1 = device.get_obs(drain=False)
    obs2 = device.get_obs(drain=False)
    assert obs1 is not None and obs2 is not None
    assert np.shares_memory(obs1.arm_state.arm_pos,
                            obs2.arm_state.arm_pos) is False

    orig_val = float(obs2.arm_state.arm_pos[0])
    obs1.arm_state.arm_pos[0] = orig_val + 10.0
    assert float(obs2.arm_state.arm_pos[0]) == orig_val


def test_send_ctrl_sets_header_stamp_from_target_time_ns() -> None:
    """Target time in nanoseconds populates the ROS message header stamp."""
    device, recorder, _ = build_device()
    arm_cmd = ArmCommand(ctrl_mode="pos", pos=[0.1, 0.2, 0.3])
    target_time_ns = 1_700_000_001_500_000_000
    device.send_ctrl(arm_command=arm_cmd, target_time_ns=target_time_ns)

    pub = recorder.publishers["left/manip_ctrl"]
    assert len(pub.published) == 1
    stamp = pub.published[0].header.stamp
    assert stamp.sec == 1_700_000_001
    assert stamp.nanosec == 500_000_000


