################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Offline replay of one task segment.

No ROS runtime is started: the device callbacks are called by hand with
synthetic messages, the clock is driven by the test and the transport is
replaced by a recorder. The test then checks the two promises of the design:
the observation rate follows the decimation and the dispatched command never
jumps more than one threshold per tick.
"""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.policy_client_base import (
    EVENT_START,
    STATE_RUNNING,
    STATE_WAITING,
)
from hex_util_zenoh.proto_util import policy_proto_parser
from hex_ros_policy_client.protocol import policy_middleware as middleware
from hex_ros_policy_client.utils.common_util.logkit import (
    _reset_state as reset_log,
)

from .fakes import FakeRosNode
from .test_device_hub import write_table
from .test_dual_node import BASE_PARAMS

from types import SimpleNamespace

import hex_ros_msgs.msg as ros_msgs

pytest.importorskip("hex_ros_msgs.msg")
pytest.importorskip("sensor_msgs.msg")

NS_PER_MS = 1_000_000
TICK_MS = 1
TICKS = 1000
OBS_RATE_HZ = 30.0
TOLERANCE = 2

REPLAY_TABLE = """
cameras:
  fixed:
    type: camera
    color_topic: fixed/image_color_raw
    resize_w: 4
    resize_h: 4
arms:
  main:
    type: arm
    state_topic: main/manip_state
    ctrl_topic: main/manip_ctrl
    grip_limit: [0.0, 1.0]
"""


class ImageDouble(SimpleNamespace):
    """Message double shaped like ``sensor_msgs/Image``."""


def state_message(stamp_ns: int, joints: np.ndarray):
    """Build one manipulation state message.

    Args:
        stamp_ns (int): Stamp of the message.
        joints (np.ndarray): Joint positions.

    Returns:
        Any: Filled ``HexRosRoboManipStateStamped``.
    """
    message = ros_msgs.HexRosRoboManipStateStamped()
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    arm = message.manip_state.arm_state
    arm.jnt.position = joints.tolist()
    arm.jnt.velocity = [0.0] * len(joints)
    arm.jnt.effort = [0.0] * len(joints)
    grip = message.manip_state.grip_state
    grip.jnt.position = [0.0]
    grip.jnt.velocity = [0.0]
    grip.jnt.effort = [0.0]
    return message


def image_message(stamp_ns: int) -> ImageDouble:
    """Build one colour image message.

    Args:
        stamp_ns (int): Stamp of the message.

    Returns:
        ImageDouble: Message stand-in with a 8x8 image.
    """
    data = np.zeros((8, 8, 3), dtype=np.uint8).tobytes()
    return ImageDouble(
        data=data, encoding="bgr8", width=8, height=8,
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000,
                                  nanosec=stamp_ns % 1_000_000_000)))


def build_replay(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Create a running client on a replay friendly table.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch the transport.
        tmp_path: Pytest temporary directory.

    Returns:
        tuple: Client, fake node and the recorded frames.
    """
    record = {"obs": [], "inference": []}
    monkeypatch.setattr(middleware.PolicyMiddleware, "start",
                        lambda self: None)
    monkeypatch.setattr(middleware.PolicyMiddleware, "close",
                        lambda self: None)
    monkeypatch.setattr(
        middleware.PolicyMiddleware, "pub_obs",
        lambda self, frame: record["obs"].append(frame) is None)
    monkeypatch.setattr(
        middleware.PolicyMiddleware, "pub_inference_state",
        lambda self, frame:
        record["inference"].append(frame["inference_enabled"]) is None)
    params = dict(BASE_PARAMS)
    params.update({"device_path": write_table(tmp_path, REPLAY_TABLE),
                   "stable_arm_order": ["main"],
                   "stable_jnt": [0.0] * 7,
                   "observation_rate_hz": OBS_RATE_HZ,
                   "control_rate_hz": 1000.0})
    node = FakeRosNode(params=params, now_ns=1_000_000_000)
    from hex_ros_policy_client.dual_manip_policy_client import (
        DualManipPolicyClient,
    )
    client = DualManipPolicyClient(node)
    client.start()
    return client, node, record


def feed_devices(client, stamp_ns: int, joints: np.ndarray) -> None:
    """Deliver one round of device messages.

    Args:
        client: Client under test.
        stamp_ns (int): Stamp of the messages.
        joints (np.ndarray): Measured joint positions.
    """
    client.robot.device_hub.arm_devices["main"].on_states(
        state_message(stamp_ns, joints))
    client.robot.device_hub.camera_devices["fixed"].on_states(
        image_message(stamp_ns))


def test_replay_one_segment(monkeypatch, tmp_path) -> None:
    """One second of replay produces the expected frames and no jump."""
    reset_log()
    client, node, record = build_replay(monkeypatch, tmp_path)
    joints = np.zeros(7)
    commands = []
    client.robot.send_ctrl = lambda target: commands.append(target)
    feed_devices(client, node.clock.now_ns, joints)
    client.step()
    assert client.state_machine.state == "RETURN_TO_STABLE"
    node.clock.now_ns += 3_000_000_000
    feed_devices(client, node.clock.now_ns, joints)
    client.step()
    assert client.state_machine.state == STATE_WAITING
    payload = policy_proto_parser.pack({"kind": "start"})
    client.policy_middleware._PolicyMiddleware__on_inference_payload(payload)
    client.step()
    assert client.state_machine.state == STATE_RUNNING
    chunk = {
        "header": {"secs": 0, "nsecs": node.clock.now_ns},
        "chunk_step_interval_ms": 33,
        "action_mode": "joint",
        "arm_ctrl_mode": "pos",
        "arm": {"main": [[1.0] * 7 for _ in range(30)]},
        "grip": {"main": [[0.5] for _ in range(30)]},
    }
    client.policy_middleware._PolicyMiddleware__on_action_payload(
        policy_proto_parser.pack(chunk))
    for _ in range(TICKS):
        node.clock.now_ns += TICK_MS * NS_PER_MS
        feed_devices(client, node.clock.now_ns, joints)
        client.step()
    duration_s = TICKS * TICK_MS / 1000.0
    expected = duration_s * OBS_RATE_HZ
    assert abs(len(record["obs"]) - expected) <= TOLERANCE
    assert commands, "no command was dispatched during the segment"
    threshold = BASE_PARAMS["arm_jnt_threshold"]
    for target in commands:
        for command in target["arm"].values():
            assert np.max(np.abs(np.asarray(command.pos))) <= threshold + 1e-9
    assert record["inference"][-1] is True


def test_replay_stops_publishing_after_the_task(monkeypatch, tmp_path) -> None:
    """Leaving the running state silences the observation link."""
    reset_log()
    client, node, record = build_replay(monkeypatch, tmp_path)
    joints = np.zeros(7)
    feed_devices(client, node.clock.now_ns, joints)
    client.step()
    node.clock.now_ns += 3_000_000_000
    feed_devices(client, node.clock.now_ns, joints)
    client.step()
    assert client.state_machine.state == STATE_WAITING
    for _ in range(60):
        node.clock.now_ns += TICK_MS * NS_PER_MS
        feed_devices(client, node.clock.now_ns, joints)
        client.step()
    assert record["obs"] == []
    assert record["inference"][-1] is False
