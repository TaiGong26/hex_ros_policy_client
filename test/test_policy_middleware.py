################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the frame wire format, the schema and the transport."""

from __future__ import annotations

import numpy as np
import pytest

from hex_util_zenoh.proto_util import policy_proto_parser
from hex_util_zenoh.proto_util import policy_proto_schema
from hex_util_zenoh.proto_util.policy_proto_schema import (
    DualArmActionManipFrame,
    InferenceFrame,
    ObservationFrame,
)
from hex_ros_policy_client.protocol import policy_middleware as middleware
from hex_ros_policy_client.protocol import proto_helper
from hex_ros_policy_client.struct.device_struct import (
    ArmObservation,
    ArmState,
    CameraObservation,
    GripState,
    ImageFrame,
)
from hex_ros_policy_client.struct.geom_struct import Pose

from .fakes import Capture

pytest.importorskip("msgpack")

NS_PER_SEC = 1_000_000_000


def observation() -> dict:
    """Build an assembler output with one camera and one arm.

    Returns:
        dict: Nested observation dictionary.
    """
    frame = ImageFrame(ts_ns=1, encoding="jpeg", width=1, height=1,
                       data=np.array([1, 2, 3], dtype=np.uint8))
    arm = ArmState(arm_pos=np.zeros(2), arm_vel=np.zeros(2),
                   arm_eff=np.zeros(2), ee_base=Pose())
    grip = GripState(grip_pos=np.zeros(1), grip_vel=np.zeros(1),
                     grip_eff=np.zeros(1))
    return {
        "stamp_ns": 1234,
        "camera": {"fixed": CameraObservation(rgb=frame)},
        "arm": {"left": ArmObservation(stamp_ns=99, arm_state=arm,
                                       grip_state=grip)},
    }


def action_frame(steps: int = 2, stamp_ns: int = 0,
                 action_mode: str = "abs",
                 arm_ctrl_mode: str = "pos") -> dict:
    """Build one action frame as the model side sends it.

    Args:
        steps (int): Number of steps of the chunk.
        stamp_ns (int): Reference time in nanoseconds.
        action_mode (str): Mode of the steps.
        arm_ctrl_mode (str): Control mode of the arm.

    Returns:
        dict: Frame of the protocol.
    """
    return {
        "header": {"secs": stamp_ns // NS_PER_SEC,
                   "nsecs": stamp_ns % NS_PER_SEC},
        "chunk_step_interval_ms": 33,
        "action_mode": action_mode,
        "arm_ctrl_mode": arm_ctrl_mode,
        "arm": {"left": [[0.1 * step, 0.0] for step in range(steps)]},
        "grip": {"left": [[0.0]] * steps},
    }


def test_observation_frame_roundtrip() -> None:
    """An observation frame survives pack and unpack."""
    frame = proto_helper.build_observation_frame(observation())
    decoded = policy_proto_parser.unpack(policy_proto_parser.pack(frame))
    assert decoded["header"] == {"secs": 0, "nsecs": 1234}
    assert decoded["camera"]["fixed"]["rgb"]["encoding"] == "jpeg"
    assert decoded["arm"]["left"]["pos"] == [0.0, 0.0]
    assert "stamp_ns" not in decoded["arm"]["left"]


def test_inference_frame_carries_one_flag() -> None:
    """The enable frame is the flag alone, as the protocol declares."""
    frame = proto_helper.build_inference_frame(True)
    assert frame == {"inference_enabled": True}
    assert policy_proto_parser.unpack(policy_proto_parser.pack(frame)) == frame


def test_device_name_map_is_identity_by_default() -> None:
    """Without a mapping the device name is sent unchanged."""
    assert proto_helper.model_name("left") == "left"


def test_device_name_map_override() -> None:
    """A table entry renames the field the model receives."""
    frame = proto_helper.build_observation_frame(
        observation(), {"left": "left_arm"})
    assert "left_arm" in frame["arm"]
    assert "left" not in frame["arm"]


def test_unpack_rejects_garbage() -> None:
    """A payload that is not msgpack raises a value error."""
    with pytest.raises(ValueError):
        policy_proto_parser.unpack(b"\xc1")


def test_decode_chunk_passes_steps_through() -> None:
    """The step lists travel as the sender wrote them."""
    decoded = proto_helper.decode_action_chunk(action_frame(steps=2))
    assert decoded["arm"]["left"] == [[0.0, 0.0], [0.1, 0.0]]
    assert decoded["grip"]["left"] == [[0.0], [0.0]]
    assert decoded["chunk_step_interval_ms"] == 33
    assert decoded["action_mode"] == "abs"
    assert decoded["arm_ctrl_mode"] == "pos"


def test_decode_chunk_converts_the_stamp() -> None:
    """The header stamp becomes nanoseconds."""
    decoded = proto_helper.decode_action_chunk(
        action_frame(stamp_ns=2 * NS_PER_SEC + 7))
    assert decoded["stamp_ns"] == 2 * NS_PER_SEC + 7


def test_decode_chunk_rejects_missing_key() -> None:
    """A chunk without the arm group is rejected."""
    broken = action_frame()
    del broken["arm"]
    with pytest.raises(ValueError):
        proto_helper.decode_action_chunk(broken)


def test_decode_chunk_does_not_judge_the_group() -> None:
    """A group of the wrong type is passed on, the action layer rejects it."""
    broken = action_frame()
    broken["grip"] = [1.0, 2.0]
    decoded = proto_helper.decode_action_chunk(broken)
    assert decoded["grip"] == [1.0, 2.0]


def test_decode_chunk_does_not_judge_the_modes() -> None:
    """Unknown modes are passed on, the action layer decides about them."""
    decoded = proto_helper.decode_action_chunk(
        action_frame(action_mode="absolute"))
    assert decoded["action_mode"] == "absolute"
    decoded = proto_helper.decode_action_chunk(
        action_frame(arm_ctrl_mode="wrist"))
    assert decoded["arm_ctrl_mode"] == "wrist"


def test_decode_chunk_rejects_bad_stamp() -> None:
    """A stamp without secs and nsecs is rejected."""
    broken = action_frame()
    broken["header"] = {"stamp_ns": 1}
    with pytest.raises(ValueError):
        proto_helper.decode_action_chunk(broken)


def test_frame_keys_come_from_the_frame_type() -> None:
    """The field list has no second copy outside the TypedDict."""
    keys = policy_proto_parser.frame_keys(InferenceFrame)
    assert keys == {"inference_enabled"}
    assert policy_proto_parser.frame_keys(DualArmActionManipFrame) == {
        "header", "chunk_step_interval_ms", "action_mode", "arm_ctrl_mode",
        "arm", "grip"}
    assert policy_proto_parser.frame_keys(ObservationFrame) == {
        "header", "camera", "arm"}


def build_middleware(logger: Capture | None = None,
                     capacity: int = 4) -> middleware.PolicyMiddleware:
    """Create a middleware instance without opening a session.

    Args:
        logger (Capture | None): Logger for drop lines.
        capacity (int): Chunk queue capacity.

    Returns:
        PolicyMiddleware: Transport instance.
    """
    return middleware.PolicyMiddleware("policy/observation", "policy/action",
                                       "policy/inference",
                                       capacity=capacity,
                                       logger=logger or Capture())


def test_queue_is_empty_at_start() -> None:
    """Nothing is queued before a frame arrives."""
    assert build_middleware().request_chunk() is None


def test_capacity_must_be_positive() -> None:
    """A non positive capacity fails at construction."""
    with pytest.raises(ValueError):
        build_middleware(capacity=0)


def test_action_payload_is_queued() -> None:
    """A decoded chunk is handed to the main loop through the queue."""
    instance = build_middleware()
    payload = policy_proto_parser.pack(action_frame())
    instance._PolicyMiddleware__on_action_payload(payload)
    chunk = instance.request_chunk()
    assert chunk["action_mode"] == "abs"
    assert chunk["arm"]["left"] == [[0.0, 0.0], [0.1, 0.0]]


def test_bad_action_payload_is_dropped() -> None:
    """A broken chunk is logged and does not reach the queue."""
    logger = Capture()
    instance = build_middleware(logger)
    instance._PolicyMiddleware__on_action_payload(b"\xc1")
    assert instance.request_chunk() is None
    assert logger.messages()[0].startswith("[frame.schema/schema_invalid] ")


def test_queue_overflow_drops_the_oldest() -> None:
    """The queue keeps its capacity and discards the oldest entry."""
    logger = Capture()
    capacity = 3
    instance = build_middleware(logger, capacity=capacity)
    for step in range(capacity + 1):
        instance._PolicyMiddleware__on_action_payload(
            policy_proto_parser.pack(action_frame(steps=1)))
    assert instance.request_chunk() is not None
    assert "chunk.queue/full" in logger.messages()[-1]


def test_publish_without_session_is_safe() -> None:
    """Publishing before open reports failure instead of raising."""
    instance = build_middleware()
    frame = proto_helper.build_inference_frame(enabled=True)
    assert instance.pub_inference_state(frame=frame) is False


def test_transport_holds_one_util_node() -> None:
    """The transport owns a HexZenohNode instead of reimplementing Zenoh."""
    from hex_util_zenoh import HexZenohNode

    instance = build_middleware()
    holder = instance._PolicyMiddleware__mid_ware
    assert isinstance(holder, HexZenohNode)
    assert not issubclass(middleware.PolicyMiddleware, HexZenohNode)


class _Publisher:
    """Stand-in of a Zenoh publisher that records what it is given."""

    def __init__(self, topic: str, sink: list) -> None:
        self.topic = topic
        self.sink = sink

    def put(self, data: bytes) -> None:
        """Record one payload.

        Args:
            data (bytes): Payload handed to the link.
        """
        self.sink.append((self.topic, data))


class FakeZenoh:
    """Stand-in of ``HexZenohNode`` that keeps its real behaviour.

    ``create_pub`` mirrors the real one, which stores the publisher and does
    not return it on the first call. A test that reaches ``pub`` before the
    endpoint exists therefore fails the same way the real node would.
    """

    def __init__(self, config=None) -> None:
        self.publishers: dict = {}
        self.callbacks: dict = {}
        self.sent: list = []
        self.opened = 0
        self.closed = 0
        self.__working = False

    @property
    def is_working(self) -> bool:
        """Whether a session is open.

        Returns:
            bool: ``True`` between ``start`` and ``close``.
        """
        return self.__working

    def start(self) -> None:
        """Open the session."""
        self.opened += 1
        self.__working = True

    def close(self) -> None:
        """Close the session."""
        self.closed += 1
        self.__working = False

    def create_pub(self, topic_name: str):
        """Declare one publisher exactly as the real node does.

        Args:
            topic_name (str): Key of the link.
        """
        if topic_name in self.publishers:
            return self.publishers[topic_name]
        self.publishers[topic_name] = _Publisher(topic_name, self.sent)

    def create_sub(self, topic_name: str, callback) -> None:
        """Declare one subscriber.

        Args:
            topic_name (str): Key of the link.
            callback (callable): Handler of one payload.
        """
        self.callbacks[topic_name] = callback

    def pub(self, topic_name: str, data: bytes) -> None:
        """Publish one payload the way the real node does.

        Args:
            topic_name (str): Key of the link.
            data (bytes): Payload to hand over.
        """
        publisher = self.publishers.get(topic_name)
        if publisher is None:
            publisher = self.create_pub(topic_name)
        publisher.put(data)


def with_fake_node(monkeypatch, capacity: int = 4, logger=None):
    """Build a middleware whose session is the faithful stand-in.

    Args:
        monkeypatch: Pytest fixture.
        capacity (int): Chunk queue capacity.
        logger (Capture | None): Logger that receives the drop lines.

    Returns:
        tuple: Middleware and its stand-in node.
    """
    fake = FakeZenoh()
    monkeypatch.setattr(middleware, "HexZenohNode", lambda config=None: fake)
    return build_middleware(logger=logger, capacity=capacity), fake


def test_start_declares_both_publishers_and_the_action_subscriber(
        monkeypatch) -> None:
    """The usage the node relies on: both links are declared on start."""
    instance, fake = with_fake_node(monkeypatch)
    instance.start()
    assert fake.opened == 1
    assert "policy/observation" in fake.publishers
    assert "policy/inference" in fake.publishers
    assert "policy/action" in fake.callbacks


def test_observation_frames_reach_the_link_after_start(monkeypatch) -> None:
    """Publishing twice must both land, which needs the endpoint declared."""
    instance, fake = with_fake_node(monkeypatch)
    instance.start()
    frame = proto_helper.build_observation_frame(
        observation=observation(), name_map=None)
    assert instance.pub_obs(frame=frame) is True
    assert instance.pub_obs(frame=frame) is True
    assert [topic for topic, _ in fake.sent] == ["policy/observation"] * 2
    decoded = policy_proto_parser.unpack(fake.sent[0][1])
    assert set(decoded) == set(
        policy_proto_parser.frame_keys(ObservationFrame))


def test_inference_frames_reach_the_link_after_start(monkeypatch) -> None:
    """The enable flag is published every tick, so it must survive repeats."""
    instance, fake = with_fake_node(monkeypatch)
    instance.start()
    frame = proto_helper.build_inference_frame(enabled=True)
    assert instance.pub_inference_state(frame=frame) is True
    assert instance.pub_inference_state(frame=frame) is True
    assert [topic for topic, _ in fake.sent] == ["policy/inference"] * 2
    assert policy_proto_parser.unpack(fake.sent[0][1]) == {
        "inference_enabled": True}


def test_malformed_action_payload_is_dropped_and_logged(monkeypatch) -> None:
    """A payload that is not a chunk never reaches the queue."""
    logger = Capture()
    instance, fake = with_fake_node(monkeypatch, logger=logger)
    instance.start()
    callback = fake.callbacks["policy/action"]
    callback(b"not-a-frame")
    assert instance.request_chunk() is None
    assert any("frame.schema/schema_invalid" in line
               for line in logger.messages())


def test_close_releases_the_session(monkeypatch) -> None:
    """Closing the transport closes the session once."""
    instance, fake = with_fake_node(monkeypatch)
    instance.start()
    instance.close()
    assert fake.closed == 1
    assert fake.is_working is False


def test_request_chunk_returns_the_newest_and_drains_the_queue() -> None:
    """A newer plan supersedes the older ones, so only it is handed over.

    The queue keeps the frames as they arrived; reading them is the job of the
    action business, so the assertion compares frames, not decoded chunks.
    """
    instance = build_middleware(capacity=4)
    frames = [action_frame(steps=1, stamp_ns=stamp) for stamp in (1, 2, 3)]
    for frame in frames:
        instance._PolicyMiddleware__on_action_payload(
            policy_proto_parser.pack(frame))
    assert instance.request_chunk() == frames[2]
    assert instance.request_chunk() is None
