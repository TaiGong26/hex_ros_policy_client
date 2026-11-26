################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-24
################################################################
"""Unit tests for the rewritten CameraDevice."""

from __future__ import annotations

import sys
import types
from array import array
from typing import Any

import numpy as np
import pytest

# Provide sensor_msgs.msg.Image stub if sensor_msgs is absent in environment
try:
    from sensor_msgs.msg import Image
except ImportError:
    _sensor_msgs = types.ModuleType("sensor_msgs")
    _sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

    class Image:
        """Stand-in for sensor_msgs/Image."""

    _sensor_msgs_msg.Image = Image
    _sensor_msgs.msg = _sensor_msgs_msg
    sys.modules.setdefault("sensor_msgs", _sensor_msgs)
    sys.modules.setdefault("sensor_msgs.msg", _sensor_msgs_msg)

from hex_ros_policy_client.device.camera_device import (
    CROP_MODES,
    CameraDevice,
)


class EndpointRecorder:
    """Record endpoint creation calls and store callable handles."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.handles: dict[str, Any] = {}

    def create_subscription(self, msg_type: Any, topic: str,
                            callback: Any) -> str:
        self.calls.append(("sub", msg_type, topic, callback.__name__))
        handle = f"sub:{topic}"
        self.handles[handle] = callback
        return handle

    def create_publisher(self, msg_type: Any, topic: str) -> str:
        self.calls.append(("pub", msg_type, topic))
        return f"pub:{topic}"


class CaptureLogger:
    """Capture logger warning calls."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def logw(self, message: str) -> None:
        self.warning(message)


def build_camera(name: str = "overhead",
                 params: dict | None = None) -> tuple[
                     CameraDevice, EndpointRecorder, CaptureLogger]:
    """Helper to instantiate a CameraDevice with fake endpoints and logger."""
    default_params = {
        "ros_cfg": {
            "namespace": name,
            "subscribers": {
                "color_topic": "image_color_raw",
                "depth_topic": None,
            },
        },
        "crop_mode": "cover",
        "resize_w": 128,
        "resize_h": 128,
        "jpeg_quality": 90,
    }
    cfg = params if params is not None else default_params
    recorder = EndpointRecorder()
    logger = CaptureLogger()
    device = CameraDevice(
        name=name,
        params=cfg,
        create_subscription=recorder.create_subscription,
        create_publisher=recorder.create_publisher,
        logger=logger,
    )
    return device, recorder, logger


def make_colour_msg(width: int = 160,
                    height: int = 120,
                    encoding: str = "bgr8",
                    stamp: tuple[int, int] = (10, 0)) -> Any:
    """Create a mock ROS Image message for colour frames."""
    raw = bytes(np.random.randint(0, 256, width * height * 3, dtype=np.uint8))
    return types.SimpleNamespace(
        data=array("B", raw),
        encoding=encoding,
        width=width,
        height=height,
        header=types.SimpleNamespace(
            stamp=types.SimpleNamespace(sec=stamp[0], nanosec=stamp[1]),
        ),
    )


def make_depth_msg(width: int = 160,
                   height: int = 120,
                   stamp: tuple[int, int] = (10, 0)) -> Any:
    """Create a mock ROS Image message for depth frames."""
    arr = np.random.randint(0, 65535, width * height, dtype=np.uint16)
    return types.SimpleNamespace(
        data=arr.tobytes(),
        encoding="mono16",
        width=width,
        height=height,
        header=types.SimpleNamespace(
            stamp=types.SimpleNamespace(sec=stamp[0], nanosec=stamp[1]),
        ),
    )


# =====================================================================
# 1. 构造与配置解析（Fail-fast 与边界）
# =====================================================================

def test_init_with_color_only() -> None:
    """Device without depth creates one subscription and no publishers."""
    device, recorder, _ = build_camera("overhead")
    assert device.name == "overhead"
    assert "color_topic" in device.subscribers
    assert "depth_topic" not in device.subscribers
    assert device.publishers == {}
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call == (
        "sub", Image, "overhead/image_color_raw", "on_color_states")


def test_init_default_params_fallback() -> None:
    """Missing optional params default to 128, 128, 90, cover."""
    params = {
        "ros_cfg": {
            "namespace": "overhead",
            "subscribers": {"color_topic": "image_color_raw"},
        },
    }
    device, _, _ = build_camera(params=params)
    stored = device._CameraDevice__params
    assert stored["crop_mode"] == "cover"
    assert stored["resize_w"] == 128
    assert stored["resize_h"] == 128
    assert stored["jpeg_quality"] == 90


def test_init_rejects_unsupported_crop_mode() -> None:
    """Unsupported crop mode fails fast during init."""
    params = {
        "ros_cfg": {"subscribers": {"color_topic": "img"}},
        "crop_mode": "stretch",
    }
    with pytest.raises(ValueError) as caught:
        build_camera(params=params)
    assert "unsupported crop_mode 'stretch'" in str(caught.value)


def test_init_rejects_missing_ros_cfg() -> None:
    """Entry without ros_cfg is rejected by base class."""
    with pytest.raises(ValueError) as caught:
        build_camera(params={"crop_mode": "cover"})
    assert "missing required keys ['ros_cfg']" in str(caught.value)


def test_init_rejects_empty_endpoints() -> None:
    """Declaring only None topics results in no endpoint and fails fast."""
    params = {
        "ros_cfg": {
            "subscribers": {"color_topic": None, "depth_topic": None},
        },
    }
    with pytest.raises(ValueError) as caught:
        build_camera(params=params)
    assert "declares neither a subscriber nor a publisher" in str(caught.value)


# =====================================================================
# 2. 队列机制与二分单调性（Monotonic Ordering）
# =====================================================================

def test_enqueue_maintains_monotonic_stamp_ordering() -> None:
    """OutOfOrder arriving images are placed in sorted stamp order."""
    device, recorder, _ = build_camera()
    cb = recorder.handles["sub:overhead/image_color_raw"]
    cb(make_colour_msg(stamp=(10, 0)))
    cb(make_colour_msg(stamp=(30, 0)))
    cb(make_colour_msg(stamp=(20, 0)))

    queue = device._CameraDevice__queues["color"]
    timestamps = [item[0] for item in queue]
    assert timestamps == [10_000_000_000, 20_000_000_000, 30_000_000_000]


def test_enqueue_respects_queue_maxlen() -> None:
    """Exceeding queue maxlen evicts the oldest item."""
    device, recorder, _ = build_camera()
    cb = recorder.handles["sub:overhead/image_color_raw"]
    for sec in range(35):
        cb(make_colour_msg(stamp=(sec, 0)))

    queue = device._CameraDevice__queues["color"]
    assert len(queue) == 30
    assert queue[0][0] == 5 * 1_000_000_000
    assert queue[-1][0] == 34 * 1_000_000_000


# =====================================================================
# 3. depth 话题自洽与零开销（按需创建）
# =====================================================================

def test_depth_null_is_completely_idle() -> None:
    """When depth is None, no depth queue exists and calls short circuit."""
    device, _, _ = build_camera()
    assert "depth" not in device._CameraDevice__queues
    # Invoking depth handler should be a no-op
    device.on_depth_states(make_depth_msg())
    assert "depth" not in device._CameraDevice__queues


def test_depth_configured_creates_queue_and_receives() -> None:
    """When depth topic is configured, depth queue is created and filled."""
    params = {
        "ros_cfg": {
            "namespace": "overhead",
            "subscribers": {
                "color_topic": "image_color_raw",
                "depth_topic": "image_depth_raw",
            },
        },
    }
    device, recorder, _ = build_camera(params=params)
    assert "depth" in device._CameraDevice__queues
    d_cb = recorder.handles["sub:overhead/image_depth_raw"]
    d_cb(make_depth_msg(stamp=(15, 0)))
    assert len(device._CameraDevice__queues["depth"]) == 1
    assert device._CameraDevice__queues["depth"][0][0] == 15 * 1_000_000_000


# =====================================================================
# 4. 图像几何变换（cover vs resize）
# =====================================================================

def test_crop_mode_cover_crops_centred_overflow() -> None:
    """Cover mode scales to fill target and crops centred overflow."""
    params = {
        "ros_cfg": {"subscribers": {"color_topic": "img"}},
        "crop_mode": "cover",
        "resize_w": 100,
        "resize_h": 100,
    }
    device, recorder, _ = build_camera(params=params)
    cb = recorder.handles["sub:img"]
    # Wide image: 200x100 -> scaled to 200x100, left/right cropped 50px
    cb(make_colour_msg(width=200, height=100))
    obs = device.get_obs()
    assert obs is not None
    assert obs.rgb.width == 100
    assert obs.rgb.height == 100


def test_crop_mode_resize_stretches_axes() -> None:
    """Resize mode stretches axes independently to target size."""
    params = {
        "ros_cfg": {"subscribers": {"color_topic": "img"}},
        "crop_mode": "resize",
        "resize_w": 80,
        "resize_h": 60,
    }
    device, recorder, _ = build_camera(params=params)
    cb = recorder.handles["sub:img"]
    cb(make_colour_msg(width=320, height=240))
    obs = device.get_obs()
    assert obs is not None
    assert obs.rgb.width == 80
    assert obs.rgb.height == 60


# =====================================================================
# 5. get_obs(drain) 语义与空队列保护
# =====================================================================

def test_get_obs_empty_queue_returns_none() -> None:
    """Empty queue safely returns None without raising exceptions."""
    device, _, _ = build_camera()
    assert device.get_obs() is None


def test_get_obs_drain_true_clears_queue() -> None:
    """get_obs(drain=True) clears the queue upon retrieval."""
    device, recorder, _ = build_camera()
    cb = recorder.handles["sub:overhead/image_color_raw"]
    cb(make_colour_msg(stamp=(10, 0)))
    obs = device.get_obs(drain=True)
    assert obs is not None
    assert len(device._CameraDevice__queues["color"]) == 0
    assert device.get_obs() is None


def test_get_obs_drain_false_retains_queue() -> None:
    """get_obs(drain=False) keeps queue intact for subsequent reads."""
    device, recorder, _ = build_camera()
    cb = recorder.handles["sub:overhead/image_color_raw"]
    cb(make_colour_msg(stamp=(10, 0)))
    obs1 = device.get_obs(drain=False)
    assert obs1 is not None
    assert len(device._CameraDevice__queues["color"]) == 1
    obs2 = device.get_obs(drain=False)
    assert obs2 is not None


# =====================================================================
# 6. 解码容错与告警（运行时坏数据丢弃）
# =====================================================================

def test_decode_invalid_encoding_logs_warning_and_drops() -> None:
    """Message with unknown encoding logs a warning and returns None."""
    device, recorder, logger = build_camera()
    cb = recorder.handles["sub:overhead/image_color_raw"]
    cb(make_colour_msg(encoding="unsupported_encoding"))
    obs = device.get_obs()
    assert obs is None
    assert len(logger.warnings) == 1
    assert "decode_failed" in logger.warnings[0]


def test_decode_corrupted_buffer_size_drops_frame() -> None:
    """Corrupted buffer size is rejected and logs decode_failed warning."""
    device, recorder, logger = build_camera()
    cb = recorder.handles["sub:overhead/image_color_raw"]
    msg = make_colour_msg(width=100, height=100)
    msg.data = array("B", bytes(10))  # truncated buffer
    cb(msg)
    obs = device.get_obs()
    assert obs is None
    assert len(logger.warnings) == 1
    assert "decode_failed" in logger.warnings[0]


# =====================================================================
# 7. 内存隔离与克隆（Anti-sharing / Zero-leak）
# =====================================================================

def test_clone_rgb_isolation_from_message_buffer() -> None:
    """Output rgb array does not share memory with incoming message."""
    device, recorder, _ = build_camera()
    cb = recorder.handles["sub:overhead/image_color_raw"]
    msg = make_colour_msg()
    cb(msg)
    obs = device.get_obs(drain=False)
    msg_raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    assert np.shares_memory(obs.rgb.data, msg_raw) is False


def test_clone_rgb_multiple_calls_isolated() -> None:
    """Consecutive get_obs calls produce completely independent data arrays."""
    device, recorder, _ = build_camera()
    cb = recorder.handles["sub:overhead/image_color_raw"]
    cb(make_colour_msg())
    obs1 = device.get_obs(drain=False)
    obs2 = device.get_obs(drain=False)
    assert np.shares_memory(obs1.rgb.data, obs2.rgb.data) is False

    orig_val = int(obs2.rgb.data[0])
    obs1.rgb.data[0] = (orig_val + 50) % 256
    assert int(obs2.rgb.data[0]) == orig_val


def test_clone_depth_isolation_and_independence() -> None:
    """Depth data array is isolated from raw message and other frames."""
    params = {
        "ros_cfg": {
            "namespace": "overhead",
            "subscribers": {
                "color_topic": "image_color_raw",
                "depth_topic": "image_depth_raw",
            },
        },
    }
    device, recorder, _ = build_camera(params=params)
    c_cb = recorder.handles["sub:overhead/image_color_raw"]
    d_cb = recorder.handles["sub:overhead/image_depth_raw"]
    d_msg = make_depth_msg()
    d_cb(d_msg)
    c_cb(make_colour_msg())

    obs = device.get_obs(drain=False)
    assert obs.depth is not None
    d_raw = np.frombuffer(d_msg.data, dtype=np.uint8)
    assert np.shares_memory(obs.depth.data, d_raw) is False

    obs1 = device.get_obs(drain=False)
    obs2 = device.get_obs(drain=False)
    assert np.shares_memory(obs1.depth.data, obs2.depth.data) is False
    orig_val = int(obs2.depth.data[0])
    obs1.depth.data[0] = (orig_val + 30) % 256
    assert int(obs2.depth.data[0]) == orig_val
