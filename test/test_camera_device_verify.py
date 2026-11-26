"""Independent checks of ``CameraDevice``.

One block per behaviour: construction, queue order, depth stream, geometry,
drain semantics, decode tolerance and buffer isolation. The device table is
read from the real ``config/ros2/dual_manip_devices.yaml`` so the checks
fail if the shipped configuration stops matching the device.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    from sensor_msgs.msg import Image
except ImportError:
    _sensor_msgs = types.ModuleType("sensor_msgs")
    _sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

    class Image:
        """Stand-in for ``sensor_msgs/Image``."""

    _sensor_msgs_msg.Image = Image
    _sensor_msgs.msg = _sensor_msgs_msg
    sys.modules.setdefault("sensor_msgs", _sensor_msgs)
    sys.modules.setdefault("sensor_msgs.msg", _sensor_msgs_msg)
    from sensor_msgs.msg import Image

yaml = pytest.importorskip("yaml")

from hex_ros_policy_client.device.camera_device import (  # noqa: E402
    CROP_MODES,
    DEPTH_ENCODING,
    JPEG_ENCODING,
    CameraDevice,
)

SEC_NS = 1_000_000_000
QUEUE_CAPACITY = 30
DEFAULT_RESIZE_W = 128
DEFAULT_RESIZE_H = 128
DEFAULT_JPEG_QUALITY = 90
COLOR_KEY = "color_topic"
DEPTH_KEY = "depth_topic"
COLOR_STREAM = "color"
DEPTH_STREAM = "depth"
DEVICE_TABLE = (Path(__file__).resolve().parents[1]
                / "config" / "ros2" / "dual_manip_devices.yaml")
PROBE_ROS_CFG = {"namespace": "probe",
                 "subscribers": {COLOR_KEY: "image_color_raw"}}
DEPTH_ROS_CFG = {"namespace": "probe",
                 "subscribers": {COLOR_KEY: "image_color_raw",
                                 DEPTH_KEY: "image_depth_raw"}}


class EndpointRecorder:
    """Record endpoint creation and keep the callbacks that were handed over.

    Attributes:
        calls (list): One tuple per endpoint, in creation order.
        handles (dict): Topic name to the callback of its subscription.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.handles: dict[str, Any] = {}

    def create_subscription(self, msg_type: Any, topic: str,
                            callback: Any) -> str:
        self.calls.append(("subscription", msg_type, topic))
        handle = f"sub:{topic}"
        self.handles[handle] = callback
        return handle

    def create_publisher(self, msg_type: Any, topic: str) -> str:
        self.calls.append(("publisher", msg_type, topic))
        return f"pub:{topic}"


class WarningCapture:
    """Stand-in for the logger object the device writes warnings to.

    Attributes:
        lines (list): Messages passed to ``warning``, in arrival order.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def warning(self, message: str) -> None:
        self.lines.append(message)

    def logw(self, message: str) -> None:
        self.warning(message)


def camera_table() -> dict:
    """Read the camera section of the shipped device table.

    Returns:
        dict: Device name to its entry.
    """
    return yaml.safe_load(DEVICE_TABLE.read_text(encoding="utf-8"))["cameras"]


def build_device(params: dict | None = None,
                 name: str = "overhead") -> tuple:
    """Create a camera from the shipped table or from an explicit entry.

    Args:
        params (dict | None): Device entry; the shipped one is used when
            omitted.
        name (str): Device name.

    Returns:
        tuple: Device, endpoint recorder and warning capture.
    """
    recorder = EndpointRecorder()
    logger = WarningCapture()
    device = CameraDevice(
        name=name,
        params=camera_table()[name] if params is None else params,
        create_subscription=recorder.create_subscription,
        create_publisher=recorder.create_publisher,
        logger=logger,
    )
    return device, recorder, logger


def runtime_params(crop_mode: str = CROP_MODES[0],
                   resize_w: int = 100,
                   resize_h: int = 100,
                   ros_cfg: dict | None = None) -> dict:
    """Build a device entry with explicit image settings.

    Args:
        crop_mode (str): Geometry mode.
        resize_w (int): Target width in pixels.
        resize_h (int): Target height in pixels.
        ros_cfg (dict | None): ROS part, a colour only one when omitted.

    Returns:
        dict: Device entry.
    """
    return {
        "ros_cfg": PROBE_ROS_CFG if ros_cfg is None else ros_cfg,
        "crop_mode": crop_mode,
        "resize_w": resize_w,
        "resize_h": resize_h,
    }


def make_image(data: Any, width: int, height: int,
               encoding: str = "bgr8", ts_ns: int = 0) -> Any:
    """Build a message shaped like ``sensor_msgs/Image``.

    Args:
        data: Byte buffer of the image.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        encoding (str): ROS encoding string.
        ts_ns (int): Stamp in nanoseconds.

    Returns:
        Any: Message stand-in.
    """
    stamp = types.SimpleNamespace(sec=ts_ns // SEC_NS,
                                  nanosec=ts_ns % SEC_NS)
    return types.SimpleNamespace(
        data=data,
        encoding=encoding,
        width=width,
        height=height,
        header=types.SimpleNamespace(stamp=stamp),
    )


def colour_message(width: int = 64,
                   height: int = 48,
                   ts_ns: int = 0,
                   seed: int = 0) -> Any:
    """Build one colour message with reproducible noise.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        ts_ns (int): Stamp in nanoseconds.
        seed (int): Seed of the noise.

    Returns:
        Any: Message stand-in holding a mutable buffer.
    """
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, width * height * 3, dtype=np.uint8)
    return make_image(data=bytearray(payload.tobytes()),
                      width=width, height=height, encoding="bgr8",
                      ts_ns=ts_ns)


def depth_message(width: int = 64,
                  height: int = 48,
                  ts_ns: int = 0) -> Any:
    """Build one depth message of constant distance.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        ts_ns (int): Stamp in nanoseconds.

    Returns:
        Any: Message stand-in holding ``uint16`` pixels.
    """
    payload = np.full(width * height, 1000, dtype=np.uint16)
    return make_image(data=payload, width=width, height=height,
                      encoding="mono16", ts_ns=ts_ns)


def gradient(width: int, height: int, channels: int) -> np.ndarray:
    """Build an image whose pixels all differ.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        channels (int): Number of channels.

    Returns:
        np.ndarray: Image of shape ``(height, width, channels)``.
    """
    total = width * height * channels
    return (np.arange(total) % 251).astype(np.uint8).reshape(
        height, width, channels)


def queue_stamps(device: CameraDevice, stream: str) -> list[int]:
    """Read the stamps currently held by one queue.

    Args:
        device (CameraDevice): Device under test.
        stream (str): ``color`` or ``depth``.

    Returns:
        list[int]: Stamps in nanoseconds, oldest first.
    """
    return [item[0] for item in device._CameraDevice__queues[stream]]


def queue_names(device: CameraDevice) -> set:
    """Read which queues the device created.

    Args:
        device (CameraDevice): Device under test.

    Returns:
        set: Stream names.
    """
    return set(device._CameraDevice__queues)


def image_settings(device: CameraDevice) -> dict:
    """Read the image settings the device resolved.

    Args:
        device (CameraDevice): Device under test.

    Returns:
        dict: Setting name to value.
    """
    return dict(device._CameraDevice__params)


def apply_geometry(device: CameraDevice, image: np.ndarray,
                   nearest: bool = False) -> np.ndarray:
    """Run the geometry step of the device in isolation.

    Args:
        device (CameraDevice): Device under test.
        image (np.ndarray): Source image.
        nearest (bool): Force nearest neighbour, required for depth.

    Returns:
        np.ndarray: Image brought to the target size.
    """
    return device._CameraDevice__apply_geometry(image=image,
                                                nearest=nearest)


# ── 1. construction and configuration ─────────────────────────────
@pytest.mark.parametrize("name,topic", [
    ("overhead", "overhead/image_color_raw"),
    ("left_wrist", "left_wrist/image_color_raw"),
    ("right_wrist", "right_wrist/image_color_raw"),
])
def test_shipped_table_declares_only_the_colour_stream(name: str,
                                                       topic: str) -> None:
    device, recorder, _ = build_device(name=name)
    assert device.subscribers == {COLOR_KEY: f"sub:{topic}"}
    assert device.publishers == {}
    assert DEPTH_KEY not in device.subscribers
    assert queue_names(device) == {COLOR_STREAM}
    assert recorder.calls == [("subscription", Image, topic)]


def test_absent_settings_take_the_documented_defaults() -> None:
    params = {"ros_cfg": PROBE_ROS_CFG}
    device, _, _ = build_device(params=params)
    assert image_settings(device) == {
        "crop_mode": CROP_MODES[0],
        "resize_w": DEFAULT_RESIZE_W,
        "resize_h": DEFAULT_RESIZE_H,
        "jpeg_quality": DEFAULT_JPEG_QUALITY,
    }


def test_unsupported_crop_mode_fails_fast() -> None:
    params = runtime_params(crop_mode="stretch")
    with pytest.raises(ValueError, match="unsupported crop_mode"):
        build_device(params=params)


def test_missing_ros_cfg_fails_fast() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        build_device(params={"crop_mode": CROP_MODES[0]})


@pytest.mark.parametrize("subscribers", [{}, {COLOR_KEY: None}])
def test_camera_without_any_endpoint_fails_fast(
        subscribers: dict) -> None:
    params = {"ros_cfg": {"namespace": "probe",
                          "subscribers": subscribers}}
    with pytest.raises(ValueError, match="neither a subscriber nor a"):
        build_device(params=params)


# ── 2. queue order and capacity ───────────────────────────────────
def test_in_order_arrival_is_kept() -> None:
    device, _, _ = build_device()
    for step in range(1, 4):
        device.on_color_states(colour_message(ts_ns=step * SEC_NS))
    assert queue_stamps(device, COLOR_STREAM) == [
        SEC_NS, 2 * SEC_NS, 3 * SEC_NS]


def test_out_of_order_arrival_is_sorted() -> None:
    device, _, _ = build_device()
    for seconds in (10, 30, 20):
        device.on_color_states(colour_message(ts_ns=seconds * SEC_NS))
    assert queue_stamps(device, COLOR_STREAM) == [
        10 * SEC_NS, 20 * SEC_NS, 30 * SEC_NS]


def test_capacity_drops_the_oldest_frame() -> None:
    device, _, _ = build_device()
    for step in range(QUEUE_CAPACITY + 5):
        device.on_color_states(colour_message(ts_ns=step * SEC_NS))
    stamps = queue_stamps(device, COLOR_STREAM)
    assert len(stamps) == QUEUE_CAPACITY
    assert stamps[0] == 5 * SEC_NS
    assert stamps[-1] == (QUEUE_CAPACITY + 4) * SEC_NS


def test_equal_stamps_keep_the_arrival_order() -> None:
    device, _, _ = build_device()
    first = colour_message(ts_ns=SEC_NS, seed=1)
    second = colour_message(ts_ns=SEC_NS, seed=2)
    device.on_color_states(first)
    device.on_color_states(second)
    stored = device._CameraDevice__queues[COLOR_STREAM]
    assert len(stored) == 2
    assert stored[-1][1] is second


# ── 3. depth stream ───────────────────────────────────────────────
def test_absent_depth_stream_creates_no_queue() -> None:
    device, _, _ = build_device()
    assert queue_names(device) == {COLOR_STREAM}
    device.on_depth_states(depth_message(ts_ns=SEC_NS))
    assert queue_names(device) == {COLOR_STREAM}


def test_absent_depth_stream_ignores_depth_messages() -> None:
    device, _, logger = build_device()
    device.on_color_states(colour_message(ts_ns=SEC_NS))
    device.on_depth_states(depth_message(ts_ns=2 * SEC_NS))
    assert queue_stamps(device, COLOR_STREAM) == [SEC_NS]
    assert logger.lines == []


def test_absent_depth_stream_leaves_the_observation_without_depth() -> None:
    device, _, _ = build_device()
    device.on_color_states(colour_message(ts_ns=SEC_NS))
    observation = device.get_obs()
    assert observation is not None
    assert observation.depth is None


def test_declared_depth_stream_gets_its_own_endpoint_and_queue() -> None:
    device, recorder, _ = build_device(
        params=runtime_params(ros_cfg=DEPTH_ROS_CFG))
    assert device.subscribers == {
        COLOR_KEY: "sub:probe/image_color_raw",
        DEPTH_KEY: "sub:probe/image_depth_raw"}
    assert queue_names(device) == {COLOR_STREAM, DEPTH_STREAM}
    assert [call[2] for call in recorder.calls] == [
        "probe/image_color_raw", "probe/image_depth_raw"]
    assert [call[0] for call in recorder.calls] == [
        "subscription", "subscription"]


def test_depth_arrival_is_sorted_too() -> None:
    device, _, _ = build_device(params=runtime_params(ros_cfg=DEPTH_ROS_CFG))
    for seconds in (30, 10, 20):
        device.on_depth_states(depth_message(ts_ns=seconds * SEC_NS))
    assert queue_stamps(device, DEPTH_STREAM) == [
        10 * SEC_NS, 20 * SEC_NS, 30 * SEC_NS]


def test_depth_frame_is_decoded_as_mono16_bytes() -> None:
    device, _, _ = build_device(
        params=runtime_params(resize_w=100, resize_h=100,
                              ros_cfg=DEPTH_ROS_CFG))
    device.on_depth_states(depth_message(ts_ns=7 * SEC_NS))
    device.on_color_states(colour_message(ts_ns=7 * SEC_NS))
    observation = device.get_obs()
    assert observation.rgb.encoding == JPEG_ENCODING
    assert observation.depth.encoding == DEPTH_ENCODING
    assert (observation.depth.width, observation.depth.height) == (100, 100)
    assert observation.depth.ts_ns == 7 * SEC_NS
    assert observation.depth.data.size == 100 * 100 * 2


# ── 4. geometry ───────────────────────────────────────────────────
def test_cover_crops_the_sides_of_a_wide_image() -> None:
    device, _, _ = build_device(params=runtime_params())
    image = gradient(200, 100, 3)
    cropped = apply_geometry(device, image)
    assert cropped.shape == (100, 100, 3)
    assert np.array_equal(cropped, image[:, 50:150])


def test_cover_crops_both_edges_of_a_tall_image() -> None:
    device, _, _ = build_device(params=runtime_params())
    image = gradient(100, 200, 3)
    cropped = apply_geometry(device, image)
    assert cropped.shape == (100, 100, 3)
    assert np.array_equal(cropped, image[50:150, :])


def test_cover_fills_the_target_without_stretching() -> None:
    device, _, _ = build_device(
        params=runtime_params(crop_mode="cover", resize_w=100, resize_h=40))
    image = gradient(640, 480, 3)
    cropped = apply_geometry(device, image)
    assert cropped.shape == (40, 100, 3)


def test_resize_stretches_both_axes() -> None:
    device, _, _ = build_device(
        params=runtime_params(crop_mode="resize", resize_w=50, resize_h=40))
    image = gradient(200, 100, 3)
    stretched = apply_geometry(device, image)
    assert stretched.shape == (40, 50, 3)


def test_depth_geometry_uses_nearest_neighbour() -> None:
    device, _, _ = build_device(
        params=runtime_params(crop_mode="resize", resize_w=4, resize_h=4))
    image = np.zeros((10, 10), dtype=np.uint16)
    image[::2, :] = 1000
    nearest = apply_geometry(device, image, nearest=True)
    assert nearest.dtype == np.uint16
    assert set(np.unique(nearest)) <= {0, 1000}
    averaged = apply_geometry(device, image, nearest=False)
    assert set(np.unique(averaged)) - {0, 1000}


def test_depth_frame_keeps_nearest_neighbour_end_to_end() -> None:
    device, _, _ = build_device(
        params=runtime_params(crop_mode="resize", resize_w=4, resize_h=4,
                              ros_cfg=DEPTH_ROS_CFG))
    checker = np.zeros(20 * 20, dtype=np.uint16)
    checker.reshape(20, 20)[::2, :] = 1000
    device.on_depth_states(make_image(data=checker, width=20, height=20,
                                      encoding="mono16", ts_ns=SEC_NS))
    device.on_color_states(colour_message(ts_ns=SEC_NS))
    observation = device.get_obs()
    pixels = np.frombuffer(observation.depth.data, dtype=np.uint16)
    assert pixels.size == 4 * 4
    assert set(np.unique(pixels)) <= {0, 1000}


# ── 5. drain semantics ────────────────────────────────────────────
def test_empty_queue_returns_none() -> None:
    device, _, _ = build_device()
    assert device.get_obs() is None
    assert device.get_obs(drain=False) is None
    assert queue_stamps(device, COLOR_STREAM) == []


def test_drain_clears_the_colour_queue() -> None:
    device, _, _ = build_device()
    device.on_color_states(colour_message(ts_ns=SEC_NS))
    assert device.get_obs(drain=True) is not None
    assert queue_stamps(device, COLOR_STREAM) == []
    assert device.get_obs(drain=True) is None


def test_drain_false_keeps_the_colour_queue() -> None:
    device, _, _ = build_device()
    device.on_color_states(colour_message(ts_ns=SEC_NS))
    first = device.get_obs(drain=False)
    second = device.get_obs(drain=False)
    assert first.rgb.ts_ns == second.rgb.ts_ns == SEC_NS
    assert len(queue_stamps(device, COLOR_STREAM)) == 1


def test_drain_clears_the_depth_queue_as_well() -> None:
    device, _, _ = build_device(params=runtime_params(ros_cfg=DEPTH_ROS_CFG))
    device.on_depth_states(depth_message(ts_ns=SEC_NS))
    device.on_color_states(colour_message(ts_ns=SEC_NS))
    observation = device.get_obs(drain=True)
    assert observation.depth is not None
    assert queue_stamps(device, DEPTH_STREAM) == []


# ── 6. decode tolerance ───────────────────────────────────────────
def test_unknown_encoding_is_reported_and_dropped() -> None:
    device, _, logger = build_device()
    device.on_color_states(
        make_image(data=bytearray(64 * 48 * 3), width=64, height=48,
                   encoding="invalid_enc", ts_ns=SEC_NS))
    assert device.get_obs() is None
    assert len(logger.lines) == 1
    assert "[camera.overhead/decode_failed]" in logger.lines[0]
    assert "unsupported encoding" in logger.lines[0]


def test_short_buffer_is_reported_and_dropped() -> None:
    device, _, logger = build_device()
    device.on_color_states(
        make_image(data=bytearray(10), width=100, height=100,
                   encoding="bgr8", ts_ns=SEC_NS))
    assert device.get_obs() is None
    assert len(logger.lines) == 1
    assert "[camera.overhead/decode_failed]" in logger.lines[0]
    assert "buffer holds 10 bytes" in logger.lines[0]


def test_broken_depth_frame_keeps_the_colour_frame_usable() -> None:
    device, _, logger = build_device(
        params=runtime_params(ros_cfg=DEPTH_ROS_CFG))
    device.on_depth_states(
        make_image(data=bytearray(10), width=64, height=48,
                   encoding="mono16", ts_ns=SEC_NS))
    device.on_color_states(colour_message(ts_ns=SEC_NS))
    observation = device.get_obs()
    assert observation is not None
    assert observation.rgb.encoding == JPEG_ENCODING
    assert observation.depth is None
    assert len(logger.lines) == 1
    assert "depth frame dropped" in logger.lines[0]


# ── 7. buffer isolation ───────────────────────────────────────────
def test_colour_payload_does_not_share_the_message_buffer() -> None:
    device, _, _ = build_device()
    message = colour_message()
    device.on_color_states(message)
    observation = device.get_obs()
    assert np.shares_memory(observation.rgb.data, message.data) is False


def test_two_reads_do_not_share_memory() -> None:
    device, _, _ = build_device()
    device.on_color_states(colour_message())
    first = device.get_obs(drain=False)
    second = device.get_obs(drain=False)
    assert np.shares_memory(first.rgb.data, second.rgb.data) is False


def test_writing_to_a_frame_leaves_the_queue_untouched() -> None:
    device, _, _ = build_device()
    message = colour_message()
    original = bytes(message.data)
    device.on_color_states(message)
    first = device.get_obs(drain=False)
    second = device.get_obs(drain=False)
    expected = second.rgb.data.copy()
    first.rgb.data[0] = 255
    assert bytes(message.data) == original
    assert np.array_equal(second.rgb.data, expected)
    assert int(second.rgb.data[0]) == int(expected[0])


def test_depth_payload_does_not_share_the_message_buffer() -> None:
    device, _, _ = build_device(params=runtime_params(ros_cfg=DEPTH_ROS_CFG))
    message = depth_message(ts_ns=SEC_NS)
    device.on_depth_states(message)
    device.on_color_states(colour_message(ts_ns=SEC_NS))
    observation = device.get_obs(drain=True)
    assert np.shares_memory(observation.depth.data, message.data) is False
