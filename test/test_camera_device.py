################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the camera device."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hex_ros_policy_client.device.camera_device import (
    CROP_MODES,
    CameraDevice,
    apply_geometry,
    decode_image,
    encode_jpeg,
)

from .fakes import FakeNode

pytest.importorskip("sensor_msgs.msg")

CAMERA_PARAMS = {
    "type": "camera",
    "color_topic": "fixed/image_color_raw",
    "depth_topic": "fixed/image_depth_raw",
    "crop_mode": "cover",
    "resize_w": 128,
    "resize_h": 128,
    "jpeg_quality": 90,
}


def gradient(width: int, height: int, channels: int = 3) -> np.ndarray:
    """Build a deterministic test image.

    Args:
        width (int): Image width.
        height (int): Image height.
        channels (int): Channel count, one for mono images.

    Returns:
        np.ndarray: Image with a horizontal gradient.
    """
    row = np.linspace(0, 255, width, dtype=np.uint8)
    plane = np.tile(row, (height, 1))
    if channels == 1:
        return plane
    return np.dstack([plane] * channels)


def image_message(image: np.ndarray, encoding: str) -> SimpleNamespace:
    """Wrap an array into an object shaped like ``sensor_msgs/Image``.

    Args:
        image (np.ndarray): Image data.
        encoding (str): ROS encoding string.

    Returns:
        SimpleNamespace: Message stand-in.
    """
    height, width = image.shape[:2]
    return SimpleNamespace(
        data=image.tobytes(),
        encoding=encoding,
        width=width,
        height=height,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=7, nanosec=11)),
    )


def build_device(params: dict | None = None) -> tuple[CameraDevice, FakeNode]:
    """Create a camera device bound to a fake node.

    Args:
        params (dict | None): Overrides of the default device table entry.

    Returns:
        tuple: Device and the node it uses.
    """
    node = FakeNode()
    merged = dict(CAMERA_PARAMS)
    merged.update(params or {})
    return CameraDevice(node, "fixed", merged), node


def test_cover_keeps_aspect_ratio() -> None:
    """Cover fills the target and crops the overflow."""
    result = apply_geometry(gradient(640, 480), "cover", 128, 128)
    assert result.shape[:2] == (128, 128)


def test_cover_crops_the_sides_of_a_wide_image() -> None:
    """A wide image loses its side borders, not its top and bottom."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, 320:, :] = 255
    result = apply_geometry(image, "cover", 128, 128)
    middle = result[64]
    assert middle[0, 0] == 0
    assert middle[-1, 0] == 255


def test_resize_stretches_both_axes() -> None:
    """Resize maps the source rectangle onto the target rectangle."""
    result = apply_geometry(gradient(640, 480), "resize", 128, 128)
    assert result.shape[:2] == (128, 128)


def test_unknown_mode_rejected() -> None:
    """An unsupported mode is a configuration error."""
    with pytest.raises(ValueError):
        apply_geometry(gradient(64, 64), "squash", 32, 32)


def test_decode_rejects_unknown_encoding() -> None:
    """An unsupported encoding is reported, not silently misread."""
    with pytest.raises(ValueError):
        decode_image(b"\x00" * 16, "yuv999", 2, 2)


def test_decode_rejects_size_mismatch() -> None:
    """A buffer that does not match the declared size is rejected."""
    with pytest.raises(ValueError):
        decode_image(b"\x00" * 8, "rgb8", 2, 2)


def test_jpeg_encoding_is_lossy_but_smaller() -> None:
    """The colour frame leaves the device as a JPEG byte stream."""
    encoded = encode_jpeg(gradient(128, 128), 90)
    assert encoded.dtype == np.uint8
    assert encoded.size > 0
    imported = bytes(encoded)
    assert imported[:2] == b"\xff\xd8"


def test_colour_frame_is_encoded() -> None:
    """A colour message produces a JPEG frame of the target size."""
    device, node = build_device()
    node.subscription("fixed/image_color_raw").callback(
        image_message(gradient(640, 480), "bgr8"))
    observation = device.get_obs()
    assert observation.rgb.encoding == "jpeg"
    assert (observation.rgb.width, observation.rgb.height) == (128, 128)
    assert observation.rgb.ts_ns == 7 * 1_000_000_000 + 11


def test_depth_is_optional() -> None:
    """Without a depth message the depth frame stays empty."""
    device, node = build_device()
    node.subscription("fixed/image_color_raw").callback(
        image_message(gradient(640, 480), "bgr8"))
    assert device.get_obs().depth is None


def test_depth_is_not_compressed() -> None:
    """Depth keeps its values and only changes geometry."""
    params = {"resize_w": 32, "resize_h": 32, "crop_mode": "resize"}
    device, node = build_device(params)
    depth = np.arange(640 * 480, dtype=np.uint16).reshape(480, 640)
    node.subscription("fixed/image_depth_raw").callback(
        image_message(depth, "mono16"))
    node.subscription("fixed/image_color_raw").callback(
        image_message(gradient(640, 480), "bgr8"))
    frame = device.get_obs().depth
    assert frame.encoding == "mono16"
    restored = np.frombuffer(frame.data.tobytes(), dtype=np.uint16)
    assert restored.size == 32 * 32
    assert set(np.unique(restored)) <= set(np.unique(depth))


def test_bad_crop_mode_fails_fast() -> None:
    """An invalid table value stops the start-up instead of defaulting."""
    with pytest.raises(ValueError):
        build_device({"crop_mode": "squash"})


def test_crop_modes_table() -> None:
    """The supported modes are exactly cover and resize."""
    assert CROP_MODES == ("cover", "resize")
