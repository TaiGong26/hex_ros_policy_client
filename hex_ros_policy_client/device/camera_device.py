################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Camera device bound to ROS topics.

The device subscribes the colour stream and, when the table declares one, the
depth stream. Geometry is applied here so that the layers above receive frames
of exactly the size the model expects.

Colour is encoded as JPEG because the model side wants a compact payload.
Depth is never compressed: a lossy codec would corrupt the measured values. It
still goes through the same geometry, otherwise colour and depth would no
longer be aligned.

Attributes:
    ENCODING_CHANNELS: Bytes per pixel of the encodings we accept.
    CROP_MODES: Supported values of the ``crop_mode`` table key.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..struct.device_struct import CameraObservation, ImageFrame
from .device_base import DeviceBase
from ..utils import drop

ENCODING_CHANNELS = {
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
    "mono8": 1,
    "mono16": 1,
    "8uc1": 1,
    "8uc3": 3,
    "16uc1": 1,
}
# Encodings whose pixels are wider than one byte.
ENCODING_DTYPE = {
    "mono16": np.uint16,
    "16uc1": np.uint16,
}
CROP_MODES = ("cover", "resize")
JPEG_ENCODING = "jpeg"
DEPTH_ENCODING = "mono16"
DEFAULT_QOS_DEPTH = 1
DEFAULT_JPEG_QUALITY = 90


class CameraDevice(DeviceBase):
    """One colour camera with an optional depth stream.

    Attributes:
        TYPE (str): Device table type key.
        REQUIRED_KEYS (tuple): Keys the device table has to provide.
    """

    TYPE = "camera"
    REQUIRED_KEYS = ("type", "color_topic")

    def __init__(
        self,
        node: Any,
        name: str,
        params: dict,
    ) -> None:
        """Bind the subscriptions and read the device parameters.

        Args:
            node: ROS node used to create the endpoints.
            name (str): Device name taken from the device table.
            params (dict): Entry of the ``cameras`` table, already validated.

        Raises:
            ValueError: If ``crop_mode`` is not supported.
        """
        self.__name = name
        # Name the model side uses; the device keeps its own name internally.
        self.model_field = str(params.get("model_field") or name)
        self.__node = node
        self.__crop_mode = str(params.get("crop_mode", CROP_MODES[0]))
        if self.__crop_mode not in CROP_MODES:
            raise ValueError(
                f"camera {name!r}: unsupported crop_mode "
                f"{self.__crop_mode!r}, expected one of {list(CROP_MODES)}")
        self.__resize_w = int(params.get("resize_w", 128))
        self.__resize_h = int(params.get("resize_h", 128))
        self.__jpeg_quality = int(params.get("jpeg_quality",
                                             DEFAULT_JPEG_QUALITY))
        self.__logger = _device_logger(
            node=node,
            name=name,
        )
        self.__depth_message: Any | None = None
        self.__observation: CameraObservation | None = None
        depth = int(params.get("qos_depth", DEFAULT_QOS_DEPTH))
        self.__color_subscription = node.create_subscription(
            _image_type(), params["color_topic"], self.on_states, depth)
        self.__depth_subscription = None
        depth_topic = params.get("depth_topic")
        if depth_topic:
            self.__depth_subscription = node.create_subscription(
                _image_type(), depth_topic, self.__on_depth, depth)

    @property
    def name(self) -> str:
        """Name of this device as declared in the device table.

        Returns:
            str: Device name.
        """
        return self.__name

    def on_states(self, states: Any) -> None:
        """Store a colour frame and rebuild the observation.

        Args:
            states: ``sensor_msgs/Image`` message of the colour stream.
        """
        try:
            image = decode_image(
                data=states.data,
                encoding=states.encoding,
                width=states.width,
                height=states.height,
            )
            prepared = apply_geometry(
                image=image,
                crop_mode=self.__crop_mode,
                resize_w=self.__resize_w,
                resize_h=self.__resize_h,
                nearest=False,
            )
            payload = encode_jpeg(
                image=prepared,
                quality=self.__jpeg_quality,
            )
        except (ValueError, cv2.error) as error:
            drop(
                logger=self.__logger,
                point=f"camera.{self.__name}",
                obj="colour frame dropped",
                cause=f"encode failed: {error}",
                reason="decode_failed",
            )
            return
        stamp = states.header.stamp
        self.__observation = CameraObservation(
            rgb=ImageFrame(
                ts_ns=stamp.sec * 1_000_000_000 + stamp.nanosec,
                encoding=JPEG_ENCODING,
                width=self.__resize_w,
                height=self.__resize_h,
                data=payload,
            ),
            depth=self.__build_depth(),
        )

    def get_obs(self) -> CameraObservation | None:
        """Return the newest observation.

        Returns:
            CameraObservation | None: Latest sample, ``None`` before the first
            colour frame.
        """
        return self.__observation

    def __on_depth(self, message: Any) -> None:
        """Keep the newest depth message.

        Args:
            message: ``sensor_msgs/Image`` message of the depth stream.
        """
        self.__depth_message = message

    def __build_depth(self) -> ImageFrame | None:
        """Convert the newest depth message with the colour geometry.

        Returns:
            ImageFrame | None: Depth frame, ``None`` when the camera has no
            depth stream or no depth arrived yet.
        """
        message = self.__depth_message
        if message is None:
            return None
        try:
            image = decode_image(
                data=message.data,
                encoding=message.encoding,
                width=message.width,
                height=message.height,
            )
            prepared = apply_geometry(
                image=image,
                crop_mode=self.__crop_mode,
                resize_w=self.__resize_w,
                resize_h=self.__resize_h,
                nearest=True,
            )
        except (ValueError, cv2.error) as error:
            drop(
                logger=self.__logger,
                point=f"camera.{self.__name}",
                obj="depth frame dropped",
                cause=f"encode failed: {error}",
                reason="decode_failed",
            )
            return None
        stamp = message.header.stamp
        return ImageFrame(
            ts_ns=stamp.sec * 1_000_000_000 + stamp.nanosec,
            encoding=DEPTH_ENCODING,
            width=self.__resize_w,
            height=self.__resize_h,
            data=np.frombuffer(prepared.tobytes(), dtype=np.uint8),
        )

    def _release(self) -> None:
        """Destroy the endpoints created by this device."""
        handles = (("destroy_subscription", self.__color_subscription),
                   ("destroy_subscription", self.__depth_subscription))
        for destroy, handle in handles:
            method = getattr(self.__node, destroy, None)
            if handle is not None and callable(method):
                method(handle)
        self.__color_subscription = None
        self.__depth_subscription = None


def decode_image(data: Any,
                 encoding: str,
                 width: int,
                 height: int) -> np.ndarray:
    """Turn a raw image payload into an array.

    Args:
        data: Byte buffer of the image message.
        encoding (str): ROS image encoding.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        np.ndarray: Two dimensional array for mono, three dimensional array
        with the original channel order otherwise.

    Raises:
        ValueError: If the encoding is not supported or the size does not
            match the buffer.
    """
    channels = ENCODING_CHANNELS.get(encoding.lower())
    if channels is None:
        raise ValueError(f"unsupported encoding {encoding!r}")
    dtype = ENCODING_DTYPE.get(encoding.lower(), np.uint8)
    raw = bytes(data)
    flat = np.frombuffer(raw, dtype=dtype)
    expected = width * height * channels
    if flat.size != expected:
        raise ValueError(
            f"buffer holds {len(raw)} bytes, expected {expected} values of "
            f"{np.dtype(dtype).name} for {width}x{height}x{channels}")
    if channels == 1:
        return flat.reshape(height, width)
    return flat.reshape(height, width, channels)


def apply_geometry(image: np.ndarray,
                   crop_mode: str,
                   resize_w: int,
                   resize_h: int,
                   nearest: bool = False) -> np.ndarray:
    """Bring an image to the target size using the configured mode.

    ``cover`` scales by the smaller ratio so the target is filled and the
    overflow is cropped away, which keeps the aspect ratio. ``resize`` scales
    each axis independently and therefore stretches the image.

    Args:
        image (np.ndarray): Source image.
        crop_mode (str): ``cover`` or ``resize``.
        resize_w (int): Target width in pixels.
        resize_h (int): Target height in pixels.
        nearest (bool): Use nearest neighbour, required for depth.

    Returns:
        np.ndarray: Image of exactly ``resize_w`` by ``resize_h``.

    Raises:
        ValueError: If the mode is unknown or the target size is not positive.
    """
    if crop_mode not in CROP_MODES:
        raise ValueError(f"unsupported crop_mode {crop_mode!r}")
    if resize_w <= 0 or resize_h <= 0:
        raise ValueError(f"invalid target size {resize_w}x{resize_h}")
    height, width = image.shape[:2]
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    if crop_mode == "resize":
        return cv2.resize(image, (resize_w, resize_h),
                          interpolation=interpolation)
    scale = min(width / resize_w, height / resize_h)
    scaled_w = max(int(round(width / scale)), resize_w)
    scaled_h = max(int(round(height / scale)), resize_h)
    scaled = cv2.resize(image, (scaled_w, scaled_h),
                        interpolation=interpolation)
    left = (scaled_w - resize_w) // 2
    top = (scaled_h - resize_h) // 2
    return scaled[top:top + resize_h, left:left + resize_w]


def encode_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode a colour image as JPEG bytes.

    Args:
        image (np.ndarray): Colour image with one or three channels.
        quality (int): JPEG quality in ``[1, 100]``.

    Returns:
        np.ndarray: Encoded bytes as a one dimensional ``uint8`` array.

    Raises:
        ValueError: If the image cannot be encoded.
    """
    bgr = image
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    ok, buffer = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("jpeg encoding failed")
    return np.frombuffer(buffer.tobytes(), dtype=np.uint8)


def _image_type() -> Any:
    """Return the ROS image message type.

    Imported lazily so the module can be imported in tests that do not install
    the message package.

    Returns:
        Any: ``sensor_msgs/Image`` class.
    """
    from sensor_msgs.msg import Image
    return Image


def _device_logger(node: Any, name: str) -> Any:
    """Pick the logger of the ROS node, falling back to the stdlib one.

    Args:
        node: ROS node or a test double.
        name (str): Device name used in the fallback logger name.

    Returns:
        Any: Logger that exposes ``log(level, message)``.
    """
    getter = getattr(node, "get_logger", None)
    if callable(getter):
        return getter()
    return logging.getLogger(f"{__name__}.{name}")
