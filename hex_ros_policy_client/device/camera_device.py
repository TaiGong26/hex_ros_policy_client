################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-24
################################################################

from __future__ import annotations

import bisect
from collections import deque
from typing import Any, Callable

import cv2
import numpy as np
from sensor_msgs.msg import Image

from ..struct.device_struct import CameraObservation, ImageFrame
from .ros_device_base import RosDeviceBase

CROP_MODES = ("cover", "resize")
JPEG_ENCODING = "jpeg"
DEPTH_ENCODING = "mono16"
_KEY_COLOR = "color_topic"
_KEY_DEPTH = "depth_topic"


class CameraDevice(RosDeviceBase):
    """One colour camera with an optional depth stream.

    The endpoints come from the device table: ``color_topic`` is required and
    ``depth_topic`` may be declared as ``null``, which is how a camera without
    a depth stream says so.

    Attributes:
        SUBSCRIBER_SPECS (dict): Entry key to message type and handler name.
        PUBLISHER_SPECS (dict): Empty, a camera only listens.
    """

    SUBSCRIBER_SPECS = {
        _KEY_COLOR: (Image, "on_color_states"),
        _KEY_DEPTH: (Image, "on_depth_states"),
    }
    
    PUBLISHER_SPECS: dict[str, tuple] = {}

    _ENCODING_CHANNELS = {
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
    _ENCODING_DTYPE = {
        "mono16": np.uint16,
        "16uc1": np.uint16,
    }

    def __init__(
        self, 
        name: str, 
        params: dict,
        create_subscription: Callable,
        create_publisher: Callable,
        logger: Any
    ) -> None:
        """Read the image settings and create the endpoints.

        The settings are read before the endpoints are created, because a
        message may arrive as soon as a subscription exists and the handler
        reads them.

        Args:
            name (str): Device name taken from the device table.
            params (dict): Entry of the ``cameras`` table.
            create_subscription (Callable): ``(msg_type, topic, callback)``
                returning the handle of the new subscription.
            create_publisher (Callable): ``(msg_type, topic)`` returning the
                handle of the new publisher, unused by a camera.
            logger (Any): Logger object exposing standard logging methods.

        Raises:
            ValueError: If ``crop_mode`` is not supported, or if the entry
                itself is rejected by the base class.
        """
        crop_mode = str(params.get("crop_mode", CROP_MODES[0]))
        if crop_mode not in CROP_MODES:
            raise ValueError(
                f"camera {name!r}: unsupported crop_mode "
                f"{crop_mode!r}, expected one of {list(CROP_MODES)}")
        self.__params: dict[str, Any] = {
            "crop_mode": crop_mode,
            "resize_w": int(params.get("resize_w", 128)),
            "resize_h": int(params.get("resize_h", 128)),
            "jpeg_quality": int(params.get("jpeg_quality", 90)),
        }
        
        self.__queues: dict[str, deque[tuple[int, Image]]] = {}
        
        super().__init__(
            name=name,
            params=params,
            create_subscription=create_subscription,
            create_publisher=create_publisher,
            logger=logger,
        )
        
        self.__queues["color"] = deque(maxlen=30)
        if _KEY_DEPTH in self.subscribers:
            self.__queues["depth"] = deque(maxlen=30)

    def on_color_states(self, states: Image) -> None:
        """Enqueue one raw colour image maintaining monotonic stamp ordering.

        Args:
            states (Image): ROS Image message of the colour stream.
        """
        self.__enqueue(stream="color", message=states)

    def on_depth_states(self, states: Image) -> None:
        """Enqueue one raw depth image maintaining monotonic stamp ordering.

        Args:
            states (Image): ROS Image message of the depth stream.
        """
        if _KEY_DEPTH not in self.subscribers:
            return
        self.__enqueue(stream="depth", message=states)

    def get_obs(self, drain: bool = True) -> CameraObservation | None:
        """Decode and package the latest observation from the queues.

        Args:
            drain (bool): When True, retrieves the latest element and clears
                the queue; otherwise retains queue contents.

        Returns:
            CameraObservation | None: Latest observation constructed from the
            queue, or None if the color queue is empty or decode fails.
        """
        color_queue = self.__queues["color"]
        try:
            if not color_queue:
                return None
            _, color_msg = color_queue[-1]
            if drain:
                color_queue.clear()
        except IndexError:
            return None
        rgb_frame = self.__process_color(message=color_msg)
        if rgb_frame is None:
            return None
        
        depth_frame: ImageFrame | None = None
        if _KEY_DEPTH in self.subscribers:
            depth_msg: Image | None = None
            depth_queue = self.__queues.get("depth")
            if depth_queue:
                try:
                    _, depth_msg = depth_queue[-1]
                    if drain:
                        depth_queue.clear()
                except IndexError:
                    depth_msg = None
            if depth_msg is not None:
                depth_frame = self.__process_depth(message=depth_msg)
        return CameraObservation(rgb=rgb_frame, depth=depth_frame)

    def __enqueue(self, stream: str, message: Image) -> None:
        """Insert an image into the queue ordered monotonically by stamp.

        Args:
            stream (str): Stream identifier, ``color`` or ``depth``.
            message (Image): Message to store.
        """
        queue = self.__queues[stream]
        ts_ns = _stamp_ns(stamp=message.header.stamp)
        item = (ts_ns, message)
        if not queue or ts_ns >= queue[-1][0]:
            queue.append(item)
            return
        timestamps = [stored[0] for stored in queue]
        index = bisect.bisect_right(timestamps, ts_ns)
        queue.insert(index, item)

    def __process_color(self, message: Image) -> ImageFrame | None:
        """Decode and compress one color image message.

        Args:
            message (Image): Source color image message.

        Returns:
            ImageFrame | None: Processed image frame, or None on failure.
        """
        try:
            image = self.__decode_image(
                data=message.data,
                encoding=message.encoding,
                width=message.width,
                height=message.height,
            )
            prepared = self.__apply_geometry(image=image, nearest=False)
            payload = self.__encode_jpeg(
                image=prepared,
                quality=self.__params["jpeg_quality"],
            )
        except (ValueError, cv2.error) as error:
            self._logger.logw(
                f"[camera.{self._name}/decode_failed] colour frame "
                f"dropped: {error}")
            return None
        return ImageFrame(
            ts_ns=_stamp_ns(stamp=message.header.stamp),
            encoding=JPEG_ENCODING,
            width=prepared.shape[1],
            height=prepared.shape[0],
            data=payload.copy(),
        )

    def __process_depth(self, message: Image) -> ImageFrame | None:
        """Decode and crop one depth image message.

        Args:
            message (Image): Source depth image message.

        Returns:
            ImageFrame | None: Processed depth frame, or None on failure.
        """
        try:
            image = self.__decode_image(
                data=message.data,
                encoding=message.encoding,
                width=message.width,
                height=message.height,
            )
            prepared = self.__apply_geometry(image=image, nearest=True)
        except (ValueError, cv2.error) as error:
            self._logger.logw(
                f"[camera.{self._name}/decode_failed] depth frame "
                f"dropped: {error}")
            return None
        depth_bytes = np.frombuffer(prepared.tobytes(), dtype=np.uint8)
        return ImageFrame(
            ts_ns=_stamp_ns(stamp=message.header.stamp),
            encoding=DEPTH_ENCODING,
            width=prepared.shape[1],
            height=prepared.shape[0],
            data=depth_bytes.copy(),
        )

    def __apply_geometry(
        self, 
        image: np.ndarray,
        nearest: bool
    ) -> np.ndarray:
        """Apply crop or resize according to configured crop_mode.

        Args:
            image (np.ndarray): Source image array.
            nearest (bool): Whether to force nearest neighbour interpolation.

        Returns:
            np.ndarray: Image resized or cropped to target size.
        """
        mode = self.__params["crop_mode"]
        resize_w = self.__params["resize_w"]
        resize_h = self.__params["resize_h"]
        height, width = image.shape[:2]
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        
        if mode == "resize":
            return cv2.resize(
                image, 
                (resize_w, resize_h),
                interpolation=interpolation
            )
        
        scale = min(width / resize_w, height / resize_h)
        scaled_w = max(int(round(width / scale)), resize_w)
        scaled_h = max(int(round(height / scale)), resize_h)
        scaled = cv2.resize(image, (scaled_w, scaled_h),
                            interpolation=interpolation)
        left = (scaled_w - resize_w) // 2
        top = (scaled_h - resize_h) // 2
        return scaled[top:top + resize_h, left:left + resize_w]

    @staticmethod
    def __encode_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
        """Encode a colour image as JPEG bytes.

        Args:
            image (np.ndarray): Colour image array.
            quality (int): JPEG quality parameter.

        Returns:
            np.ndarray: Encoded uint8 byte array.

        Raises:
            ValueError: If JPEG encoding fails.
        """
        bgr = image
        if image.ndim == 3 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        ok, buffer = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise ValueError("jpeg encoding failed")
        return np.frombuffer(buffer.tobytes(), dtype=np.uint8)

    @staticmethod
    def __decode_image(data: Any,
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
        channels = CameraDevice._ENCODING_CHANNELS.get(encoding.lower())
        if channels is None:
            raise ValueError(f"unsupported encoding {encoding!r}")
        dtype = CameraDevice._ENCODING_DTYPE.get(encoding.lower(), np.uint8)
        raw = bytes(data)
        flat = np.frombuffer(raw, dtype=dtype)
        expected = width * height * channels
        if flat.size != expected:
            raise ValueError(
                f"buffer holds {len(raw)} bytes, expected {expected} values "
                f"of {np.dtype(dtype).name} for {width}x{height}x{channels}")
        if channels == 1:
            return flat.reshape(height, width)
        return flat.reshape(height, width, channels)

def _stamp_ns(stamp: Any) -> int:
    """Turn a ROS stamp into nanoseconds.

    Args:
        stamp: ``builtin_interfaces/Time`` of one message header.

    Returns:
        int: Time in nanoseconds.
    """
    return stamp.sec * 1_000_000_000 + stamp.nanosec
