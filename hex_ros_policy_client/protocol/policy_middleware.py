################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional

from hex_util_zenoh import HexZenohNode
from hex_util_zenoh.proto_util import (
    DualArmActionManipFrame,
    InferenceFrame,
    ObservationFrame,
    check_frame,
    frame_keys,
    pack,
    unpack,
)


# Keys of the enable frame published by this node.
ENABLE_KEYS = frame_keys(InferenceFrame)


class PolicyMiddleware:
    """Transport of the three policy links over one session.
    """

    def __init__(self,
                 observation_topic: str,
                 action_topic: str,
                 inference_topic: str,
                 capacity: int,
                 logger: Any,) -> None:
        """Store the link names and the queue capacity.

        Args:
            observation_topic (str): Key of the observation link.
            action_topic (str): Key of the action chunk link.
            inference_topic (str): Key of the inference enable link.
            capacity (int): Maximum number of chunks kept while waiting,
                positive.
            logger (Any): Logger that provides ``logw`` and ``logd``.

        Raises:
            ValueError: If ``capacity`` is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.__mid_ware = HexZenohNode()
        self.__observation_topic = observation_topic
        self.__action_topic = action_topic
        self.__inference_topic = inference_topic
        self.__logger = logger
        self.__res_chunck: deque[Optional[dict]] = deque(maxlen=capacity)

    def start(self) -> None:
        """Start the session and declare the endpoints of the three links.
        """
        self.__mid_ware.start()
        self.__mid_ware.create_pub(self.__observation_topic)
        self.__mid_ware.create_pub(self.__inference_topic)
        self.__mid_ware.create_sub(self.__action_topic,
                                   self.__on_action_payload)

    def close(self) -> None:
        """Close the session and release the endpoints.
        """
        self.__mid_ware.close()

    def pub_obs(self, frame: dict) -> bool:
        """Publish one observation frame.

        Args:
            frame (dict): Frame with the keys of ``ObservationFrame``.

        Returns:
            bool: ``True`` when the frame was handed to the session, ``False``
            when the session is not running.

        Raises:
            ValueError: If ``frame`` does not match its declared keys.
        """
        if not self.__mid_ware.is_working:
            return False
        check_frame(frame=frame, frame_type=ObservationFrame)
        self.__mid_ware.pub(self.__observation_topic, pack(frame=frame))
        return True

    def pub_inference_state(self, frame: dict) -> bool:
        """Publish the inference enable frame of this tick.

        Args:
            frame (dict): Frame with the keys of ``InferenceFrame``.

        Returns:
            bool: ``True`` when the frame was handed to the session, ``False``
            when the session is not running.

        Raises:
            ValueError: If ``frame`` does not match its declared keys.
        """
        if not self.__mid_ware.is_working:
            return False
        check_frame(frame=frame, frame_type=InferenceFrame)
        self.__mid_ware.pub(self.__inference_topic, pack(frame=frame))
        return True

    def request_chunk(self) -> dict | None:
        """Return the newest queued action chunk, emptying the queue.

        Returns:
            dict | None: Frame with the keys of ``DualArmActionManipFrame``,
            ``None`` when the queue is empty.
        """
        if not self.__res_chunck:
            return None
        return HexZenohNode.deque_helper(self.__res_chunck, latest=True)

    def __on_action_payload(self, payload: bytes) -> None:
        """Check one action payload and append it, dropping a bad one.

        Args:
            payload (bytes): Payload handed over by the subscription.
        """
        try:
            frame = unpack(payload=payload)
            check_frame(frame=frame, frame_type=DualArmActionManipFrame)
        except ValueError as error:
            self.__logger.logw(
                "[frame.schema/schema_invalid] action frame dropped: "
                f"unpack or field check failed: {error}")
            return
        if len(self.__res_chunck) == self.__res_chunck.maxlen:
            self.__logger.logw(
                "[chunk.queue/full] action queue full: "
                f"capacity {self.__res_chunck.maxlen} is full, "
                "dropping the oldest frame")
        self.__res_chunck.append(frame)
