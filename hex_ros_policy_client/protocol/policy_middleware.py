################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Transport of the three policy links on top of ``hex_util_zenoh``.

Three links connect this package to the model side:

* observation, this package to the model, only while the state is running;
* inference enable and events, both directions on one topic;
* action chunks, model to this package.

The session is not owned here: the class holds one
:class:`hex_util_zenoh.HexZenohNode` and forwards the lifecycle, the endpoints
and the queue helper to it, so the Zenoh side stays the util package
What is left here is which topics exist, what a payload means and how many
chunks are kept. Frame shapes and the byte format live in
``hex_util_zenoh.proto_util``, so a format change never reaches this file.

The chunk queue here only buffers what arrived; the action processor owns the
queue that decides which chunk is being played, and both take their capacity
from the same parameter.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable

from hex_util_zenoh import HexZenohNode

from ..utils import drop
from hex_util_zenoh.proto_util import (
    DualArmActionManipFrame,
    InferenceFrame,
    ObservationFrame,
    check_frame,
    frame_keys,
    pack,
    unpack,
)



# Keys of the enable frame published by this node; the inference topic is
# bidirectional, so its own frames come back and must not be read as events.
ENABLE_KEYS = frame_keys(InferenceFrame)


class PolicyMiddleware:
    """Transport of the policy links on top of a ``HexZenohNode``.

    Every member is private: the links and the session holder are an internal
    arrangement, and the public surface is the four calls below plus the
    lifecycle of the base node.
    """

    def __init__(self,
                 observation_topic: str,
                 action_topic: str,
                 inference_topic: str,
                 capacity: int,
                 on_event: Callable[[dict], None] | None = None,
                 logger: Any | None = None) -> None:
        """Store the link names; the session is started later.

        Args:
            observation_topic (str): Key of the observation link.
            action_topic (str): Key of the action chunk link.
            inference_topic (str): Key of the inference and event link.
            capacity (int): Maximum number of chunks kept while waiting.
            on_event (Callable | None): Handler called with every frame of the
                event link that is not an enable frame; the meaning of a frame
                is decided by the handler, not by the transport.
            logger (Any | None): Logger used for drop lines.

        Raises:
            ValueError: If the capacity is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.__mid_ware = HexZenohNode()
        self.__observation_topic = observation_topic
        self.__action_topic = action_topic
        self.__inference_topic = inference_topic
        self.__on_event = on_event
        self.__logger = logger or logging.getLogger(__name__)
        self.__pending: deque = deque(maxlen=capacity)
        self.__chunks_dropped = 0
        self.__chunks_overflowed = 0

    def start(self) -> None:
        """Start the session and declare every endpoint of the three links."""
        self.__mid_ware.start()
        self.__mid_ware.create_pub(self.__observation_topic)
        self.__mid_ware.create_pub(self.__inference_topic)
        self.__mid_ware.create_sub(self.__action_topic,
                                   self.__on_action_payload)
        self.__mid_ware.create_sub(self.__inference_topic,
                                   self.__on_inference_payload)

    def close(self) -> None:
        """Close the session and release every endpoint."""
        self.__mid_ware.close()

    def pub_obs(self, frame: dict) -> bool:
        """Publish one observation frame.

        The frame is a plain dictionary: building it from the domain objects
        is the job of the observation business, this layer only checks the
        transport fields and packs the bytes.

        Args:
            frame (dict): Frame that matches ``ObservationFrame``.

        Returns:
            bool: Whether the frame reached the transport.

        Raises:
            ValueError: If the frame does not match its declared fields.
        """
        if not self.__mid_ware.is_working:
            return False
        check_frame(frame=frame, frame_type=ObservationFrame)
        self.__mid_ware.pub(self.__observation_topic, pack(frame=frame))
        return True

    def pub_inference_state(self, frame: dict) -> bool:
        """Publish the inference enable frame of this tick.

        Args:
            frame (dict): Frame that matches ``InferenceFrame``.

        Returns:
            bool: Whether the frame reached the transport.

        Raises:
            ValueError: If the frame does not match its declared fields.
        """
        if not self.__mid_ware.is_working:
            return False
        check_frame(frame=frame, frame_type=InferenceFrame)
        self.__mid_ware.pub(self.__inference_topic, pack(frame=frame))
        return True

    def request_chunk(self) -> dict | None:
        """Take the newest decoded action chunk.

        A newer plan supersedes an older one, so the queue is drained and only
        its last entry is returned. The call is passive: chunks are decoded in
        the subscription callback, so the main loop never blocks on the
        network. The empty check is local because ``deque_helper`` only guards
        the oldest-first branch.

        Returns:
            dict | None: Normalised chunk, ``None`` when nothing arrived.
        """
        if not self.__pending:
            return None
        return HexZenohNode.deque_helper(self.__pending, latest=True)

    def __on_action_payload(self, payload: bytes) -> None:
        """Check an action payload and queue the raw frame.

        The queue holds the frame as it arrived, not an interpretation of it:
        turning it into action semantics is the job of the action business,
        which reads it through :meth:`request_chunk`. A frame that does not
        match its declared fields is dropped here, so nothing malformed can
        reach the consumer.

        Args:
            payload (bytes): Payload handed over by the subscription.
        """
        try:
            frame = unpack(payload=payload)
            check_frame(frame=frame, frame_type=DualArmActionManipFrame)
        except ValueError as error:
            self.__chunks_dropped += 1
            drop(
                logger=self.__logger,
                point="frame.schema",
                obj="action frame dropped",
                cause=f"unpack or field check failed: {error}",
                reason="schema_invalid",
                count=self.__chunks_dropped,
            )
            return
        if len(self.__pending) == self.__pending.maxlen:
            self.__chunks_overflowed += 1
            drop(
                logger=self.__logger,
                point="chunk.queue",
                obj="action queue full",
                cause=f"capacity {self.__pending.maxlen} is full, "
                    f"dropping the oldest frame",
                reason="full",
                count=self.__chunks_overflowed,
            )
        self.__pending.append(frame)

    def __on_inference_payload(self, payload: bytes) -> None:
        """Unpack a frame of the event link and hand it to the handler.

        The link is bidirectional, so the enable frames of this node come back
        on it; they are recognised by their declared fields and skipped. What
        the remaining frames mean is decided by the handler, not here.

        Args:
            payload (bytes): Payload handed over by the subscription.
        """
        try:
            frame = unpack(payload=payload)
        except ValueError as error:
            drop(
                logger=self.__logger,
                point="frame.decode",
                obj="event frame dropped",
                cause=f"msgpack unpack failed: {error}",
                reason="decode_failed",
            )
            return
        if set(frame) == ENABLE_KEYS:
            return
        if self.__on_event is not None:
            self.__on_event(frame)
