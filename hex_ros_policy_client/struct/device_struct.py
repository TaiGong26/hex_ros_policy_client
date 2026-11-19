################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Device level data structures.

Data only: the structures describe what a device produced, they never convert
or validate physical values.

Required and optional fields are expressed by the type itself. A field without
a default and without ``| None`` is required; a field typed ``X | None`` with a
``None`` default is optional. No comment marks them again, so the two places
cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from .geom_struct import Pose


@dataclass(frozen=True)
class ArmState:
    """State of one manipulator arm.

    Attributes:
        arm_pos (np.ndarray): Joint positions in radians.
        arm_vel (np.ndarray): Joint velocities in radians per second.
        arm_eff (np.ndarray): Joint efforts.
        ee_base (Pose): End effector pose in the arm base frame.
    """

    arm_pos: np.ndarray
    arm_vel: np.ndarray
    arm_eff: np.ndarray
    ee_base: Pose


@dataclass(frozen=True)
class GripState:
    """State of one gripper.

    The raw values are kept as they arrive; the mapping to the model range
    belongs to the device layer.

    Attributes:
        grip_pos (np.ndarray): Gripper position.
        grip_vel (np.ndarray): Gripper velocity.
        grip_eff (np.ndarray): Gripper effort.
    """

    grip_pos: np.ndarray
    grip_vel: np.ndarray
    grip_eff: np.ndarray


@dataclass(frozen=True)
class ArmObservation:
    """Latest state of one arm and its gripper.

    Arm and gripper come from the same ``manip_state`` message, so they share
    the single outer header stamp instead of keeping two clocks.

    Attributes:
        stamp_ns (int | None): Stamp of the source message, ``None`` when the
            device never received anything yet.
        arm_state (ArmState | None): Arm part of the message.
        grip_state (GripState | None): Gripper part of the message.
    """

    stamp_ns: int | None = None
    arm_state: ArmState | None = None
    grip_state: GripState | None = None


@dataclass(frozen=True)
class ImageFrame:
    """One image with its encoding and geometry.

    Attributes:
        ts_ns (int): Capture time taken from the image header stamp.
        encoding (str): Pixel format such as ``rgb8`` or ``mono16``, or
            ``jpeg`` for already encoded bytes.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        data (np.ndarray): Pixel buffer of type ``uint8``; for ``jpeg`` it is
            the encoded byte stream.
    """

    ts_ns: int
    encoding: str
    width: int
    height: int
    data: np.ndarray


@dataclass(frozen=True)
class CameraObservation:
    """One frame of one camera.

    Attributes:
        rgb (ImageFrame): Colour frame, always present.
        depth (ImageFrame | None): Depth frame, ``None`` when the camera is
            declared without a depth topic.
    """

    rgb: ImageFrame
    depth: ImageFrame | None = None


class ObservationFrame(TypedDict):
    """One assembled observation, the input of the wire frame.

    The name repeats the frame of the protocol on purpose: both carry the same
    three keys. The two device maps hold the samples of this package instead of
    the wire payloads, and the stamp stays integer nanoseconds instead of the
    ``secs``/``nsecs`` split. Time inside this package is always integer
    nanoseconds, so the split is made once, by ``build_observation_frame``, on
    the boundary.

    Attributes:
        stamp_ns (int): Frame stamp produced by the observation layer.
        camera (dict): Camera samples keyed by device name.
        arm (dict): Arm samples keyed by device name.
    """

    stamp_ns: int
    camera: dict[str, CameraObservation]
    arm: dict[str, ArmObservation]
