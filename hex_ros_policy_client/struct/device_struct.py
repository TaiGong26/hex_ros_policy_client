################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from .geom_struct import Pose


@dataclass(frozen=True)
class ArmState:
    """State of one manipulator arm.

    Attributes:
        arm_pos (np.ndarray): Joint positions in radians, one entry per joint.
        arm_vel (np.ndarray): Joint velocities in radians per second, same
            length as ``arm_pos``.
        arm_eff (np.ndarray): Joint efforts, same length as ``arm_pos``.
        ee_base (Pose): End effector pose in the arm base frame.
    """

    arm_pos: np.ndarray
    arm_vel: np.ndarray
    arm_eff: np.ndarray
    ee_base: Pose


@dataclass(frozen=True)
class GripState:
    """State of one gripper.

    Attributes:
        grip_pos (np.ndarray): Gripper position as reported by the device.
        grip_vel (np.ndarray): Gripper velocity as reported by the device.
        grip_eff (np.ndarray): Gripper effort as reported by the device.
    """

    grip_pos: np.ndarray
    grip_vel: np.ndarray
    grip_eff: np.ndarray


@dataclass(frozen=True)
class ArmObservation:
    """Latest state of one arm and its gripper.

    Attributes:
        stamp_ns (int | None): Stamp of the source message in nanoseconds,
            ``None`` when nothing has arrived yet.
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
        ts_ns (int): Capture time in nanoseconds.
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
        rgb (ImageFrame): Colour frame.
        depth (ImageFrame | None): Depth frame, ``None`` when the camera has no
            depth stream.
    """

    rgb: ImageFrame
    depth: ImageFrame | None = None


class DeviceObservationFrame(TypedDict):
    """One assembled observation, with integer nanoseconds as its stamp.

    Attributes:
        stamp_ns (int): Stamp of the frame in nanoseconds.
        camera (dict[str, CameraObservation]): Camera samples keyed by device
            name.
        arm (dict[str, ArmObservation]): Arm samples keyed by device name.
    """

    stamp_ns: int
    camera: dict[str, CameraObservation]
    arm: dict[str, ArmObservation]
