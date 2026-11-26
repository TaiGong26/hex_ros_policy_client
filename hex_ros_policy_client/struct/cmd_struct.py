################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom_struct import Pose

# Control modes of the two commands, one list each. The list is the only copy
# of the strings: being in the list is the whole validation, and the first
# entry is the mode this package picks when no mode is given. The strings are
# the ones the policy wire carries, so a chunk is used without renaming
# anything; the translation to the ROS enumerations happens in the device
# layer, the only place that knows the message.
ARM_CTRL_MODES = ("pos", "ee", "mit")
GRIP_CTRL_MODES = ("pos", "eff")


@dataclass(frozen=True)
class ArmCommand:
    """Control command for one arm at one instant.

    Attributes:
        ctrl_mode (str): One of ``ARM_CTRL_MODES``.
        pos (np.ndarray | list[float] | None): Joint target in radians, used
            when ``ctrl_mode`` is ``pos`` or ``mit``.
        pose (Pose | None): End effector target, used when ``ctrl_mode`` is
            ``ee``.
        vel (np.ndarray | list[float] | None): Joint velocity target, used
            when ``ctrl_mode`` is ``mit``.
        eff (np.ndarray | list[float] | None): Joint feedforward effort, used
            when ``ctrl_mode`` is ``mit``.
        kp (np.ndarray | list[float] | None): Joint stiffness gains, used
            when ``ctrl_mode`` is ``mit``; falls back to param if None.
        kd (np.ndarray | list[float] | None): Joint damping gains, used
            when ``ctrl_mode`` is ``mit``; falls back to param if None.

    Raises:
        ValueError: If ``ctrl_mode`` is not in ``ARM_CTRL_MODES``.
    """

    ctrl_mode: str = ARM_CTRL_MODES[0]
    pos: np.ndarray | list[float] | None = None
    pose: Pose | None = None
    vel: np.ndarray | list[float] | None = None
    eff: np.ndarray | list[float] | None = None
    kp: np.ndarray | list[float] | None = None
    kd: np.ndarray | list[float] | None = None

    def __post_init__(self) -> None:
        """Reject a control mode this command does not know.

        Raises:
            ValueError: If ``ctrl_mode`` is not in ``ARM_CTRL_MODES``.
        """
        if self.ctrl_mode not in ARM_CTRL_MODES:
            raise ValueError(f"unknown arm ctrl_mode: {self.ctrl_mode!r}")


@dataclass(frozen=True)
class GripCommand:
    """Control command for one gripper at one instant.

    Attributes:
        ctrl_mode (str): One of ``GRIP_CTRL_MODES``.
        pos (np.ndarray | list[float] | None): Gripper target, used when
            ``ctrl_mode`` is ``pos``.
        eff (np.ndarray | list[float] | None): Feed forward effort, used when
            ``ctrl_mode`` is ``eff``.

    Raises:
        ValueError: If ``ctrl_mode`` is not in ``GRIP_CTRL_MODES``.
    """

    ctrl_mode: str = GRIP_CTRL_MODES[0]
    pos: np.ndarray | list[float] | None = None
    eff: np.ndarray | list[float] | None = None

    def __post_init__(self) -> None:
        """Reject a control mode this command does not know.

        Raises:
            ValueError: If ``ctrl_mode`` is not in ``GRIP_CTRL_MODES``.
        """
        if self.ctrl_mode not in GRIP_CTRL_MODES:
            raise ValueError(f"unknown grip ctrl_mode: {self.ctrl_mode!r}")


@dataclass
class ControlIntent:
    """Commands of one tick, grouped by command type.

    Attributes:
        arm (dict[str, ArmCommand]): Arm commands keyed by device name.
        grip (dict[str, GripCommand]): Gripper commands keyed by device name.
        stamp_ns (int | None): Target time in nanoseconds for this intent.
    """

    arm: dict[str, ArmCommand] = field(default_factory=dict)
    grip: dict[str, GripCommand] = field(default_factory=dict)
    stamp_ns: int | None = None
