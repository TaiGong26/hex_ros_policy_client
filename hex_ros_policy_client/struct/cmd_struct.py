################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Command structures shared by every layer.

The structures live in the data layer because the device layer needs the very
same types to send them. Placing them in ``core`` or ``utils`` would either
invert the dependency direction or mix data with device implementations.

The outer keys of a target are the command types; the inner keys are device
names, that is the keys of the ``arms`` table in the device configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from .geom_struct import Pose

# Control modes of the two commands, one list each. The list is the only copy
# of the strings: being in the list is the whole validation, and the first
# entry is the mode this package picks when no mode is given. The strings are
# the ones the policy wire carries, so a chunk is used without renaming
# anything; the translation to the ROS enumerations happens in the device
# layer, the only place that knows the message.
ARM_CTRL_MODES = ("pos", "ee")
GRIP_CTRL_MODES = ("pos", "eff")

TARGET_ARM = "arm"
TARGET_GRIP = "grip"

# Modes a frame may carry.
ACTION_MODES = (
    "delta",
    "joint",
    "joint_diff",
)


@dataclass(frozen=True)
class ArmCommand:
    """Control command for one arm at one instant.

    Attributes:
        ctrl_mode (str): One of ``ARM_CTRL_MODES``; ``pos`` by default.
        pos (np.ndarray | list[float] | None): Joint target, used by ``pos``.
        pose (Pose | None): End effector target, used by ``ee``.

    Raises:
        ValueError: If ``ctrl_mode`` is not in ``ARM_CTRL_MODES``.
    """

    ctrl_mode: str = ARM_CTRL_MODES[0]
    pos: np.ndarray | list[float] | None = None
    pose: Pose | None = None

    def __post_init__(self) -> None:
        """Reject a mode this command does not know.

        Raises:
            ValueError: If the mode is not in ``ARM_CTRL_MODES``.
        """
        if self.ctrl_mode not in ARM_CTRL_MODES:
            raise ValueError(f"unknown arm ctrl_mode: {self.ctrl_mode!r}")


@dataclass(frozen=True)
class GripCommand:
    """Control command for one gripper at one instant.

    Attributes:
        ctrl_mode (str): One of ``GRIP_CTRL_MODES``; ``pos`` by default.
        pos (np.ndarray | list[float] | None): Gripper target, used by ``pos``.
        eff (np.ndarray | list[float] | None): Feed forward effort, used by
            ``eff``.

    Raises:
        ValueError: If ``ctrl_mode`` is not in ``GRIP_CTRL_MODES``.
    """

    ctrl_mode: str = GRIP_CTRL_MODES[0]
    pos: np.ndarray | list[float] | None = None
    eff: np.ndarray | list[float] | None = None

    def __post_init__(self) -> None:
        """Reject a mode this command does not know.

        Raises:
            ValueError: If the mode is not in ``GRIP_CTRL_MODES``.
        """
        if self.ctrl_mode not in GRIP_CTRL_MODES:
            raise ValueError(f"unknown grip ctrl_mode: {self.ctrl_mode!r}")


# Outer key: command type. Inner key: device name.
Target = dict[str, dict[str, Union[ArmCommand, GripCommand]]]
