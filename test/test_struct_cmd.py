################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the command structures."""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.struct.cmd_struct import (
    ARM_CTRL_MODES,
    GRIP_CTRL_MODES,
    ArmCommand,
    GripCommand,
)
from hex_ros_policy_client.struct.geom_struct import Pose


@pytest.mark.parametrize("mode", ARM_CTRL_MODES)
def test_arm_command_accepts_every_mode_in_the_list(mode: str) -> None:
    """Every mode of the list builds and keeps the mode it was given."""
    command = ArmCommand(ctrl_mode=mode, pos=[0.0] * 7)
    assert command.ctrl_mode == mode


@pytest.mark.parametrize("mode", GRIP_CTRL_MODES)
def test_grip_command_accepts_every_mode_in_the_list(mode: str) -> None:
    """Every mode of the list builds and keeps the mode it was given."""
    command = GripCommand(ctrl_mode=mode, pos=[0.0])
    assert command.ctrl_mode == mode


def test_default_mode_is_the_first_of_the_list() -> None:
    """A command built without a mode takes the first entry of its list."""
    assert ArmCommand().ctrl_mode == ARM_CTRL_MODES[0]
    assert GripCommand().ctrl_mode == GRIP_CTRL_MODES[0]


@pytest.mark.parametrize("builder", [
    lambda: ArmCommand(ctrl_mode="unknown", pos=[0.0]),
    lambda: GripCommand(ctrl_mode="unknown", pos=[0.0]),
])
def test_unknown_mode_rejected(builder) -> None:
    """A mode outside the list is a construction error."""
    with pytest.raises(ValueError):
        builder()


def test_extra_field_is_tolerated() -> None:
    """A field not selected by the mode is ignored, not rejected."""
    command = ArmCommand(pos=np.zeros(7), pose=Pose())
    assert command.pose is not None


def test_command_has_no_timestamp() -> None:
    """Commands carry no time: dispatch means execute this tick."""
    command = ArmCommand(pos=[0.0])
    assert not hasattr(command, "ts_ns")
