################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the recorder."""

from __future__ import annotations

import pathlib
import re

import pytest

from hex_ros_policy_client.utils import (
    KIND_CHUNK,
    KIND_STATE,
    KIND_TARGET,
    KIND_WEIGHT,
    STAGE_LIMITED,
    STAGE_RAW,
    CsvRecorder,
)
from hex_ros_policy_client.utils.common_util import (
    csv_recorder as recorder_module,
)
from hex_ros_policy_client.struct.cmd_struct import (
    ARM_CTRL_MODES,
    GRIP_CTRL_MODES,
    TARGET_ARM,
    TARGET_GRIP,
    ArmCommand,
    GripCommand,
)


def target(values: list) -> dict:
    """Build a target for one arm.

    Args:
        values (list): Joint values.

    Returns:
        dict: Target keyed by command type.
    """
    return {
        TARGET_ARM: {"left": ArmCommand(pos=values)},
        TARGET_GRIP: {"left": GripCommand(pos=[0.5])},
    }


def test_columns_follow_the_arm_count(tmp_path) -> None:
    """A single arm table produces no column of a second arm."""
    single = CsvRecorder(str(tmp_path / "single"), ("main",), 7)
    assert "main_jnt_6" in single.columns[KIND_CHUNK]
    assert "main_grip" in single.columns[KIND_CHUNK]
    assert not any(column.startswith("left") or column.startswith("right")
                   for column in single.columns[KIND_CHUNK])


def test_two_arms_produce_two_column_blocks(tmp_path) -> None:
    """Two arms produce one block of columns each."""
    dual = CsvRecorder(str(tmp_path / "dual"), ("left", "right"), 7)
    columns = dual.columns[KIND_CHUNK]
    assert "left_jnt_0" in columns
    assert "right_jnt_6" in columns
    assert len([name for name in columns if name.endswith("_grip")]) == 2


def test_writes_every_recorder_file(tmp_path) -> None:
    """One record of each kind lands in its own file."""
    recorder = CsvRecorder(str(tmp_path), ("left",), 2)
    recorder.start()
    recorder.record_state(1, "change", "A", "B", "stable_arrived")
    recorder.record_weight(2, 7, 0.5)
    recorder.record_chunk(7, 2, 33, 0, {"left": [0.1, 0.2]},
                          {"left": [0.5]})
    recorder.record_target(3, STAGE_RAW, target([0.1, 0.2]))
    recorder.record_target(3, STAGE_LIMITED, target([0.05, 0.1]))
    recorder.close()
    rows = {KIND_STATE: 1, KIND_WEIGHT: 1, KIND_CHUNK: 1, KIND_TARGET: 2}
    for kind in (KIND_STATE, KIND_WEIGHT, KIND_CHUNK, KIND_TARGET):
        content = (tmp_path / f"{kind}.csv").read_text(encoding="utf-8")
        assert content.count("\n") == rows[kind] + 1


def test_unknown_stage_is_refused(tmp_path) -> None:
    """A target record without a known stage is a programming error."""
    recorder = CsvRecorder(str(tmp_path), ("left",), 2)
    recorder.start()
    with pytest.raises(ValueError):
        recorder.record_target(1, "rawish", target([0.0, 0.0]))
    recorder.close()


def test_close_is_idempotent(tmp_path) -> None:
    """Closing twice is harmless."""
    recorder = CsvRecorder(str(tmp_path), ("left",), 2)
    recorder.start()
    recorder.close()
    recorder.close()


def test_source_has_no_side_literal() -> None:
    """The module never enumerates arm sides."""
    source = pathlib.Path(recorder_module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"for \w+ in \(\s*[\"'](left|right)[\"']", source)
    assert '"left"' not in source
    assert '"right"' not in source
