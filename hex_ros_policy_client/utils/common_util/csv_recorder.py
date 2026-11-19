################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Recorder for the data a rosbag cannot capture.

A bag already records the observation frames and the raw ``manip_state``
stream, so repeating them here would only double the volume. What is kept is
the data that exists only inside this process: the decoded action chunks, the
weights of the time integration, the target before and after the step limit,
and the state transitions with their internal events.

Columns and files are generated from the device table. No side name appears in
the code, so the same recorder serves one arm and several.

Attributes:
    KIND_STATE: Recorder file that holds migrations and events.
    KIND_CHUNK: Recorder file that holds decoded chunks.
    KIND_WEIGHT: Recorder file that holds integration weights.
    KIND_TARGET: Recorder file that holds targets before and after limiting.
    STAGES: Allowed values of the target stage column.
"""

from __future__ import annotations

import csv
from pathlib import Path

KIND_STATE = "states"
KIND_CHUNK = "chunks"
KIND_WEIGHT = "weights"
KIND_TARGET = "targets"
RECORDER_KINDS = (KIND_STATE, KIND_CHUNK, KIND_WEIGHT, KIND_TARGET)

STAGE_RAW = "raw"
STAGE_LIMITED = "limited"
STAGES = (STAGE_RAW, STAGE_LIMITED)

EVENT_COLUMNS = ("now_ns", "kind", "old", "new", "detail")
WEIGHT_COLUMNS = ("now_ns", "seq", "weight")


class CsvRecorder:
    """Writer for the four recorder files.

    Attributes:
        directory (Path): Directory the files are written to.
        columns (dict): Recorder kind to the column tuple of that file.
    """

    def __init__(self, directory: str, arm_names: tuple[str, ...],
                 arm_dof: int) -> None:
        """Prepare the column sets of this topology.

        Args:
            directory (str): Directory the files are written to.
            arm_names (tuple): Arm names from the device hub.
            arm_dof (int): Degrees of freedom of one arm.

        Raises:
            ValueError: If the degree of freedom is not positive.
        """
        if arm_dof <= 0:
            raise ValueError(f"arm_dof must be positive, got {arm_dof}")
        self.directory = Path(directory)
        self.__arm_names = tuple(arm_names)
        self.columns = {
            KIND_STATE: EVENT_COLUMNS,
            KIND_WEIGHT: WEIGHT_COLUMNS,
            KIND_CHUNK: ("seq", "stamp_ns", "step_ms", "step_index")
            + _device_columns(
                arm_names=self.__arm_names,
                arm_dof=arm_dof,
            ),
            KIND_TARGET: ("now_ns", "stage")
            + _device_columns(
                arm_names=self.__arm_names,
                arm_dof=arm_dof,
            ),
        }
        self.__files: dict[str, object] = {}
        self.__writers: dict[str, csv.writer] = {}

    def start(self) -> None:
        """Open the files and write the header of every fresh file."""
        if self.__writers:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        for kind in RECORDER_KINDS:
            path = self.directory / f"{kind}.csv"
            fresh = not path.exists()
            handle = path.open("a", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            if fresh:
                writer.writerow(self.columns[kind])
            self.__files[kind] = handle
            self.__writers[kind] = writer

    def close(self) -> None:
        """Close the files; safe to call twice."""
        for handle in self.__files.values():
            handle.close()
        self.__files = {}
        self.__writers = {}

    def record_state(self, now_ns: int, kind: str, old: str, new: str,
                     detail: str = "") -> None:
        """Record one migration or internal event.

        Args:
            now_ns (int): Time of the record.
            kind (str): ``change`` or ``event``.
            old (str): State before, empty for an event.
            new (str): State after, empty for an event.
            detail (str): Event name or free note.
        """
        self.__write(
            kind=KIND_STATE,
            row=(now_ns, kind, old, new, detail),
        )

    def record_weight(
        self,
        now_ns: int,
        seq: int,
        weight: float,
    ) -> None:
        """Record the weight one chunk contributed to the current target.

        Args:
            now_ns (int): Time of the record.
            seq (int): Sequence number of the chunk.
            weight (float): Weight it was given.
        """
        self.__write(
            kind=KIND_WEIGHT,
            row=(now_ns, seq, f"{weight:.6f}"),
        )

    def record_chunk(self, seq: int, stamp_ns: int, step_ms: int,
                     step_index: int, arm_row: dict,
                     grip_row: dict) -> None:
        """Record one step of a decoded chunk.

        Args:
            seq (int): Sequence number of the chunk.
            stamp_ns (int): Stamp of the chunk.
            step_ms (int): Step used for the chunk.
            step_index (int): Index of the step inside the chunk.
            arm_row (dict): Arm name to joint values of this step.
            grip_row (dict): Arm name to gripper value of this step.
        """
        row = [seq, stamp_ns, step_ms, step_index]
        row.extend(_device_values(
            arm_names=self.__arm_names,
            arm_row=arm_row,
            grip_row=grip_row,
        ))
        self.__write(
            kind=KIND_CHUNK,
            row=tuple(row),
        )

    def record_target(self, now_ns: int, stage: str,
                      target: dict) -> None:
        """Record one target, before or after the step limit.

        Args:
            now_ns (int): Time of the record.
            stage (str): ``STAGE_RAW`` or ``STAGE_LIMITED``.
            target (dict): Target keyed by command type and device name.
        """
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        arm_row = {name: getattr(command, "pos", [])
                   for name, command in target.get("arm", {}).items()}
        grip_row = {name: getattr(command, "pos", [])
                    for name, command in target.get("grip", {}).items()}
        row = [now_ns, stage]
        row.extend(_device_values(
            arm_names=self.__arm_names,
            arm_row=arm_row,
            grip_row=grip_row,
        ))
        self.__write(
            kind=KIND_TARGET,
            row=tuple(row),
        )

    def __write(
        self,
        kind: str,
        row: tuple,
    ) -> None:
        """Write one row when the recorder is open.

        Args:
            kind (str): Recorder kind.
            row (tuple): Values in the column order of that kind.
        """
        writer = self.__writers.get(kind)
        if writer is None:
            return
        writer.writerow(row)


def _device_columns(arm_names: tuple[str, ...],
                    arm_dof: int) -> tuple[str, ...]:
    """Build the per device column names.

    Args:
        arm_names (tuple): Arm names from the device hub.
        arm_dof (int): Degrees of freedom of one arm.

    Returns:
        tuple: Column names, generated per device and joint.
    """
    columns = []
    for name in arm_names:
        columns.extend(f"{name}_jnt_{index}" for index in range(arm_dof))
        columns.append(f"{name}_grip")
    return tuple(columns)


def _device_values(arm_names: tuple[str, ...], arm_row: dict,
                   grip_row: dict) -> list:
    """Flatten per device values into the column order.

    Args:
        arm_names (tuple): Arm names from the device hub.
        arm_row (dict): Arm name to joint values.
        grip_row (dict): Arm name to gripper values.

    Returns:
        list: Values for one row, empty where a device has no value.
    """
    values = []
    for name in arm_names:
        # ``or`` cannot be used here: the values arrive as arrays, and the
        # truth value of an array with several elements is undefined.
        joints = arm_row.get(name)
        values.extend([] if joints is None else list(joints))
        grip = grip_row.get(name)
        grip_values = [] if grip is None else list(grip)
        values.append(grip_values[0] if grip_values else "")
    return values
