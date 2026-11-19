################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Action chunk processor with time integration.

A chunk is a sequence of future targets spaced by one step. Several chunks
usually overlap in time, and the model revises its plan with every new chunk,
so the target of one device is the sample of every covering chunk merged by the
configured strategy.

Two decisions live outside this class, in tables, so that they can be replaced
without touching it: the restorer that turns increments into absolute
positions, and the weights that merge several samples. This class only owns the
time axis: which chunk covers the current instant, and which sample of it does.

Attributes:
    DEFAULT_CAPACITY: Number of chunks kept before the oldest is dropped.
    DEFAULT_STEP_MS: Step used when a chunk does not carry a positive one.
    DEFAULT_AVERAGE: Strategy used when the configuration names none.
    NS_PER_MS: Nanoseconds in one millisecond.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable

import numpy as np

from ..struct.cmd_struct import ARM_CTRL_MODES, GRIP_CTRL_MODES
from ..struct.cmd_struct import (
    TARGET_ARM,
    TARGET_GRIP,
    ArmCommand,
    GripCommand,
)
from ..utils import drop
from ..utils import (
    DEFAULT_DECAY,
    weight_ema,
    weight_recency,
)
from .action_process_base import ActionProcessBase
from .action_restore import restore

DEFAULT_CAPACITY = 100
DEFAULT_STEP_MS = 33
DEFAULT_AVERAGE = "exponential"
MERGE_EXPONENTIAL = "exponential"
MERGE_PLAIN = "plain"
MERGE_STRATEGIES = (MERGE_EXPONENTIAL, MERGE_PLAIN)
NS_PER_MS = 1_000_000

_CHUNK_KEYS = ("stamp_ns", "chunk_step_interval_ms", "action_mode",
               "arm_ctrl_mode", "arm", "grip")


class TimeIntegrationChunk(ActionProcessBase):
    """Action processor that merges the overlapping part of several chunks.

    Attributes:
        last_weights (dict): Sequence number to weight used by the last call of
            :meth:`get_target`, exposed for the recorder.
    """

    def __init__(self,
                 arm_names: tuple[str, ...],
                 now_fn: Callable[[], int],
                 interpolator: Callable[..., np.ndarray],
                 capacity: int = DEFAULT_CAPACITY,
                 step_ms_default: int = DEFAULT_STEP_MS,
                 average_strategy: str = DEFAULT_AVERAGE,
                 decay: float = DEFAULT_DECAY,
                 logger: Any | None = None) -> None:
        """Create a processor without any chunk.

        Args:
            arm_names (tuple): Arm names of the topology.
            now_fn (Callable): Clock injected by the node.
            interpolator (Callable): Interpolation function from the table.
            capacity (int): Maximum number of chunks kept.
            step_ms_default (int): Step used when a chunk carries none.
            average_strategy (str): Name of the weight table used to merge the
                samples of one device.
            decay (float): Ratio between two neighbouring samples of the
                exponential strategy.
            logger (Any | None): Logger used for drop lines.

        Raises:
            ValueError: If the capacity, the strategy or the decay is invalid.
        """
        super().__init__(
            arm_names=arm_names,
            now_fn=now_fn,
            interpolator=interpolator,
            logger=logger,
        )
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if step_ms_default <= 0:
            raise ValueError(
                f"step_ms_default must be positive, got {step_ms_default}")
        # Fails at construction, never at the first frame.
        if average_strategy not in MERGE_STRATEGIES:
            raise ValueError(
                f"unknown weighting {average_strategy!r}, "
                f"expected one of {MERGE_STRATEGIES}")
        if not np.isfinite(float(decay)):
            raise ValueError(f"decay must be finite, got {decay}")
        self.__capacity = capacity
        self.__step_ns_default = int(step_ms_default) * NS_PER_MS
        self.__average = average_strategy
        self.__decay = float(decay)
        self.__chunks: deque = deque(maxlen=capacity)
        self.__serial = 0
        self.__dropped = 0
        self.__unsupported = 0
        self.__used_weights: dict[int, float] = {}
        self.last_weights: dict[int, float] = {}

    def push_chunk(self, chunk: dict) -> None:
        """Accept one chunk received from the model side.

        Two steps happen here, in this order: the steps of the chunk are
        restored to absolute positions with the restorer of ``action_mode``,
        and what stays in the queue is therefore already absolute. The merge of
        overlapping chunks happens later, in :meth:`get_target`.

        Args:
            chunk (dict): Decoded chunk of the protocol.
        """
        try:
            normalised = self.__normalise(chunk=chunk)
        except (ValueError, TypeError) as error:
            self.__dropped += 1
            drop(
                logger=self._logger,
                point="chunk.push",
                obj="action chunk dropped",
                cause=f"invalid structure: {error}",
                reason="schema_invalid",
                count=self.__dropped,
            )
            return
        if normalised is None:
            return
        if len(self.__chunks) == self.__capacity:
            old = self.__chunks[0]
            drop(
                logger=self._logger,
                point="chunk.queue",
                obj="action chunk dropped",
                cause=f"capacity {self.__capacity} is full, dropping the "
                f"oldest seq={old['seq']}",
                reason="full",
            )
        self.__chunks.append(normalised)

    def get_target(self) -> dict | None:
        """Return the target of the current tick.

        Returns:
            dict | None: Target keyed by ``TARGET_ARM`` and ``TARGET_GRIP``,
            ``None`` when no chunk covers the current instant.
        """
        now_ns = self._now_fn()
        self.__prune(now_ns=now_ns)
        self.__used_weights = {}
        arms = self.__group_target(
            group=TARGET_ARM,
            now_ns=now_ns,
        )
        grips = self.__group_target(
            group=TARGET_GRIP,
            now_ns=now_ns,
        )
        self.last_weights = dict(self.__used_weights)
        if not arms and not grips:
            drop(
                logger=self._logger,
                point="chunk.target",
                obj="no target this tick",
                cause="no chunk covers this instant, holding the "
                "measured snapshot",
                reason="queue_empty",
                level=logging.DEBUG,
            )
            return None
        commands: dict[str, dict] = {TARGET_ARM: {}, TARGET_GRIP: {}}
        for name, value in arms.items():
            commands[TARGET_ARM][name] = ArmCommand(pos=value)
        for name, value in grips.items():
            commands[TARGET_GRIP][name] = GripCommand(pos=value)
        return commands

    def reset(self) -> None:
        """Drop every stored chunk."""
        self.__chunks.clear()
        self.last_weights = {}

    def __group_target(
        self,
        group: str,
        now_ns: int,
    ) -> dict[str, np.ndarray]:
        """Merge the covering samples of one command group.

        Args:
            group (str): ``TARGET_ARM`` or ``TARGET_GRIP``.
            now_ns (int): Current time.

        Returns:
            dict: Device name to the target of this tick.
        """
        samples: dict[str, list[np.ndarray]] = {}
        for chunk in reversed(self.__chunks):
            for name, matrix in chunk[group].items():
                value = self.__sample(
                    chunk=chunk,
                    matrix=matrix,
                    now_ns=now_ns,
                )
                if value is None:
                    continue
                samples.setdefault(name, []).append(value)
        target = {}
        for name, values in samples.items():
            target[name] = self.__merge(samples=values)
        return target

    def __merge(self, samples: list) -> np.ndarray:
        """Merge the samples covering this instant for one device.

        Args:
            samples (list): Samples ordered oldest first.

        Returns:
            np.ndarray: Merged target.
        """
        if self.__average == MERGE_EXPONENTIAL:
            return weight_recency(samples=samples, weight=self.__decay)
        merged = samples[0]
        for sample in samples[1:]:
            merged = weight_ema(
                previous=merged,
                current=sample,
                weight=self.__decay,
            )
        return merged

    def __sample(self, chunk: dict, matrix: np.ndarray,
                 now_ns: int) -> np.ndarray | None:
        """Interpolate one device matrix at the current instant.

        Args:
            chunk (dict): Normalised chunk.
            matrix (np.ndarray): Absolute samples of shape ``(steps, dof)``.
            now_ns (int): Current time.

        Returns:
            np.ndarray | None: Interpolated value, ``None`` when the chunk does
            not cover the current instant.
        """
        elapsed_ns = now_ns - chunk["stamp_ns"]
        if elapsed_ns < 0:
            return None
        step_ns = chunk["step_ns"]
        steps = matrix.shape[0]
        index = int(elapsed_ns // step_ns)
        if index >= steps:
            return None
        if index == steps - 1:
            return matrix[index]
        elapsed_in_step = elapsed_ns - index * step_ns
        return self._interpolator(matrix[index], matrix[index + 1],
                                  elapsed_in_step, step_ns)

    def __prune(self, now_ns: int) -> None:
        """Drop the chunks that lie entirely in the past.

        Args:
            now_ns (int): Current time.
        """
        while self.__chunks:
            chunk = self.__chunks[0]
            steps = max(matrix.shape[0]
                        for group in (TARGET_ARM, TARGET_GRIP)
                        for matrix in chunk[group].values())
            if now_ns < chunk["stamp_ns"] + steps * chunk["step_ns"]:
                return
            self.__chunks.popleft()

    def __normalise(self, chunk: dict) -> dict | None:
        """Validate one chunk, restore its steps and tag it with a sequence.

        Args:
            chunk (dict): Chunk as delivered by the middleware.

        Returns:
            dict | None: Chunk with absolute matrices, ``None`` when the chunk
            was dropped as unsupported.

        Raises:
            ValueError: If a required key is missing or a matrix is empty.
        """
        missing = [key for key in _CHUNK_KEYS if key not in chunk]
        if missing:
            raise ValueError(f"missing fields {missing}")
        arm_mode = chunk["arm_ctrl_mode"]
        if arm_mode not in ARM_CTRL_MODES:
            raise ValueError(f"unknown arm_ctrl_mode={arm_mode!r}")
        if arm_mode != ARM_CTRL_MODES[0]:
            self.__unsupported += 1
            drop(
                logger=self._logger,
                point="chunk.mode",
                obj="action chunk dropped",
                cause=f"arm_ctrl_mode={chunk['arm_ctrl_mode']} is not "
                f"dispatched at this stage",
                reason="unsupported_mode",
                count=self.__unsupported,
            )
            return None
        step_ms = chunk["chunk_step_interval_ms"]
        step_ns = (self.__step_ns_default if not step_ms
                   else int(step_ms) * NS_PER_MS)
        if step_ns <= 0:
            raise ValueError(
                f"chunk_step_interval_ms must be positive, got {step_ms}")
        self.__serial += 1
        normalised = {
            "seq": self.__serial,
            "stamp_ns": int(chunk["stamp_ns"]),
            "step_ns": step_ns,
            "action_mode": chunk["action_mode"],
        }
        for group in (TARGET_ARM, TARGET_GRIP):
            entries = chunk[group]
            if not isinstance(entries, dict):
                raise ValueError(f"{group} is not a mapping")
            matrices = {}
            for name, values in entries.items():
                matrix = np.asarray(values, dtype=np.float64)
                if matrix.ndim == 1:
                    matrix = matrix.reshape(1, -1)
                if matrix.ndim != 2 or matrix.shape[0] == 0:
                    raise ValueError(f"{group}.{name} has an invalid shape")
                matrices[name] = restore(
                    action_mode=chunk["action_mode"],
                    steps=matrix,
                )
            if not matrices:
                raise ValueError(f"{group} is empty")
            normalised[group] = matrices
        return normalised
