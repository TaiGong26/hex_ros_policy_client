################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Observation assembler with a fixed step time base.

The assembler keeps only the newest grouped observation. Every source carries
its own stamp, which is used for the freshness decision; the frame stamp itself
comes from the fixed step strategy, so consecutive frames are exactly one
observation period apart.

A frame is discarded as a whole when a single required source is missing or too
old. Partial filling would silently pair data from different instants, which is
worse than not publishing at all.
"""

from __future__ import annotations

from typing import Callable

from ..struct.device_struct import ObservationFrame
from .observation_base import ObservationBase


class HistoryObservation(ObservationBase):
    """Observation assembler for the current topology."""

    def __init__(self,
                 camera_names: tuple[str, ...],
                 arm_names: tuple[str, ...],
                 freshness_ns: int,
                 period_ns: int,
                 now_fn: Callable[[], int],
                 sync_fn: Callable[[dict, dict, int], int],
                 logger=None) -> None:
        """Create an assembler without any sample.

        Args:
            camera_names (tuple): Camera names of the topology.
            arm_names (tuple): Arm names of the topology.
            freshness_ns (int): Age above which a source counts as stale.
            period_ns (int): Observation period of the frame stamp.
            now_fn (Callable): Clock injected by the node.
            sync_fn (Callable): Time sync strategy from the table.
            logger (Any | None): Logger used for drop lines.
        """
        super().__init__(
            camera_names=camera_names,
            arm_names=arm_names,
            freshness_ns=freshness_ns,
            period_ns=period_ns,
            now_fn=now_fn,
            sync_fn=sync_fn,
            logger=logger,
        )
        self.__latest: dict | None = None

    def push_obs(self, observation: dict) -> None:
        """Store the newest grouped observation.

        Args:
            observation (dict): Output of ``Robot.read_obs()``.
        """
        self.__latest = observation

    def get_obs(self) -> ObservationFrame | None:
        """Assemble one observation frame.

        Returns:
            ObservationFrame | None: Frame with ``stamp_ns``, or ``None`` when
            a required source is missing or too old. Nothing is published in
            that case.
        """
        if self.__latest is None:
            return None
        problems = self._bad_sources(observation=self.__latest)
        if problems:
            self._report_drop(problems=problems)
            return None
        return {
            "stamp_ns": self._next_stamp(observation=self.__latest),
            "camera": dict(self.__latest["camera"]),
            "arm": dict(self.__latest["arm"]),
        }

    def reset(self) -> None:
        """Forget the stored sample and restart the frame sequence."""
        self.__latest = None
        self._reset_sync()
