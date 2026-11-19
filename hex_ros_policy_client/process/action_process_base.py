################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Abstract action chunk processor.

The processor turns the chunks received from the model into the target of the
current tick. It is a pure computation class: it holds no device, performs no
transport and never reads the clock on its own, so it can be tested offline
with recorded chunks.

Only three methods are public. The interpolation function is injected at
construction time, which means the way a target is reached is a parameter, not
a code path inside the class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class ActionProcessBase(ABC):
    """Base class of the action chunk processors.

    Attributes:
        arm_names (tuple): Arm names of the topology.
    """

    def __init__(self,
                 arm_names: tuple[str, ...],
                 now_fn: Callable[[], int],
                 interpolator: Callable[..., object],
                 logger=None) -> None:
        """Store the topology, the injected clock and the interpolator.

        Args:
            arm_names (tuple): Arm names of the topology.
            now_fn (Callable): Clock injected by the node.
            interpolator (Callable): Interpolation function from the table.
            logger (Any | None): Logger used for drop lines.
        """
        self.arm_names = tuple(arm_names)
        self._now_fn = now_fn
        self._interpolator = interpolator
        self._logger = logger

    @abstractmethod
    def push_chunk(self, chunk: dict) -> None:
        """Accept one chunk received from the model side.

        Args:
            chunk (dict): Decoded chunk with ``stamp_ns``,
                ``chunk_step_interval_ms``, ``action_mode``, ``arm_ctrl_mode``
                and one ``steps x dof`` step list per device under ``arm`` and
                ``grip``; turning those lists into arrays is the job of this
                layer, not of the protocol helper.
        """

    @abstractmethod
    def get_target(self) -> dict | None:
        """Return the target of the current tick.

        Returns:
            dict | None: Target keyed by ``TARGET_ARM`` and ``TARGET_GRIP``,
            ``None`` when no chunk covers the current instant.
        """

    @abstractmethod
    def reset(self) -> None:
        """Drop every stored chunk."""
