################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Abstract device.

A device is a topic, not a controller: it binds its subscription in the
constructor and is released by the robot that owns it. There is deliberately no
``start`` or ``stop`` here, so no lifecycle state can drift out of sync.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DeviceBase(ABC):
    """Base class of every device.

    Subclasses receive data through :meth:`on_states`, which is their own ROS
    subscription callback, and expose the newest sample through
    :meth:`get_obs`. The robot pulls with ``get_obs`` and never pushes.
    """

    @abstractmethod
    def on_states(self, states: Any) -> None:
        """Store the newest sample of this device.

        Args:
            states: Message delivered by the subscription.
        """

    @abstractmethod
    def get_obs(self) -> Any:
        """Return the newest sample.

        Returns:
            Any: Latest observation, or ``None`` before the first message.
        """

    def _release(self) -> None:
        """Release the handles this device created.

        The method is internal: releasing belongs to the robot that owns the
        device, and a device must not expose a lifecycle of its own.
        """
        return None
