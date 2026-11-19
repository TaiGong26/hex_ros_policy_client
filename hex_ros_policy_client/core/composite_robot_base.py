################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Abstract composite robot.

The robot is the verb of the device side: it creates and releases the devices,
merges their samples into one grouped dictionary and transposes a target into
per device commands. It never parses the device table itself; that is the job
of :class:`DeviceHub`.

Only four methods are public. Anything else - a second observation entry, a
per mode accessor - would give the layers above a second way to reach the
devices.
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any

from ..struct.cmd_struct import TARGET_ARM, TARGET_GRIP
from ..utils import drop
from ..device.device_hub import DeviceHub


class CompositeRobotBase(ABC):
    """Base class of every composite robot.

    Attributes:
        device_path (str): Path of the device table, held as a string only.
    """

    def __init__(
        self,
        node: Any,
        device_path: str,
    ) -> None:
        """Store the node and the table path.

        Args:
            node: ROS node the devices bind their endpoints to.
            device_path (str): Path of the device table.
        """
        self.__node = node
        self.device_path = device_path
        self.__device_hub: DeviceHub | None = None

    @property
    def device_hub(self) -> DeviceHub | None:
        """Devices of this robot, or ``None`` before start.

        Returns:
            DeviceHub | None: Current hub.
        """
        return self.__device_hub

    def start(self) -> None:
        """Create the devices declared by the device table.

        The call is idempotent: starting twice keeps the devices created by the
        first call.
        """
        if self.__device_hub is not None:
            return
        self.__device_hub = DeviceHub.from_table(
            node=self.__node,
            table_path=self.device_path,
        )

    def stop(self) -> None:
        """Release every device.

        The call is idempotent: stopping twice releases nothing twice.
        """
        if self.__device_hub is None:
            return
        hub = self.__device_hub
        self.__device_hub = None
        for devices in (hub.camera_devices, hub.arm_devices):
            for device in devices.values():
                device._release()

    def read_obs(self) -> dict:
        """Merge the newest sample of every device.

        Returns:
            dict: ``{"camera": {name: sample}, "arm": {name: sample}}`` with
            the keys of the device table, ``None`` where a device has no
            sample yet.
        """
        hub = self.__device_hub
        if hub is None:
            return {"camera": {}, "arm": {}}
        return {
            "camera": {name: device.get_obs()
                       for name, device in hub.camera_devices.items()},
            "arm": {name: device.get_obs()
                    for name, device in hub.arm_devices.items()},
        }

    def send_ctrl(self, target: dict) -> None:
        """Transpose a target into per device commands and dispatch them.

        The target is organised by command type, the devices consume it by
        device, so the transpose happens here exactly once and never travels
        further up.

        Args:
            target (dict): ``{TARGET_ARM: {name: command},
                TARGET_GRIP: {name: command}}``.
        """
        routed = _route_by_device(target=target)
        hub = self.__device_hub
        if hub is None:
            return
        for name, commands in routed.items():
            device = hub.arm_devices.get(name)
            if device is None:
                drop(
                    logger=_robot_logger(node=self.__node),
                    point="ctrl.route",
                    obj="command dropped",
                    cause=f"device {name} is not in arm_devices, "
                    f"no control topic",
                    reason="unknown_device",
                )
                continue
            for command in commands:
                device.send_ctrl(command=command)


def _route_by_device(target: dict) -> dict[str, list]:
    """Turn a type first target into device first commands.

    Args:
        target (dict): Target keyed by command type.

    Returns:
        dict: Device name to the commands addressed to it.
    """
    routed: dict[str, list] = {}
    for group in (TARGET_ARM, TARGET_GRIP):
        for name, command in target.get(group, {}).items():
            routed.setdefault(name, []).append(command)
    return routed


def _robot_logger(node: Any) -> Any:
    """Pick the logger of the ROS node, falling back to the stdlib one.

    Args:
        node: ROS node or a test double.

    Returns:
        Any: Logger that exposes ``log(level, message)``.
    """
    getter = getattr(node, "get_logger", None)
    if callable(getter):
        return getter()
    return logging.getLogger(__name__)
