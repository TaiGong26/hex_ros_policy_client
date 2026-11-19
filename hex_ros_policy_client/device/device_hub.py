################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Device container and factory.

The hub answers one question: which devices exist for this topology. It is a
noun, not a behaviour: it reads the device table, validates it without any
fallback and builds the devices, then holds the two dictionaries. Reading
observations, sending commands and the device lifecycle belong to the robot, so
that every new robot reuses this single parser.

Attributes:
    DEVICE_BUILDERS: Table type to implementation class; a new device type adds
        one entry here and one file next to the existing devices.
    TABLE_SECTIONS: Sections of the table and the group they feed.
"""

from __future__ import annotations

from typing import Any

import yaml

from .arm_device import ArmDevice
from .camera_device import CameraDevice

DEVICE_BUILDERS = {
    ArmDevice.TYPE: ArmDevice,
    CameraDevice.TYPE: CameraDevice,
}
TABLE_SECTIONS = ("cameras", "arms")


class DeviceHub:
    """Holds the devices of one robot.

    Attributes:
        camera_devices (dict): Camera name to :class:`CameraDevice`.
        arm_devices (dict): Arm name to :class:`ArmDevice`.
        camera_names (tuple): Camera names in table order.
        arm_names (tuple): Arm names in table order.
    """

    def __init__(
        self,
        camera_devices: dict,
        arm_devices: dict,
    ) -> None:
        """Store the two device dictionaries.

        Args:
            camera_devices (dict): Camera name to device.
            arm_devices (dict): Arm name to device.
        """
        self.__camera_devices = camera_devices
        self.__arm_devices = arm_devices

    @property
    def camera_devices(self) -> dict:
        """Cameras of this robot.

        Returns:
            dict: Camera name to device.
        """
        return self.__camera_devices

    @property
    def arm_devices(self) -> dict:
        """Arms of this robot.

        Returns:
            dict: Arm name to device.
        """
        return self.__arm_devices

    @property
    def camera_names(self) -> tuple[str, ...]:
        """Camera names in table order.

        Returns:
            tuple: Camera names.
        """
        return tuple(self.__camera_devices)

    @property
    def arm_names(self) -> tuple[str, ...]:
        """Arm names in table order.

        Returns:
            tuple: Arm names.
        """
        return tuple(self.__arm_devices)

    @staticmethod
    def from_table(node: Any, table_path: str) -> "DeviceHub":
        """Read the device table and build every declared device.

        Args:
            node: ROS node the devices bind their endpoints to.
            table_path (str): Path of the device table.

        Returns:
            DeviceHub: Hub holding the created devices.

        Raises:
            FileNotFoundError: If the table file does not exist.
            ValueError: If the table is empty, malformed or declares an
                unknown device type.
        """
        table = _read_table(table_path=table_path)
        devices = {builder_type: {}
                   for builder_type in DEVICE_BUILDERS}
        seen: dict[str, str] = {}
        fields: dict[str, str] = {}
        for section in TABLE_SECTIONS:
            entries = table.get(section) or {}
            if not isinstance(entries, dict):
                raise ValueError(
                    f"device table section {section} must be a mapping")
            for name, params in entries.items():
                device = _build_device(
                    node=node,
                    section=section,
                    name=name,
                    params=params,
                )
                if name in seen:
                    raise ValueError(
                        f"device name {name!r} appears in both {seen[name]} "
                        f"and {section}")
                seen[name] = section
                _model_field(
                    section=section,
                    name=name,
                    params=params,
                    fields=fields,
                )
                devices[device.TYPE][name] = device
        if not any(devices.values()):
            raise ValueError(f"device table {table_path} declares no device")
        return DeviceHub(
            camera_devices=devices[CameraDevice.TYPE],
            arm_devices=devices[ArmDevice.TYPE],
        )


def _model_field(section: str, name: str, params: dict,
                 fields: dict[str, str]) -> str | None:
    """Read and check the optional ``model_field`` of one device.

    Args:
        section (str): Section of the table the device comes from.
        name (str): Device name.
        params (dict): Declared parameters of the device.
        fields (dict): Field names already claimed, updated in place.

    Returns:
        str | None: Declared field name, ``None`` when the device keeps its
        own name.

    Raises:
        ValueError: If the field is not a string or is already claimed.
    """
    if "model_field" not in params:
        return None
    field = params["model_field"]
    if not isinstance(field, str) or not field:
        raise ValueError(
            f"model_field of device {name!r} ({section}) must be a non empty "
            f"string")
    owner = fields.get(field)
    if owner is not None:
        raise ValueError(
            f"model_field {field!r} is claimed by both {owner!r} and {name!r}")
    fields[field] = name
    return field


def _read_table(table_path: str) -> dict:
    """Read the device table file.

    Args:
        table_path (str): Path of the table.

    Returns:
        dict: Parsed table.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not hold a mapping.
    """
    try:
        with open(table_path, "r", encoding="utf-8") as handle:
            table = yaml.safe_load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"device table not found: {table_path}") from None
    if not isinstance(table, dict):
        raise ValueError(
            f"top level of device table {table_path} must be a mapping")
    return table


def _build_device(node: Any, section: str, name: str,
                  params: Any) -> Any:
    """Validate one table entry and create the device it declares.

    Args:
        node: ROS node the device binds to.
        section (str): Table section the entry came from.
        name (str): Device name, that is the key of the entry.
        params (Any): Entry contents.

    Returns:
        Any: Created device.

    Raises:
        ValueError: If the entry is malformed or the type is unknown.
    """
    if not isinstance(params, dict):
        raise ValueError(f"entry of device {name!r} must be a mapping")
    device_type = params.get("type")
    builder = DEVICE_BUILDERS.get(device_type)
    if builder is None:
        raise ValueError(
            f"type={device_type!r} of device {name!r} is unknown, "
            f"expected one of {sorted(DEVICE_BUILDERS)}")
    missing = [key for key in builder.REQUIRED_KEYS if key not in params]
    if missing:
        raise ValueError(
            f"device {name!r} ({section}) is missing required keys {missing}")
    return builder(node, name, params)
