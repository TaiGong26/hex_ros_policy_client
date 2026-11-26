################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from typing import Any, Callable

import yaml

from ..device.arm_device import HexArmDevice
from ..device.camera_device import CameraDevice
from ..struct.cmd_struct import ControlIntent

SECTION_CAMERAS = "cameras"
SECTION_ARMS = "arms"


class DualManipRobot:
    """Robot assembled from a device table.

    Attributes:
        device_names (dict): Section name to the device names in table
            order, ``None`` before start and after stop.
    """

    _DEVICE_BUILDERS = {
        SECTION_CAMERAS: CameraDevice,
        SECTION_ARMS: HexArmDevice,
    }
    _COMMAND_ARM = "arm"
    _COMMAND_GRIP = "grip"

    def __init__(
        self,
        device_path: str,
        logger: Any,
        create_subscription: Callable,
        create_publisher: Callable,
        now_fn: Callable[[], int] | None = None,
    ) -> None:
        """Store the table path and the endpoint factories.

        Args:
            device_path (str): Path of the device table.
            logger (Any): Logger handed down to every device.
            create_subscription (Callable): ``(msg_type, topic, callback)``
                creating and returning a subscription handle.
            create_publisher (Callable): ``(msg_type, topic)`` creating and
                returning a publisher handle.
            now_fn (Callable[[], int] | None): Clock handed down to every
                device, returning the current time in nanoseconds; ``None``
                lets a device fall back to the wall clock.
        """
        self.__device_path = device_path
        self.__logger = logger
        self.__create_subscription = create_subscription
        self.__create_publisher = create_publisher
        self.__now_fn = now_fn
        self.__devices: dict[str, dict[str, CameraDevice | HexArmDevice ]] | None = None
        self.device_names: dict[str, tuple[str, ...]] = {}

    def start(self) -> None:
        """Create the devices declared by the device table.

        The call is idempotent: starting twice keeps the devices created by the
        first call, and starting after a stop builds them again.

        Raises:
            FileNotFoundError: If the table file does not exist.
            ValueError: If the table is empty, malformed or declares an
                unknown section.
        """
        if self.__devices is not None:
            return
        table = self.__read_table(table_path=self.__device_path)
        devices = self.__build_devices(table=table)
        self.__devices = devices
        self.device_names = {section: tuple(entries)
                             for section, entries in devices.items()}

    def stop(self) -> None:
        """Release every device.

        The call is idempotent. The endpoints belong to the node that created
        them, so nothing is released here; the devices and their names are
        dropped, which is what makes a later start build them again.
        """
        self.__devices = None
        self.device_names = {}

    def read_obs(self) -> dict:
        """Merge the newest sample of every device.

        The device queues are read without draining, because the assembly
        needs a pair of samples that brackets the target instant; clearing a
        queue on every tick would leave it empty most of the time and no
        pair could ever be found.

        Returns:
            dict: ``{"camera": {name: sample}, "arm": {name: sample}}`` with
            the keys of the device table, ``None`` where a device has no
            sample yet, and both groups empty before start and after stop.
        """
        if self.__devices is None:
            return {"camera": {}, "arm": {}}
        return {
            "camera": {name: device.get_obs(drain=False)
                       for name, device
                       in self.__devices[SECTION_CAMERAS].items()},
            "arm": {name: device.get_obs(drain=False)
                    for name, device
                    in self.__devices[SECTION_ARMS].items()},
        }

    def send_ctrl(self, target: ControlIntent) -> None:
        """Transpose a target into per device commands and dispatch them.

        The target is organised by command type, the devices consume it by
        device, so the transpose happens here exactly once and never travels
        further up. Both commands of one device travel in a single message.

        Args:
            target (ControlIntent): Command intent grouped by command type.
        """
        if self.__devices is None:
            self.__logger.logw(
                "[ctrl.route/not_started] command dropped: robot is stopped")
            return
        routed: dict[str, dict] = {}
        for key, commands in ((self._COMMAND_ARM, target.arm),
                              (self._COMMAND_GRIP, target.grip)):
            for name, command in commands.items():
                routed.setdefault(name, {})[key] = command
        for name, commands in routed.items():
            device = self.__devices[SECTION_ARMS].get(name)
            if device is None:
                self.__logger.logw(
                    "[ctrl.route/unknown_device] command dropped: "
                    f"device {name} is not in arm_devices, no control topic")
                continue
            arm_command = commands.get(self._COMMAND_ARM)
            if arm_command is None:
                self.__logger.logw(
                    "[ctrl.route/no_arm_command] command dropped: "
                    f"device {name} has a gripper command but no arm command")
                continue
            
            if isinstance(device, HexArmDevice):
                device.send_ctrl(
                    arm_command=arm_command,
                    grip_command=commands.get(self._COMMAND_GRIP),
                    target_time_ns=target.stamp_ns,
                )


    def move_to_stable(
        self,
        duration_ns: int,
        arrive_threshold: float,
    ) -> bool:
        """Ramp every arm to its stable pose.

        Every device is asked on every call, so no arm is left behind when
        another one already arrived.

        Args:
            duration_ns (int): Length of the ramp, in nanoseconds.
            arrive_threshold (float): Largest per joint distance to the stable
                pose that still counts as arrived, in radians.

        Returns:
            bool: ``True`` once every arm reported the arrival.
        """
        if self.__devices is None:
            self.__logger.logw(
                "[stable.route/not_started] stable skipped: robot is stopped")
            return False
        arrived = True
        for device in self.__devices[SECTION_ARMS].values():
            if not isinstance(device, HexArmDevice):
                self.__logger.logw(
                    "[stable.route/unknown_device] stable skipped: "
                    f"device {device.name} is not a HexArmDevice")
                continue
            if not device.move_to_stable(duration_ns=duration_ns,
                                         arrive_threshold=arrive_threshold):
                arrived = False
        return arrived

    def hold_position(self) -> None:
        """Re-send the pose every arm reached when the ramp finished."""
        if self.__devices is None:
            self.__logger.logw(
                "[hold.route/not_started] hold skipped: robot is stopped")
            return
        for device in self.__devices[SECTION_ARMS].values():
            if not isinstance(device, HexArmDevice):
                self.__logger.logw(
                    "[hold.route/unknown_device] hold skipped: "
                    f"device {device.name} is not a HexArmDevice")
                continue
            device.hold_position()

    def __read_table(self, table_path: str) -> dict:
        """Read the device table file.

        Args:
            table_path (str): Path of the table.

        Returns:
            dict: Parsed table.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file does not hold a mapping, or declares a
                section the robot does not know.
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
        
        unknown = sorted(set(table) - set(self._DEVICE_BUILDERS))
        if unknown:
            raise ValueError(
                f"device table {table_path} declares unknown sections "
                f"{unknown}, expected one of {sorted(self._DEVICE_BUILDERS)}")
        
        return table

    def __build_devices(self, table: dict) -> dict[str, dict]:
        """Create every device declared by the table.

        Args:
            table (dict): Parsed device table.

        Returns:
            dict: Section name to devices.

        Raises:
            ValueError: If a section is malformed, a name is repeated, a
                required key is missing or the table declares no device. The
                device classes raise on their own configuration.
        """
        devices: dict[str, dict] = {
            section: {} for section in self._DEVICE_BUILDERS}
        seen: dict[str, str] = {}
        for section, builder in self._DEVICE_BUILDERS.items():
            entries = table.get(section) or {}
            if not isinstance(entries, dict):
                raise ValueError(
                    f"device table section {section} must be a mapping")
            for name, params in entries.items():
                self.__check_entry(
                    section=section,
                    name=name,
                    params=params,
                    required=builder.REQUIRED_KEYS,
                )
                if name in seen:
                    raise ValueError(
                        f"device name {name!r} appears in both {seen[name]} "
                        f"and {section}")
                seen[name] = section
                devices[section][name] = builder(
                    name,
                    params,
                    self.__create_subscription,
                    self.__create_publisher,
                    self.__logger,
                    now_fn=self.__now_fn,
                )
        if not any(devices.values()):
            raise ValueError("device table declares no device")
        return devices

    @staticmethod
    def __check_entry(section: str, 
                      name: str, 
                      params: Any,
                      required: tuple) -> None:
        """Reject an entry that cannot be turned into a device.

        Args:
            section (str): Section the entry comes from.
            name (str): Device name, that is the key of the entry.
            params (Any): Entry contents.
            required (tuple): Top level keys the device class needs.

        Raises:
            ValueError: If the entry is not a mapping or a key is missing.
        """
        if not isinstance(params, dict):
            raise ValueError(f"entry of device {name!r} must be a mapping")
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(
                f"device {name!r} ({section}) is missing required keys "
                f"{missing}")
