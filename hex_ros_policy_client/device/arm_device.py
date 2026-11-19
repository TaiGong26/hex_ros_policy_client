################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Arm device bound to ROS topics.

The device subscribes ``manip_state`` and publishes ``manip_ctrl``. State and
control are converted here and nowhere else, so the layers above only see
``ArmObservation`` and the command structures of the data layer.

The control mode is translated through a table: a mode that is not registered
is dropped with a log line instead of raising, because one bad frame must not
kill the node. Only joint mode is registered for arms at this stage; end
effector mode is not supported yet.

Attributes:
    ARM_MODE_TABLE: Internal mode string to the arm message enumeration.
    GRIP_MODE_TABLE: Internal mode string to the gripper enumeration.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from geometry_msgs.msg import Vector3 as Vector3Msg

from hex_ros_msgs.msg import (
    HexRosRoboArmCtrl,
    HexRosRoboGripCtrl,
    HexRosRoboManipCtrlStamped,
    HexRosRoboManipStateStamped,
)

from ..struct.cmd_struct import (
    ARM_CTRL_MODES,
    GRIP_CTRL_MODES,
    ArmCommand,
    GripCommand,
)
from ..struct.device_struct import ArmObservation, ArmState, GripState
from ..struct.geom_struct import Pose, Quaternion, Vector3
from .device_base import DeviceBase
from ..utils import drop

# Mode string of the command to the enumeration of the ROS message. This is
# the only place where the two vocabularies meet: the command layer and the
# wire share the same strings.
ARM_MODE_TABLE = {
    ARM_CTRL_MODES[0]: HexRosRoboArmCtrl.CTRL_MODE_JNT,
}
GRIP_MODE_TABLE = {
    GRIP_CTRL_MODES[0]: HexRosRoboGripCtrl.CTRL_MODE_JNT,
    GRIP_CTRL_MODES[1]: HexRosRoboGripCtrl.CTRL_MODE_TAU,
}

GRIP_RANGE = (-1.0, 1.0)
DEFAULT_GRAV = (0.0, 0.0, 0.0)
DEFAULT_QOS_DEPTH = 1


class ArmDevice(DeviceBase):
    """One arm with its gripper, bound to one state and one control topic.

    Attributes:
        TYPE (str): Device table type key.
        REQUIRED_KEYS (tuple): Keys the device table has to provide.
    """

    TYPE = "arm"
    REQUIRED_KEYS = ("type", "state_topic", "ctrl_topic", "grip_limit")

    def __init__(
        self,
        node: Any,
        name: str,
        params: dict,
    ) -> None:
        """Bind the subscriptions and read the device parameters.

        Args:
            node: ROS node used to create the endpoints.
            name (str): Device name taken from the device table.
            params (dict): Entry of the ``arms`` table, already validated.
        """
        self.__name = name
        # Name the model side uses; the device keeps its own name internally.
        self.model_field = str(params.get("model_field") or name)
        self.__node = node
        self.__grip_limit = (float(params["grip_limit"][0]),
                             float(params["grip_limit"][1]))
        self.__lim_vel = float(params.get("lim_vel", 0.5))
        self.__lim_acc = float(params.get("lim_acc", 50.0))
        self.__grav = tuple(float(value)
                            for value in params.get("grav", DEFAULT_GRAV))
        self.__logger = _device_logger(
            node=node,
            name=name,
        )
        self.__observation: ArmObservation | None = None
        depth = int(params.get("qos_depth", DEFAULT_QOS_DEPTH))
        self.__publisher = node.create_publisher(
            HexRosRoboManipCtrlStamped, params["ctrl_topic"], depth)
        self.__subscription = node.create_subscription(
            HexRosRoboManipStateStamped, params["state_topic"],
            self.on_states, depth)

    @property
    def name(self) -> str:
        """Name of this device as declared in the device table.

        Returns:
            str: Device name.
        """
        return self.__name

    def on_states(self, states: Any) -> None:
        """Store the newest arm and gripper state.

        Arm and gripper come from one message and therefore share the single
        outer header stamp.

        Args:
            states: ``HexRosRoboManipStateStamped`` message.
        """
        stamp = states.header.stamp
        self.__observation = ArmObservation(
            stamp_ns=stamp.sec * 1_000_000_000 + stamp.nanosec,
            arm_state=_to_arm_state(arm_state=states.manip_state.arm_state),
            grip_state=_to_grip_state(
                grip_state=states.manip_state.grip_state),
        )

    def get_obs(self) -> ArmObservation | None:
        """Return the newest observation.

        Returns:
            ArmObservation | None: Latest sample, ``None`` before the first
            message.
        """
        return self.__observation

    def send_ctrl(self, command: ArmCommand | GripCommand) -> None:
        """Send one command to this arm.

        Args:
            command (ArmCommand | GripCommand): Command to send; the type
                selects which half of the manipulation control message is
                filled.
        """
        if isinstance(command, ArmCommand):
            self.__send_arm(command=command)
        elif isinstance(command, GripCommand):
            self.__send_grip(command=command)
        else:
            drop(
                logger=self.__logger,
                point=f"arm.{self.__name}",
                obj="command dropped",
                cause=f"unknown command type {type(command).__name__}, "
                f"no branch sends it",
                reason="unsupported_mode",
            )

    def __send_arm(self, command: ArmCommand) -> None:
        """Publish the arm half of a manipulation control message.

        Args:
            command (ArmCommand): Arm command in joint mode.
        """
        mode = ARM_MODE_TABLE.get(command.ctrl_mode)
        if mode is None:
            drop(
                logger=self.__logger,
                point=f"arm.{self.__name}",
                obj="command dropped",
                cause=f"ctrl_mode={command.ctrl_mode} is not in the "
                f"mode table",
                reason="unsupported_mode",
            )
            return
        msg = HexRosRoboManipCtrlStamped()
        arm_ctrl = msg.manip_ctrl.arm_ctrl
        arm_ctrl.ctrl_mode = mode
        arm_ctrl.jnt.pos = _as_float_list(values=command.pos)
        arm_ctrl.jnt.lim_vel = [self.__lim_vel]
        arm_ctrl.jnt.lim_acc = [self.__lim_acc]
        arm_ctrl.grav = Vector3Msg(x=self.__grav[0], y=self.__grav[1],
                                  z=self.__grav[2])
        self.__publisher.publish(msg)

    def __send_grip(self, command: GripCommand) -> None:
        """Publish the gripper half of a manipulation control message.

        The model emits a normalised gripper target; the travel range comes
        from the device table, so the mapping lives here.

        Args:
            command (GripCommand): Gripper command in joint mode.
        """
        mode = GRIP_MODE_TABLE.get(command.ctrl_mode)
        if mode is None:
            drop(
                logger=self.__logger,
                point=f"grip.{self.__name}",
                obj="command dropped",
                cause=f"ctrl_mode={command.ctrl_mode} is not in the "
                f"mode table",
                reason="unsupported_mode",
            )
            return
        msg = HexRosRoboManipCtrlStamped()
        grip_ctrl = msg.manip_ctrl.grip_ctrl
        grip_ctrl.ctrl_mode = mode
        grip_ctrl.jnt.pos = [
            self.__map_grip(value=value)
            for value in _as_float_list(values=command.pos)
        ]
        self.__publisher.publish(msg)

    def __map_grip(self, value: float) -> float:
        """Map a normalised gripper target onto the travel range.

        Args:
            value (float): Target in ``[-1.0, 1.0]``.

        Returns:
            float: Position inside ``grip_limit``.
        """
        low, high = self.__grip_limit
        clipped = min(max(value, GRIP_RANGE[0]), GRIP_RANGE[1])
        span = (clipped - GRIP_RANGE[0]) / (GRIP_RANGE[1] - GRIP_RANGE[0])
        return low + span * (high - low)

    def _release(self) -> None:
        """Destroy the endpoints created by this device."""
        for destroy, handle in (
                ("destroy_subscription", self.__subscription),
                ("destroy_publisher", self.__publisher),
        ):
            method = getattr(self.__node, destroy, None)
            if handle is not None and callable(method):
                method(handle)
        self.__subscription = None
        self.__publisher = None

def _device_logger(node: Any, name: str) -> Any:
    """Pick the logger of the ROS node, falling back to the stdlib one.

    Args:
        node: ROS node or a test double.
        name (str): Device name used in the fallback logger name.

    Returns:
        Any: Logger that exposes ``log(level, message)``.
    """
    getter = getattr(node, "get_logger", None)
    if callable(getter):
        return getter()
    return logging.getLogger(f"{__name__}.{name}")


def _to_arm_state(arm_state: Any) -> ArmState:
    """Convert the arm part of a state message.

    Args:
        arm_state: ``HexRosRoboArmState`` field of the message.

    Returns:
        ArmState: Data layer structure.
    """
    jnt = arm_state.jnt
    pose = arm_state.pose
    return ArmState(
        arm_pos=np.asarray(jnt.position, dtype=np.float64),
        arm_vel=np.asarray(jnt.velocity, dtype=np.float64),
        arm_eff=np.asarray(jnt.effort, dtype=np.float64),
        ee_base=Pose(
            position=Vector3(
                x=pose.position.x,
                y=pose.position.y,
                z=pose.position.z,
            ),
            orientation=Quaternion(
                x=pose.orientation.x,
                y=pose.orientation.y,
                z=pose.orientation.z,
                w=pose.orientation.w,
            ),
        ),
    )


def _to_grip_state(grip_state: Any) -> GripState:
    """Convert the gripper part of a state message.

    Args:
        grip_state: ``HexRosRoboGripState`` field of the message.

    Returns:
        GripState: Data layer structure.
    """
    jnt = grip_state.jnt
    return GripState(
        grip_pos=np.asarray(jnt.position, dtype=np.float64),
        grip_vel=np.asarray(jnt.velocity, dtype=np.float64),
        grip_eff=np.asarray(jnt.effort, dtype=np.float64),
    )


def _as_float_list(values: Any) -> list[float]:
    """Convert a command vector to a plain float list.

    Args:
        values: Array or list of numbers.

    Returns:
        list: Values as floats.
    """
    return [float(value) for value in np.asarray(values).ravel()]
