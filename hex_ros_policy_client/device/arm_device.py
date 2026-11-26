################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-24
################################################################
"""Hex arm device communicating via ROS manipulation topics."""

from __future__ import annotations

import bisect
from collections import deque
from typing import Any, Callable

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
from ..utils.math_util.interpolation import linear_interpolate, pos_lim
from .ros_device_base import RosDeviceBase

DEFAULT_QUEUE_CAPACITY = 30
DEFAULT_ARM_LIM_VEL = 0.5
DEFAULT_ARM_LIM_ACC = 50.0
DEFAULT_GRAV = (0.0, 0.0, -9.81)
DEFAULT_GRIP_LIM_POS = (0.0, 1.0)
DEFAULT_GRIP_LIM_VEL = 1.0
DEFAULT_GRIP_LIM_EFF = 1.0
GRIP_NORM_RANGE = (-1.0, 1.0)

_KEY_STATE = "state_topic"
_KEY_CTRL = "ctrl_topic"


class HexArmDevice(RosDeviceBase):
    """Arm and gripper device connected to ROS manipulation topics.

    Subscribes to ``manip_state`` and publishes to ``manip_ctrl``. Maintains
    an internal monotonic queue of states and translates joint or Cartesian
    end-effector control commands into ROS manipulation control messages.

    Attributes:
        SUBSCRIBER_SPECS (dict): Entry key to message type and handler name.
        PUBLISHER_SPECS (dict): Entry key to message type.
        REQUIRED_KEYS (tuple): Required top-level table keys.
    """

    REQUIRED_KEYS = ("ros_cfg", "arm_cfg", "grip_cfg")
    SUBSCRIBER_SPECS = {
        _KEY_STATE: (HexRosRoboManipStateStamped, "on_states"),
    }
    PUBLISHER_SPECS = {
        _KEY_CTRL: (HexRosRoboManipCtrlStamped,),
    }

    _ARM_MODE_TABLE = {
        ARM_CTRL_MODES[0]: HexRosRoboArmCtrl.CTRL_MODE_JNT,
        ARM_CTRL_MODES[1]: HexRosRoboArmCtrl.CTRL_MODE_EE,
        ARM_CTRL_MODES[2]: HexRosRoboArmCtrl.CTRL_MODE_MIT,
    }
    _GRIP_MODE_TABLE = {
        GRIP_CTRL_MODES[0]: HexRosRoboGripCtrl.CTRL_MODE_JNT,
        GRIP_CTRL_MODES[1]: HexRosRoboGripCtrl.CTRL_MODE_TAU,
    }
    _DEFAULT_ARM_ERR_LIMIT = 0.05

    def __init__(
        self,
        name: str,
        params: dict,
        create_subscription: Callable,
        create_publisher: Callable,
        logger: Any,
        now_fn: Callable[[], int] | None = None,
    ) -> None:
        """Read arm parameters and initialise endpoint subscriptions.

        Args:
            name (str): Device name taken from the device table.
            params (dict): Entry of the ``arms`` table.
            create_subscription (Callable): ``(msg_type, topic, callback)``
                creating and returning subscription handle.
            create_publisher (Callable): ``(msg_type, topic)`` creating and
                returning publisher handle.
            logger (Any): Logger object exposing standard logging methods.
            now_fn (Callable[[], int] | None): Function returning current time
                in nanoseconds; defaults to ``time.time_ns``.

        Raises:
            ValueError: If required base configuration is invalid.
        """
        arm_cfg = params.get("arm_cfg", {})
        grip_cfg = params.get("grip_cfg", {})

        raw_grip_lim = grip_cfg.get("lim_pos", DEFAULT_GRIP_LIM_POS)
        lim_pos = (float(raw_grip_lim[0]), float(raw_grip_lim[1]))
        grav_vals = arm_cfg.get("grav", DEFAULT_GRAV)
        grav = (float(grav_vals[0]), float(grav_vals[1]), float(grav_vals[2]))
        stable_pos_raw = arm_cfg.get("stable_position")
        stable_position = (
            [float(val) for val in stable_pos_raw]
            if stable_pos_raw is not None else None
        )
        kp_raw = arm_cfg.get("kp")
        kp = (
            [float(val) for val in kp_raw]
            if kp_raw is not None else None
        )
        kd_raw = arm_cfg.get("kd")
        kd = (
            [float(val) for val in kd_raw]
            if kd_raw is not None else None
        )

        self.__params: dict[str, Any] = {
            "arm": {
                "lim_vel": float(arm_cfg.get("lim_vel", DEFAULT_ARM_LIM_VEL)),
                "lim_acc": float(arm_cfg.get("lim_acc", DEFAULT_ARM_LIM_ACC)),
                "err_limit": float(
                    arm_cfg.get("err_limit", self._DEFAULT_ARM_ERR_LIMIT)),
                "grav": grav,
                "stable_position": stable_position,
                "kp": kp,
                "kd": kd,
            },
            "grip": {
                "lim_pos": lim_pos,
                "lim_vel": float(
                    grip_cfg.get("lim_vel", DEFAULT_GRIP_LIM_VEL)),
                "lim_eff": float(
                    grip_cfg.get("lim_eff", DEFAULT_GRIP_LIM_EFF)),
            },
        }
        self.__queue: deque[tuple[int, HexRosRoboManipStateStamped]] = (
            deque(maxlen=DEFAULT_QUEUE_CAPACITY)
        )
        self.__last_target: tuple[int, np.ndarray] | None = None
        self.__current_states: HexRosRoboManipStateStamped | None = None

        super().__init__(
            name=name,
            params=params,
            create_subscription=create_subscription,
            create_publisher=create_publisher,
            logger=logger,
            now_fn=now_fn,
        )
        self.__publisher = self.publishers[_KEY_CTRL]

    def on_states(self, message: HexRosRoboManipStateStamped) -> None:
        """Enqueue one raw state message maintaining monotonic timestamp order.

        Args:
            message (HexRosRoboManipStateStamped): State message from robot.
        """
        ts_ns = self.__stamp_ns(stamp=message.header.stamp)
        item = (ts_ns, message)
        self.__current_states = item

        if not self.__queue or ts_ns >= self.__queue[-1][0]:
            self.__queue.append(item)
            return
        timestamps = [stored[0] for stored in self.__queue]
        index = bisect.bisect_right(timestamps, ts_ns)
        self.__queue.insert(index, item)

    def get_obs(self, drain: bool = True) -> ArmObservation | None:
        """Decode and package the latest arm and gripper observation.

        Args:
            drain (bool): When True, retrieves the latest state and clears
                the internal queue; otherwise retains queue contents.

        Returns:
            ArmObservation | None: Latest observation with isolated array
            clones, or None if the queue is empty.
        """
        try:
            if not self.__queue:
                return None
            stamp_ns, message = self.__queue[-1]
            if drain:
                self.__queue.clear()
        except IndexError:
            return None
        return self.__to_arm_observation(message=message)

    def send_ctrl(
        self,
        arm_command: ArmCommand,
        grip_command: GripCommand | None = None,
        target_time_ns: int | None = None,
    ) -> None:
        """Publish combined arm and gripper commands in a single message.

        Resolves the control modes of the arm and gripper commands and
        publishes one ``HexRosRoboManipCtrlStamped`` message.

        Args:
            arm_command (ArmCommand): Arm joint or Cartesian target.
            grip_command (GripCommand | None): Gripper position or effort.
            target_time_ns (int | None): Target time in nanoseconds.

        Raises:
            ValueError: If ``arm_command`` is None.
        """
        if arm_command is None:
            raise ValueError(
                f"arm {self._name!r}: arm_command cannot be None")
        arm_ctrl = self.__build_arm_ctrl(
            command=arm_command,
            target_time_ns=target_time_ns,
        )
        if arm_ctrl is None:
            return
        grip_ctrl = self.__build_grip_ctrl(command=grip_command)
        msg = HexRosRoboManipCtrlStamped()
        if target_time_ns is not None:
            msg.header.stamp.sec = int(target_time_ns // 1_000_000_000)
            msg.header.stamp.nanosec = int(target_time_ns % 1_000_000_000)
        msg.manip_ctrl.arm_ctrl = arm_ctrl
        msg.manip_ctrl.grip_ctrl = grip_ctrl
        self.__publisher.publish(msg)

    def move_to_stable(
        self,
        duration_ns: int,
        arrive_threshold: float,
    ) -> bool:
        """Send the stable pose as a ramp of the given duration.

        The pose is re-sent on every call with a horizon that starts now, so
        the driver keeps ramping towards one fixed goal and this device keeps
        no per segment state.

        Args:
            duration_ns (int): Length of the ramp, in nanoseconds.
            arrive_threshold (float): Largest per joint distance to the stable
                pose that still counts as arrived, in radians.

        Returns:
            bool: ``True`` when every joint is within ``arrive_threshold``.
        """
        goal_pos = self.__stable_pos()
        if goal_pos is None:
            return True
        measured = self.__measured_pos()
        if measured is None:
            return False
        if len(measured) != len(goal_pos):
            self._logger.logw(
                f"[arm.{self._name}/stable_shape] stable pose skipped: "
                f"arm reports {len(measured)} joints but stable_position "
                f"has {len(goal_pos)}")
            return False
        self.send_ctrl(
            arm_command=ArmCommand(ctrl_mode=ARM_CTRL_MODES[0], pos=goal_pos),
            target_time_ns=self._now_fn() + duration_ns,
        )
        return float(np.linalg.norm(measured - goal_pos)) < arrive_threshold

    def hold_position(self) -> None:
        """Re-send the stable pose without a target time."""
        goal_pos = self.__stable_pos()
        if goal_pos is None:
            return
        self.send_ctrl(
            arm_command=ArmCommand(ctrl_mode=ARM_CTRL_MODES[0], pos=goal_pos))



    def __build_arm_ctrl(
        self,
        command: ArmCommand | None,
        target_time_ns: int | None = None,
    ) -> HexRosRoboArmCtrl | None:
        """Construct arm control ROS message structure from ArmCommand.

        Args:
            command (ArmCommand | None): Arm command to construct from.
            target_time_ns (int | None): Target time in nanoseconds for MIT.

        Returns:
            HexRosRoboArmCtrl | None: Built arm control message, or None.
        """
        if command is None:
            return None
        mode = self._ARM_MODE_TABLE.get(command.ctrl_mode)
        if mode is None:
            self._logger.logw(
                f"[arm.{self._name}/unsupported_mode] arm command dropped: "
                f"ctrl_mode={command.ctrl_mode!r} not in _ARM_MODE_TABLE")
            return None
        arm_params = self.__params["arm"]

        msg_arm_ctrl = HexRosRoboArmCtrl()
        msg_arm_ctrl.ctrl_mode = mode
        msg_arm_ctrl.jnt.lim_vel = [arm_params["lim_vel"]]
        msg_arm_ctrl.jnt.lim_acc = [arm_params["lim_acc"]]
        grav = arm_params["grav"]

        msg_arm_ctrl.grav = Vector3Msg(x=grav[0], y=grav[1], z=grav[2])

        if mode == HexRosRoboArmCtrl.CTRL_MODE_JNT:
            if command.pos is not None:
                msg_arm_ctrl.jnt.pos = np.asarray(
                    command.pos, dtype=float).ravel().tolist()
        elif mode == HexRosRoboArmCtrl.CTRL_MODE_EE:
            if command.pose is not None:
                pose = command.pose
                msg_arm_ctrl.pose.position.x = float(pose.position.x)
                msg_arm_ctrl.pose.position.y = float(pose.position.y)
                msg_arm_ctrl.pose.position.z = float(pose.position.z)
                msg_arm_ctrl.pose.orientation.w = float(pose.orientation.w)
                msg_arm_ctrl.pose.orientation.x = float(pose.orientation.x)
                msg_arm_ctrl.pose.orientation.y = float(pose.orientation.y)
                msg_arm_ctrl.pose.orientation.z = float(pose.orientation.z)
        elif mode == HexRosRoboArmCtrl.CTRL_MODE_MIT:
            dof = 0
            if command.pos is not None:
                pos_arr = np.asarray(command.pos, dtype=float).ravel()
                pos = self.__interpolate_mit_pos(
                    command_pos=pos_arr,
                    target_time_ns=target_time_ns,
                )
                if pos is None:
                    self._logger.logw(
                        f"[arm.{self._name}/mit_no_start] arm command dropped: "
                        "no measured state to ramp from")
                    return None

                msg_arm_ctrl.jnt.pos = pos
                dof = len(pos)
            if command.vel is not None:
                msg_arm_ctrl.jnt.vel = np.asarray(
                    command.vel, dtype=float).ravel().tolist()
            elif dof > 0:
                msg_arm_ctrl.jnt.vel = [0.0] * dof
            if command.eff is not None:
                msg_arm_ctrl.jnt.eff = np.asarray(
                    command.eff, dtype=float).ravel().tolist()
            elif dof > 0:
                msg_arm_ctrl.jnt.eff = [0.0] * dof
            kp = (
                np.asarray(command.kp, dtype=float).ravel().tolist()
                if command.kp is not None else arm_params["kp"]
            )
            kd = (
                np.asarray(command.kd, dtype=float).ravel().tolist()
                if command.kd is not None else arm_params["kd"]
            )
            if kp is not None:
                msg_arm_ctrl.jnt.kp = kp
            if kd is not None:
                msg_arm_ctrl.jnt.kd = kd
        return msg_arm_ctrl

    def __interpolate_mit_pos(
        self,
        command_pos: np.ndarray,
        target_time_ns: int | None,
    ) -> list[float] | None:
        """Blend the command with the previous target at the current instant.

        The command being dispatched is the end of the segment and the stored
        target the start, so only one target has to be kept. Before any
        command has been sent, the newest measured state of the queue is the
        start, which ramps the arm out of where it actually is.

        Args:
            command_pos (np.ndarray): Target position of this step.
            target_time_ns (int | None): Time the target belongs to, in
                nanoseconds.

        Returns:
            list[float]: Position to publish; the command itself when it has
            no target time, or when neither a target nor a measured state is
            known.
        """
        if target_time_ns is None:
            return command_pos.ravel().tolist()
        last_target = self.__last_target

        if last_target is None:
            if not self.__current_states:
                return None
            stamp_ns, message = self.__current_states
            last_target = (stamp_ns, 
                           np.asarray(
                                message.manip_state.arm_state.jnt.position,
                                dtype=float).ravel())

        last_target_time_ns, last_target_pos = last_target

        blended = linear_interpolate(
            start=last_target_pos,
            target=command_pos,
            elapsed_ns=self._now_fn() - last_target_time_ns,
            duration_ns=target_time_ns - last_target_time_ns,
        )

        # # limit the blended position
        # measured = self.__current_states
        # if measured is None:
        #     return None
        # stamp_ns, message = measured
        # blended = self.__limit_mit_pos(
        #     blended=blended,
        #     measured_pos=np.asarray(
        #         message.manip_state.arm_state.jnt.position,
        #         dtype=float).ravel(),
        #     travel_ns=self._now_fn() - stamp_ns,
        # )


        self.__last_target = (target_time_ns, command_pos.copy())
        return blended.ravel().tolist()

    def __limit_mit_pos(
        self,
        blended: np.ndarray,
        measured_pos: np.ndarray,
        travel_ns: int,
    ) -> np.ndarray:
        """Cap how far a command may sit from the measured position.

        The cap is the validated constant, relaxed by ``lim_vel`` once the
        measurement is old enough that a larger step is legitimate. It never
        drops below the constant, so the stiffness of MIT mode is kept.

        Args:
            blended (np.ndarray): Position the blend asks for.
            measured_pos (np.ndarray): Newest measured joint positions.
            travel_ns (int): Time since that measurement, in nanoseconds.

        Returns:
            np.ndarray: Position to send, at most ``err_limit`` away from
            ``measured_pos``.
        """
        err_limit = max(
            0.05,
            self.__params["arm"]["lim_vel"]
            * max(travel_ns, 0) / 1_000_000_000,
        )
        return pos_lim(
            target=blended,
            curr=measured_pos,
            threshold=err_limit,
        )

    def __build_grip_ctrl(
        self,
        command: GripCommand | None,
    ) -> HexRosRoboGripCtrl:
        """Construct gripper control ROS message structure from GripCommand.

        Args:
            command (GripCommand | None): Gripper command to construct from.

        Returns:
            HexRosRoboGripCtrl: Built gripper control message.
        """
        msg_grip_ctrl = HexRosRoboGripCtrl()
        if command is None:
            msg_grip_ctrl.ctrl_mode = HexRosRoboGripCtrl.CTRL_MODE_NONE
            return msg_grip_ctrl
        mode = self._GRIP_MODE_TABLE.get(command.ctrl_mode)
        if mode is None:
            self._logger.logw(
                f"[grip.{self._name}/unsupported_mode] grip command dropped: "
                f"ctrl_mode={command.ctrl_mode!r} not in _GRIP_MODE_TABLE")
            msg_grip_ctrl.ctrl_mode = HexRosRoboGripCtrl.CTRL_MODE_NONE
            return msg_grip_ctrl
        grip_params = self.__params["grip"]
        msg_grip_ctrl.ctrl_mode = mode
        msg_grip_ctrl.jnt.lim_vel = [grip_params["lim_vel"]]
        if mode == HexRosRoboGripCtrl.CTRL_MODE_JNT:
            msg_grip_ctrl.jnt.eff = [grip_params["lim_eff"]]
            if command.pos is not None:
                msg_grip_ctrl.jnt.pos = [
                    self.__denormalize_grip_position(normalized_position=val)
                    for val in np.asarray(command.pos, dtype=float).ravel()
                ]
        elif mode == HexRosRoboGripCtrl.CTRL_MODE_TAU:
            if command.eff is not None:
                msg_grip_ctrl.jnt.eff = np.asarray(
                    command.eff, dtype=float).ravel().tolist()
        return msg_grip_ctrl

    def __to_arm_observation(
        self,
        message: HexRosRoboManipStateStamped,
    ) -> ArmObservation:
        """Convert a ROS manipulation state message into an ArmObservation.

        Args:
            message (HexRosRoboManipStateStamped): Incoming ROS state message.

        Returns:
            ArmObservation: Data layer observation with cloned arrays.
        """
        manip_state = message.manip_state
        arm_state_msg = manip_state.arm_state
        grip_state_msg = manip_state.grip_state

        arm_jnt = arm_state_msg.jnt
        arm_pose = arm_state_msg.pose

        arm_state = ArmState(
            arm_pos=np.array(arm_jnt.position, dtype=np.float64, copy=True),
            arm_vel=np.array(arm_jnt.velocity, dtype=np.float64, copy=True),
            arm_eff=np.array(arm_jnt.effort, dtype=np.float64, copy=True),
            ee_base=Pose(
                position=Vector3(
                    x=float(arm_pose.position.x),
                    y=float(arm_pose.position.y),
                    z=float(arm_pose.position.z),
                ),
                orientation=Quaternion(
                    w=float(arm_pose.orientation.w),
                    x=float(arm_pose.orientation.x),
                    y=float(arm_pose.orientation.y),
                    z=float(arm_pose.orientation.z),
                ),
            ),
        )

        grip_jnt = grip_state_msg.jnt
        grip_state = GripState(
            grip_pos=np.array(grip_jnt.position, dtype=np.float64, copy=True),
            grip_vel=np.array(grip_jnt.velocity, dtype=np.float64, copy=True),
            grip_eff=np.array(grip_jnt.effort, dtype=np.float64, copy=True),
        )

        return ArmObservation(
            stamp_ns=self.__stamp_ns(stamp=message.header.stamp),
            arm_state=arm_state,
            grip_state=grip_state,
        )

    def __denormalize_grip_position(
        self,
        normalized_position: float,
    ) -> float:
        """Denormalise a [-1.0, 1.0] gripper target to physical joint position.

        A normalized value of 1.0 denotes fully closed, corresponding to the
        upper limit of ``grip_limit`` (grip_limit[1]), while -1.0 denotes
        fully open at the lower limit (grip_limit[0]).

        Args:
            normalized_position (float): Target value in ``[-1.0, 1.0]``.

        Returns:
            float: Physical position within ``grip_limit``.
        """
        low, high = self.__params["grip"]["lim_pos"]
        clipped = min(
            max(normalized_position, GRIP_NORM_RANGE[0]),
            GRIP_NORM_RANGE[1],
        )
        span = (clipped - GRIP_NORM_RANGE[0]) / (
            GRIP_NORM_RANGE[1] - GRIP_NORM_RANGE[0]
        )
        return low + span * (high - low)

    def __stable_pos(self) -> np.ndarray | None:
        """Return the configured stable pose of this arm.

        Returns:
            np.ndarray | None: Joint positions in radians, ``None`` when the
            device table declares no stable pose.
        """
        goal = self.__params["arm"]["stable_position"]
        if goal is None:
            self._logger.logw(
                f"[arm.{self._name}/stable_not_ready] stable pose skipped: "
                "stable_position is missing from the device table")
            return None
        return np.asarray(goal, dtype=np.float64).ravel()

    def __measured_pos(self) -> np.ndarray | None:
        """Return the newest measured joint positions.

        Returns:
            np.ndarray | None: Joint positions in radians, ``None`` before the
            first state message.
        """
        if self.__current_states is None:
            return None
        return np.asarray(
            self.__current_states[1].manip_state.arm_state.jnt.position,
            dtype=np.float64).ravel().copy()

    @staticmethod
    def __stamp_ns(stamp: Any) -> int:
        """Turn a ROS header stamp into integer nanoseconds.

        Args:
            stamp: ``builtin_interfaces/Time`` structure.

        Returns:
            int: Time in nanoseconds.
        """
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


