################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from typing import Any

import numpy as np

from .core.dual_manip_robot import (
    SECTION_ARMS,
    SECTION_CAMERAS,
    DualManipRobot,
)
from .core.state_machine import StateMachine
from .process.observation.observation_history import HistoryObservation
from .process.observation.observation_base import (
    SYNC_FIXED_STEP,
    SYNC_STRATEGIES,
    source_stamp,
)
from .process.action.action_window_query import (
    ActionWindowQuery,
    WindowQueryParam,
)
from .protocol.proto_helper import (
    build_inference_frame,
    build_observation_frame,
    decode_action_chunk,
)
from .protocol.policy_middleware import PolicyMiddleware
from .struct.cmd_struct import (
    ArmCommand,
    ControlIntent,
)
from .utils import STAGE_LIMITED, STAGE_RAW, CsvRecorder, DataInterface
from .utils import linear_interpolate, pos_lim
NS_PER_MS = 1_000_000

STATE_INIT = "INIT"
STATE_RETURN_TO_STABLE = "RETURN_TO_STABLE"
STATE_WAITING = "WAITING"
STATE_RUNNING = "RUNNING"
STATE_EXIT = "EXIT"

REASON_EXIT = "exit"
REASON_INIT_TIMEOUT = "init_timeout"
REASON_STABLE_TIMEOUT = "stable_timeout"
REASON_KEYBOARD_INTERRUPT = "keyboard_interrupt"




class DualManipPolicyClient():
    """Policy client assembled for manipulators plus cameras."""

    _NS_PER_S = 1_000_000_000
    _MIDDLEWARE_QUEUE_MAX = 100
    
    _EXIT_CODE_BY_REASON = {
        REASON_EXIT: 0,
        REASON_INIT_TIMEOUT: 1,
        REASON_STABLE_TIMEOUT: 1,
        REASON_KEYBOARD_INTERRUPT: 130,
    }


    def __init__(
        self,
    ) -> None:
        """Read the parameters and assemble the collaborators.

        Args:
            ros_interface: Interface that owns the node and carries the log
                lines.

        Raises:
            RuntimeError: If a parameter is missing or inconsistent.
        """
       
        try:
            self.__ros_interface =  DataInterface(name="dual_manip_policy_client")
            self.__logger =  self.__ros_interface.getlogger()
            self.__params = self.__ros_interface.get_client_param()
            self.__stable_order = tuple(self.__params["stable_arm_order"])
            self.__observation: Any | None = None
            self.__action_process: Any | None = None
            self.__current_obs: dict = {"camera": {}, "arm": {}}
            self.__started = False
            self.__shutdown_done = False
            self.__state_machine = StateMachine(
                initial_state=STATE_INIT,
                logger=self.__logger,
            )
            self.__policy_middleware = PolicyMiddleware(
                observation_topic=self.__params["observation_topic"],
                action_topic=self.__params["action_topic"],
                inference_topic=self.__params["inference_topic"],
                capacity=self._MIDDLEWARE_QUEUE_MAX,
                logger=self.__logger,
            )
            self.__robot = DualManipRobot(
                device_path=self.__params["device_path"],
                logger=self.__ros_interface.getlogger(),
                create_subscription=self.__ros_interface.create_subscription,
                create_publisher=self.__ros_interface.create_publisher,
            )
            self.__recorder: CsvRecorder | None = None
            
        except Exception as error:  # noqa: BLE001
            if self.__logger is not None:
                self.__logger.logw(
                    f"[node.init/not_ready] failed to initialize: {error}")
            else:
                print(f"[node.init/not_ready] failed to initialize: {error}")


    def __init__module(self) -> None:
        pass


    def start(self) -> None:
        """Create the devices, the topology components and the transport.

        The call is idempotent.
        """
        if self.__started:
            return
        self.__robot.start()
        self.__observation, self.__action_process = self._build_components(
            robot=self.__robot)
        self.__policy_middleware.start()
        self.__started = True

    def __step(self) -> None:
        """Run one tick of the main loop.
        """
        now_ns = self.get_now_timestamp()
        self.__current_obs = self.__robot.read_obs()
        self.publish_inference_state(now_ns=now_ns)

    def run(self) -> None:
        """Run the main loop until the terminal state is reached."""
        while self.__ros_interface.ok():
            self.__step()
            self.__ros_interface.sleep()

    
    def publish_inference_state(self, now_ns: int) -> None:
        """Publish the inference enable flag of this tick.

        The flag follows the current state and is published every tick, because
        the observation link is silent while the robot is not running and the
        model side would otherwise never see ``False``.

        Args:
            now_ns (int): Time of this tick.
        """
        enabled = self.__state_machine.state == STATE_RUNNING
        frame = build_inference_frame(enabled=enabled)
        self.__policy_middleware.pub_inference_state(frame=frame)


    def shutdown(self) -> None:
        """Release every resource once.

        The order is fixed and every step is isolated, so a failure in one of
        them does not keep the others from running.
        """
        if self.__shutdown_done:
            return
        self.__shutdown_done = True
        steps = (
            ("robot", self.__robot.stop),
            ("recorder", self.close_recorder),
            ("middleware", self.__policy_middleware.close),
            ("interface", self.__ros_interface.shutdown),
        )
        for name, action in steps:
            try:
                action()
            except Exception as error:  # noqa: BLE001
                self.__logger.logw(
                    f"[node.shutdown/not_ready] failed to release {name}: "
                    f"cleanup step failed: {error}")


    def close_recorder(self) -> None:
        """Close the recording files of the current segment.

        Called by the one time cleanup on entering the terminal state, and
        also when the task ends; closing twice is safe.
        """
        if self.__recorder is not None:
            self.__recorder.close()

    def get_now_timestamp(self) -> int:
        """Return the ROS clock of the node.

        Returns:
            int: Current ROS time in nanoseconds.
        """
        stamp = self.__ros_interface.get_nowstamp()
        return int(stamp.sec) * self._NS_PER_S + int(stamp.nanosec)

    def _build_components(self, robot: Any) -> tuple:
        """Create the observation assembler and the action processor.

        Args:
            robot: Dual manip robot whose devices were created on start.

        Returns:
            tuple: Observation assembler and action chunk processor.

        Raises:
            RuntimeError: If the stable arm order does not match the table.
        """
        names = robot.device_names[SECTION_ARMS]
        if set(self.__stable_order) != set(names):
            raise RuntimeError(
                f"stable_arm_order {list(self.__stable_order)} does not "
                f"match the arms of the device table {list(names)}")

        observation = HistoryObservation(
            camera_names=robot.device_names[SECTION_CAMERAS],
            arm_names=names,
            freshness_ns=int(self.__params["data_freshness_ms"] * NS_PER_MS),
            period_ns=int(self._NS_PER_S
                          / self.__params["observation_rate_hz"]),
            now_fn=self.get_now_timestamp,
            sync_fn=SYNC_STRATEGIES[SYNC_FIXED_STEP],
            logger=self.__logger,
        )
        action_process = ActionWindowQuery(params=WindowQueryParam(
            arm_names=names,
            now_fn=self.get_now_timestamp,
            anchor_tolerance_ns=int(
                self.__params["data_freshness_ms"] * NS_PER_MS),
            logger=self.__logger,
        ))
        if self.__params["recorder_dir"]:
            self.__recorder = CsvRecorder(
                directory=self.__params["recorder_dir"],
                arm_names=names,
                arm_dof=self.__params["arm_dof"],
            )
        return observation, action_process


def main() :
    """Run the node until it reaches the terminal state.

    Returns:
        int: Process exit code.
    """
    client = DualManipPolicyClient()
    client.start()
    try:
        client.run()
    except KeyboardInterrupt:
        pass
    finally:
        client.shutdown()

