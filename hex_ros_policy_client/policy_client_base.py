################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Abstract policy client node.

The base class owns the seven step loop, the state and event names, and the
single exit path. A concrete client assembles its collaborators and implements
only the hooks: the clock, the condition check and the components that depend
on the device topology.

The main loop has no state branch. Which state does what per tick is decided by
the behaviours registered on the state machine, so adding a state means adding
a table entry and not an ``if``.

Attributes:
    STATE_INIT: Waiting for the first sample of every device.
    STATE_RETURN_TO_STABLE: Moving to the stable pose.
    STATE_WAITING: Holding the pose, ready for a task.
    STATE_RUNNING: Executing the task, publishing observations.
    STATE_EXIT: Terminal state; every ending converges here.
    STATES: All state names.
    EVENTS: All event names.
    REASON_EXIT: Exit reason of a model side exit request.
    REASON_INIT_TIMEOUT: Exit reason of a start up timeout.
    REASON_STABLE_TIMEOUT: Exit reason of a blocked return to stable.
    REASON_KEYBOARD_INTERRUPT: Exit reason of Ctrl+C.
    EXIT_CODE_BY_REASON: Process exit code per exit reason.
    FALLBACK_EXIT_CODE: Code used when a reason is not registered.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import rclpy

from .protocol.proto_helper import build_inference_frame
from .utils import drop

STATE_INIT = "INIT"
STATE_RETURN_TO_STABLE = "RETURN_TO_STABLE"
STATE_WAITING = "WAITING"
STATE_RUNNING = "RUNNING"
STATE_EXIT = "EXIT"

STATES = (STATE_INIT, STATE_RETURN_TO_STABLE, STATE_WAITING, STATE_RUNNING,
          STATE_EXIT)

EVENT_INIT_SUCCESS = "init_success"
EVENT_INIT_TIMEOUT = "init_timeout"
EVENT_STABLE_ARRIVED = "stable_arrived"
EVENT_STABLE_TIMEOUT = "stable_timeout"
EVENT_START = "start"
EVENT_END = "end"
EVENT_EPISODE_TIMEOUT = "episode_timeout"
EVENT_EXIT = "exit"

EVENTS = (EVENT_INIT_SUCCESS, EVENT_INIT_TIMEOUT, EVENT_STABLE_ARRIVED,
          EVENT_STABLE_TIMEOUT, EVENT_START, EVENT_END, EVENT_EPISODE_TIMEOUT,
          EVENT_EXIT)

REASON_EXIT = "exit"
REASON_INIT_TIMEOUT = "init_timeout"
REASON_STABLE_TIMEOUT = "stable_timeout"
REASON_KEYBOARD_INTERRUPT = "keyboard_interrupt"

EXIT_CODE_BY_REASON = {
    REASON_EXIT: 0,
    REASON_INIT_TIMEOUT: 1,
    REASON_STABLE_TIMEOUT: 1,
    REASON_KEYBOARD_INTERRUPT: 130,
}
FALLBACK_EXIT_CODE = 1


class PolicyClientBase(ABC):
    """Base class of the policy client nodes.

    Attributes:
        ros_node: ROS node that owns the endpoints.
        robot: Composite robot of the device side.
        state_machine: Event driven machine of this node.
        policy_middleware: Transport of the policy links.
        observation: Observation assembler, created on start.
        action_process: Action chunk processor, created on start.
        current_obs (dict): Grouped observation read in this tick.
    """

    def __init__(self, ros_node: Any, robot: Any, state_machine: Any,
                 policy_middleware: Any, logger: Any | None = None) -> None:
        """Store the topology independent collaborators.

        Args:
            ros_node: ROS node that owns the endpoints.
            robot: Composite robot of the device side.
            state_machine: Event driven machine of this node.
            policy_middleware: Transport of the policy links.
            logger (Any | None): Logger used for drop lines.
        """
        self.ros_node = ros_node
        self.robot = robot
        self.state_machine = state_machine
        self.policy_middleware = policy_middleware
        self.observation: Any | None = None
        self.action_process: Any | None = None
        self.current_obs: dict = {"camera": {}, "arm": {}}
        self.__logger = logger
        self.__started = False
        self.__shutdown_done = False
        self.__exit_reason = ""
        self.__exit_code = 0

    @abstractmethod
    def get_now_timestamp(self) -> int:
        """Return the current time of the single time source.

        Returns:
            int: Current time in nanoseconds.
        """

    @abstractmethod
    def check_conditions(self, now_ns: int) -> None:
        """Post the internal events that are due in this tick.

        Args:
            now_ns (int): Time of this tick.
        """

    @abstractmethod
    def _build_components(self, hub: Any) -> tuple:
        """Create the components that depend on the device topology.

        Args:
            hub: Device hub created by the robot on start.

        Returns:
            tuple: Observation assembler and action chunk processor.
        """

    @property
    def exit_code(self) -> int:
        """Process exit code derived from the exit reason.

        Returns:
            int: Exit code, ``0`` before the shutdown.
        """
        return self.__exit_code

    @property
    def exit_reason(self) -> str:
        """Reason of the exit request.

        Returns:
            str: Reason, empty before any request.
        """
        return self.__exit_reason

    def start(self) -> None:
        """Create the devices, the topology components and the transport.

        The call is idempotent.
        """
        if self.__started:
            return
        self.robot.start()
        self.observation, self.action_process = self._build_components(
            hub=self.robot.device_hub)
        self.policy_middleware.start()
        self.__started = True

    def stop(self) -> None:
        """Release the transport and the devices.

        The call is idempotent.
        """
        self.policy_middleware.close()
        self.robot.stop()

    def step(self) -> bool:
        """Run one tick of the seven step loop.

        Returns:
            bool: ``False`` when the terminal state is reached and the
            resources are released, which ends the main loop.
        """
        now_ns = self.get_now_timestamp()
        if self.__finished():
            return False
        self.current_obs = self.robot.read_obs()
        self.check_conditions(now_ns=now_ns)
        self.state_machine.process_events()
        self.publish_inference_state(now_ns=now_ns)
        self.state_machine.run_behavior()
        return not self.__finished()

    def publish_inference_state(self, now_ns: int) -> None:
        """Publish the inference enable flag of this tick.

        The flag follows the current state and is published every tick, because
        the observation link is silent while the robot is not running and the
        model side would otherwise never see ``False``.

        Args:
            now_ns (int): Time of this tick.
        """
        enabled = self.state_machine.state == STATE_RUNNING
        frame = build_inference_frame(enabled=enabled)
        self.policy_middleware.pub_inference_state(frame=frame)

    def request_exit(self, reason: str) -> None:
        """Request an exit without migrating or releasing anything.

        The call may come from the ROS callback thread or from a signal
        handler; the migration happens in the main loop.

        Args:
            reason (str): Reason recorded for the exit code.
        """
        if reason not in EXIT_CODE_BY_REASON:
            drop(
                logger=self.__logger,
                point="node.exit",
                obj="exit reason not registered",
                cause=f"reason={reason} is not registered, falling back to "
                    f"{FALLBACK_EXIT_CODE}",
                reason="not_ready",
            )
        self.__exit_reason = reason
        self.state_machine.post(event=EVENT_EXIT)

    def shutdown(self) -> None:
        """Release every resource once.

        The order is fixed and every step is isolated, so a failure in one of
        them does not keep the others from running.
        """
        if self.__shutdown_done:
            return
        self.__shutdown_done = True
        self.__exit_code = EXIT_CODE_BY_REASON.get(self.__exit_reason,
                                                  FALLBACK_EXIT_CODE)
        steps = (
            ("robot", self.robot.stop),
            ("recorder", self.close_recorder),
            ("middleware", self.policy_middleware.close),
            ("runtime", _shutdown_runtime),
        )
        for name, action in steps:
            try:
                action()
            except Exception as error:  # noqa: BLE001
                drop(
                    logger=self.__logger,
                    point="node.shutdown",
                    obj=f"failed to release {name}",
                    cause=f"cleanup step failed: {error}",
                    reason="not_ready",
                )

    def close_recorder(self) -> None:
        """Close the recording segment of this node.

        A node without recording keeps the empty body. The method exists so the
        one time cleanup stays a single table: stopping the robot, closing the
        recorder, closing the transport and shutting the runtime down.
        """

    def noop_behavior(self) -> None:
        """Behaviour of a state that only waits for the next event."""

    def __finished(self) -> bool:
        """Whether the terminal state was reached and released.

        Returns:
            bool: ``True`` when the main loop may end.
        """
        return (self.state_machine.state == STATE_EXIT
                and self.__shutdown_done)


def _shutdown_runtime() -> None:
    """Shut the ROS runtime down when it is still running.

    ``try_shutdown`` is used instead of ``shutdown`` so a second call, for
    example from the main loop tail, does not raise.
    """
    rclpy.try_shutdown()
