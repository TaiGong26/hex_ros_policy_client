################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the policy client base class."""

from __future__ import annotations

import pytest

from hex_ros_policy_client.core.state_machine import StateMachine
from hex_ros_policy_client import policy_client_base as base
from hex_ros_policy_client.policy_client_base import (
    EVENT_EXIT,
    EVENT_INIT_SUCCESS,
    EVENTS,
    EXIT_CODE_BY_REASON,
    STATE_EXIT,
    STATE_INIT,
    STATE_RUNNING,
    STATE_WAITING,
    STATES,
    PolicyClientBase,
)

from .fakes import Capture, FakeRosNode


class FakeRobot:
    """Robot double that records its calls.

    Attributes:
        log (list): Shared call log.
        started (int): Number of start calls.
        stopped (int): Number of stop calls.
    """

    def __init__(self, log: list) -> None:
        """Create a robot bound to a shared log.

        Args:
            log (list): Shared call log.
        """
        self.log = log
        self.started = 0
        self.stopped = 0
        self.hub = object()

    def start(self) -> None:
        """Count one start."""
        self.started += 1

    def stop(self) -> None:
        """Count one stop."""
        self.stopped += 1

    def read_obs(self) -> dict:
        """Record the observation read.

        Returns:
            dict: Empty grouped observation.
        """
        self.log.append("read_obs")
        return {"camera": {}, "arm": {}}

    def send_ctrl(self, target: dict) -> None:
        """Record one dispatch.

        Args:
            target (dict): Target to send.
        """
        self.log.append("send_ctrl")

    @property
    def device_hub(self):
        """Device hub handed to the component builder.

        Returns:
            Any: Stand-in hub.
        """
        return self.hub


class FakeMiddleware:
    """Transport double that records its calls.

    Attributes:
        log (list): Shared call log.
        opened (int): Number of open calls.
        closed (int): Number of close calls.
        published (list): Published frames.
    """

    def __init__(self, log: list) -> None:
        """Create a transport bound to a shared log.

        Args:
            log (list): Shared call log.
        """
        self.log = log
        self.opened = 0
        self.closed = 0
        self.published: list = []

    def start(self) -> None:
        """Count one start."""
        self.opened += 1

    def close(self) -> None:
        """Count one close."""
        self.closed += 1

    def pub_obs(self, frame: dict) -> bool:
        """Record one observation frame.

        Args:
            frame (dict): Frame to publish.

        Returns:
            bool: Always true.
        """
        self.log.append("pub_obs")
        self.published.append(frame)
        return True

    def pub_inference_state(self, frame: dict) -> bool:
        """Record one enable frame.

        Args:
            frame (dict): Frame that matches ``InferenceFrame``.

        Returns:
            bool: Always true.
        """
        self.log.append(("inference", frame["inference_enabled"]))
        return True

    def request_chunk(self):
        """Return no chunk.

        Returns:
            None: Always empty.
        """
        return None


class FakeObservation:
    """Observation double that records its calls.

    Attributes:
        log (list): Shared call log.
        pushed (int): Number of pushes.
    """

    def __init__(self, log: list) -> None:
        """Create an assembler bound to a shared log.

        Args:
            log (list): Shared call log.
        """
        self.log = log
        self.pushed = 0
        self.sources = {}

    def push_obs(self, observation: dict) -> None:
        """Count one push.

        Args:
            observation (dict): Grouped observation.
        """
        self.pushed += 1

    def get_obs(self) -> dict:
        """Return a frame with a fixed stamp.

        Returns:
            dict: Frame with empty groups.
        """
        return {"stamp_ns": 1, "camera": {}, "arm": {}}


class FakeAction:
    """Action processor double."""

    def reset(self) -> None:
        """Do nothing."""

    def push_chunk(self, chunk: dict) -> None:
        """Ignore one chunk.

        Args:
            chunk (dict): Chunk received.
        """

    def get_target(self):
        """Return no target.

        Returns:
            None: Always empty.
        """
        return None


class RecordingClient(PolicyClientBase):
    """Concrete client with hooks that record their calls."""

    def __init__(self, log: list) -> None:
        """Assemble the doubles.

        Args:
            log (list): Shared call log.
        """
        self.log = log
        ros_node = FakeRosNode(now_ns=100)
        machine = StateMachine(STATE_INIT, Capture())
        self.middleware = FakeMiddleware(log)
        super().__init__(ros_node, FakeRobot(log), machine, self.middleware,
                         Capture())
        self.__setup()

    def __setup(self) -> None:
        """Register a minimal table."""
        machine = self.state_machine
        machine.create_event(EVENT_INIT_SUCCESS)
        machine.create_event(EVENT_EXIT)
        machine.register(state=STATE_INIT, event=EVENT_INIT_SUCCESS,
                         callback=self.__on_ready)
        machine.register(state=STATE_INIT, event=EVENT_EXIT,
                         callback=self.__on_exit)
        machine.register(state=STATE_WAITING, event=EVENT_EXIT,
                         callback=self.__on_exit)
        machine.register(state=STATE_EXIT, event=EVENT_EXIT,
                         callback=self.__on_exit)
        machine.register_behavior(STATE_WAITING, self.__waiting)
        machine.register_change(self.__on_change)
        self.__ready = True

    def __on_ready(self) -> str:
        """Record the ready callback.

        Returns:
            str: Next state.
        """
        self.log.append("on_init_success")
        return STATE_WAITING

    def __on_exit(self) -> str:
        """Record the exit callback.

        Returns:
            str: Next state.
        """
        self.log.append("on_exit")
        return STATE_EXIT

    def __waiting(self) -> None:
        """Record one waiting tick."""
        self.log.append("behavior_waiting")

    def __on_change(self, old: str, new: str) -> None:
        """Release everything when the terminal state is entered.

        Args:
            old (str): Previous state.
            new (str): New state.
        """
        if new == STATE_EXIT:
            self.shutdown()

    def get_now_timestamp(self) -> int:
        """Return the fake clock.

        Returns:
            int: Current time.
        """
        return self.ros_node.clock.now_ns

    def check_conditions(self, now_ns: int) -> None:
        """Post the ready event while still in the initial state.

        Args:
            now_ns (int): Time of the tick.
        """
        self.log.append("check_conditions")
        if self.state_machine.state == STATE_INIT and self.__ready:
            self.__ready = False
            self.state_machine.post(EVENT_INIT_SUCCESS)

    def _build_components(self, hub) -> tuple:
        """Create the doubles of the topology dependent components.

        Args:
            hub: Device hub of the robot.

        Returns:
            tuple: Observation and action doubles.
        """
        return FakeObservation(self.log), FakeAction()


def test_base_is_abstract() -> None:
    """The base class cannot be instantiated on its own."""
    with pytest.raises(TypeError):
        PolicyClientBase(None, None, None, None)


def test_constant_tables() -> None:
    """The state and event sets and the exit codes are complete."""
    assert len(STATES) == 5
    assert len(EVENTS) == 8
    assert len(set(STATES)) == 5
    assert len(set(EVENTS)) == 8
    assert EXIT_CODE_BY_REASON["exit"] == 0
    assert EXIT_CODE_BY_REASON["keyboard_interrupt"] == 130
    assert EXIT_CODE_BY_REASON["init_timeout"] == 1
    assert EXIT_CODE_BY_REASON["stable_timeout"] == 1


def test_start_is_idempotent() -> None:
    """Starting twice creates the devices and the session once."""
    client = RecordingClient([])
    client.start()
    client.start()
    assert client.robot.started == 1
    assert client.middleware.opened == 1
    assert client.observation is not None


def test_step_follows_the_seven_step_order() -> None:
    """The tick reads, checks, migrates, publishes and then behaves."""
    log: list = []
    client = RecordingClient(log)
    client.start()
    client.step()
    assert log == ["read_obs", "check_conditions", "on_init_success",
                   ("inference", False), "behavior_waiting"]


def test_inference_state_follows_the_state() -> None:
    """The enable flag equals whether the state is running."""
    log: list = []
    client = RecordingClient(log)
    client.start()
    client.step()
    assert ("inference", False) in log
    client.state_machine.post(EVENT_EXIT)
    client.step()
    assert client.state_machine.state == STATE_EXIT


def test_request_exit_does_not_migrate() -> None:
    """The request only queues an event."""
    client = RecordingClient([])
    client.start()
    client.request_exit("exit")
    assert client.state_machine.state == STATE_INIT
    client.state_machine.process_events()
    assert client.state_machine.state == STATE_EXIT
    assert client.exit_reason == "exit"


def test_terminal_state_ends_the_loop() -> None:
    """The tick returns false once the resources are released."""
    client = RecordingClient([])
    client.start()
    client.request_exit("exit")
    assert client.step() is False
    assert client.exit_code == 0


def test_shutdown_is_idempotent() -> None:
    """Releasing twice closes the robot and the session once."""
    client = RecordingClient([])
    client.start()
    client.request_exit("exit")
    client.step()
    robot = client.robot
    middleware = client.middleware
    client.shutdown()
    assert robot.stopped == 1
    assert middleware.closed == 1


def test_unregistered_reason_falls_back() -> None:
    """An unknown reason still releases the resources."""
    client = RecordingClient([])
    client.start()
    client.request_exit("something_odd")
    client.step()
    assert client.exit_code == base.FALLBACK_EXIT_CODE


def test_running_state_enables_inference() -> None:
    """The flag is true only in the running state."""
    client = RecordingClient([])
    client.state_machine.post(EVENT_INIT_SUCCESS)
    client.start()
    client.step()
    client.state_machine.register_behavior(STATE_RUNNING, client.noop_behavior)
    client.state_machine._StateMachine__state = STATE_RUNNING
    log: list = []
    client.middleware.log = log
    client.publish_inference_state(1)
    assert log == [("inference", True)]
