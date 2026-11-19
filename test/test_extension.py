################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Extension checks: a new topology must not need a code change.

Two cases are covered. First, a single arm table drives the existing node to
the waiting state without touching any module. Second, a new node is added as
one file that only replaces the behaviour table and the condition check, which
is what the design promises for a different topology.
"""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.core.state_machine import StateMachine
from hex_ros_policy_client.policy_client_base import (
    EVENT_END,
    EVENT_EXIT,
    EVENT_INIT_SUCCESS,
    EVENT_START,
    EVENT_STABLE_ARRIVED,
    EVENTS,
    STATE_EXIT,
    STATE_INIT,
    STATE_RETURN_TO_STABLE,
    STATE_RUNNING,
    STATE_WAITING,
    PolicyClientBase,
)

from .fakes import Capture, FakeRosNode
from .test_device_hub import SINGLE_TABLE
from .test_dual_node import SINGLE_PARAMS, build_client, goto, stub_devices

pytest.importorskip("hex_ros_msgs.msg")
pytest.importorskip("sensor_msgs.msg")


class MinimalClient(PolicyClientBase):
    """Example of a node that only replaces the tables.

    A new topology is expected to look like this: a table, a condition check
    and nothing else. The seven step loop, the exit path and the state machine
    come from the base class.
    """

    def __init__(self, ros_node, table: dict) -> None:
        """Assemble the base class with an empty robot.

        Args:
            ros_node: ROS node that owns the endpoints.
            table (dict): Behaviour table to register.
        """
        self.log: list = []
        machine = StateMachine(STATE_INIT, Capture())
        super().__init__(ros_node, _NullRobot(), machine, _NullMiddleware(),
                         Capture())
        for name in EVENTS:
            machine.create_event(name)
        machine.register(state=STATE_INIT, event=EVENT_INIT_SUCCESS,
                         callback=lambda: STATE_RUNNING)
        machine.register(state=STATE_RUNNING, event=EVENT_END,
                         callback=lambda: STATE_EXIT)
        machine.register(state=STATE_RUNNING, event=EVENT_EXIT,
                         callback=lambda: STATE_EXIT)
        for state, callback in table.items():
            machine.register_behavior(state=state, callback=callback)
        machine.register_change(self.__on_change)

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
        """Post the ready event on the first tick.

        Args:
            now_ns (int): Time of the tick.
        """
        if self.state_machine.state == STATE_INIT:
            self.state_machine.post(EVENT_INIT_SUCCESS)

    def _build_components(self, hub) -> tuple:
        """Create no topology components.

        Args:
            hub: Device hub, unused by this example.

        Returns:
            tuple: Two ``None`` values.
        """
        return None, None


class _NullRobot:
    """Robot double that does nothing."""

    device_hub = None

    def start(self) -> None:
        """Do nothing."""

    def stop(self) -> None:
        """Do nothing."""

    def read_obs(self) -> dict:
        """Return an empty observation.

        Returns:
            dict: Empty groups.
        """
        return {"camera": {}, "arm": {}}

    def send_ctrl(self, target: dict) -> None:
        """Ignore a target.

        Args:
            target (dict): Target to send.
        """


class _NullMiddleware:
    """Transport double that does nothing."""

    def start(self) -> None:
        """Do nothing."""

    def close(self) -> None:
        """Do nothing."""

    def pub_obs(self, frame: dict) -> bool:
        """Accept one frame.

        Args:
            frame (dict): Frame to publish.

        Returns:
            bool: Always true.
        """
        return True

    def pub_inference_state(self, frame: dict) -> bool:
        """Accept one enable frame.

        Args:
            frame (dict): Frame that matches ``InferenceFrame``.

        Returns:
            bool: Always true.
        """
        return True

    def request_chunk(self):
        """Return no chunk.

        Returns:
            None: Always empty.
        """
        return None


def test_single_arm_table_reaches_waiting(wire) -> None:
    """One arm runs the whole start up without a code change."""
    client, node = build_client(None, overrides=SINGLE_PARAMS,
                                table=SINGLE_TABLE)
    client.start()
    stub_devices(client)
    assert client.state_machine.state == STATE_INIT
    client.step()
    assert client.state_machine.state == STATE_RETURN_TO_STABLE
    node.clock.now_ns += 3_000_000_000
    client.step()
    assert client.state_machine.state == STATE_WAITING


def test_new_node_needs_one_file() -> None:
    """A new node is a table plus a condition check."""
    calls: list = []
    client = MinimalClient(FakeRosNode(now_ns=1), {
        STATE_RUNNING: lambda: calls.append("running"),
    })
    client.start()
    assert client.step() is True
    assert calls == ["running"]
    client.request_exit("exit")
    assert client.step() is False
    assert client.state_machine.state == STATE_EXIT


def test_new_node_keeps_the_base_loop() -> None:
    """The example does not override the seven step loop."""
    assert MinimalClient.step is PolicyClientBase.step
    assert MinimalClient.start is PolicyClientBase.start
    assert MinimalClient.shutdown is PolicyClientBase.shutdown
