################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the state machine."""

from __future__ import annotations

import pathlib
import re
import threading

import pytest

from hex_ros_policy_client.core import state_machine as machine_module
from hex_ros_policy_client.core.state_machine import StateMachine
from hex_ros_policy_client.utils.common_util.logkit import (
    _reset_state as reset_log,
)

from .fakes import Capture

STATE_A = "A"
STATE_B = "B"
STATE_C = "C"
EVENT_GO = "go"
EVENT_BACK = "back"
EVENT_TICK = "tick"


def build(logger=None) -> StateMachine:
    """Create a machine with a small table.

    Args:
        logger (Capture | None): Logger used to report unknown events.

    Returns:
        StateMachine: Machine starting in ``STATE_A``.
    """
    machine = StateMachine(STATE_A, logger or Capture())
    machine.create_event(EVENT_GO)
    machine.create_event(EVENT_BACK)
    machine.register(state=STATE_A, event=EVENT_GO,
                     callback=lambda: STATE_B)
    machine.register(state=STATE_B, event=EVENT_BACK,
                     callback=lambda: STATE_A)
    return machine


def test_start_state_is_required_argument() -> None:
    """The machine holds no business state of its own."""
    with pytest.raises(TypeError):
        StateMachine()


def test_event_callback_migrates() -> None:
    """The return value of an event callback is the next state."""
    machine = build()
    machine.post(EVENT_GO)
    machine.process_events()
    assert machine.state == STATE_B


def test_none_return_keeps_the_state() -> None:
    """A callback that returns ``None`` does not migrate."""
    machine = build()
    machine.create_event(EVENT_TICK)
    machine.register(state=STATE_A, event=EVENT_TICK, callback=lambda: None)
    machine.post(EVENT_TICK)
    machine.process_events()
    assert machine.state == STATE_A


def test_behavior_never_migrates() -> None:
    """A behaviour returning a state name is ignored."""
    machine = build()
    machine.register_behavior(STATE_A, lambda: STATE_C)
    machine.run_behavior()
    assert machine.state == STATE_A


def test_one_event_per_tick() -> None:
    """At most one queued event is applied per call."""
    machine = build()
    machine.post(EVENT_GO)
    machine.post(EVENT_BACK)
    machine.process_events()
    assert machine.state == STATE_B
    assert machine.pending() == 1
    machine.process_events()
    assert machine.state == STATE_A


def test_unregistered_pair_is_ignored_with_a_warning() -> None:
    """An event without a callback keeps the state and is reported."""
    reset_log()
    logger = Capture()
    machine = build(logger)
    machine.post(EVENT_BACK)
    machine.process_events()
    assert machine.state == STATE_A
    assert logger.messages()[0].startswith("[event.dispatch/unregistered] ")


def test_change_callback_receives_both_states() -> None:
    """The change callback carries the old and the new state."""
    seen = []
    machine = build()
    machine.register_change(lambda old, new: seen.append((old, new)))
    machine.post(EVENT_GO)
    machine.process_events()
    assert seen == [(STATE_A, STATE_B)]


def test_unknown_event_is_refused_at_post() -> None:
    """Posting an event that was never created is a programming error."""
    machine = build()
    with pytest.raises(ValueError):
        machine.post("nope")


def test_register_requires_created_event() -> None:
    """A callback for an unknown event name is refused."""
    machine = build()
    with pytest.raises(ValueError):
        machine.register(state=STATE_A, event="nope", callback=lambda: None)


def test_duplicate_registration_is_refused() -> None:
    """Two callbacks for one pair would be ambiguous."""
    machine = build()
    with pytest.raises(ValueError):
        machine.register(state=STATE_A, event=EVENT_GO,
                         callback=lambda: STATE_C)


def test_post_is_thread_safe() -> None:
    """Events from another thread land in the queue."""
    machine = build()
    threads = [threading.Thread(target=machine.post, args=(EVENT_GO,))
               for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert machine.pending() == 16


def test_module_holds_no_business_state_name() -> None:
    """The module source mentions no business state."""
    path = pathlib.Path(machine_module.__file__)
    source = path.read_text(encoding="utf-8")
    for name in ("INIT", "RUNNING", "WAITING", "RETURN_TO_STABLE", "EXIT"):
        assert not re.search(rf"\b{name}\b", source)


def test_module_imports_no_business_module() -> None:
    """The machine depends on the logger only."""
    path = pathlib.Path(machine_module.__file__)
    source = path.read_text(encoding="utf-8")
    imports = re.findall(r"^from .+ import|^import .+", source, re.MULTILINE)
    joined = " ".join(imports)
    for forbidden in ("device", "robot", "observation", "action", "policy"):
        assert forbidden not in joined
