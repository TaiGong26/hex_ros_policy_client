################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Exhaustive checks of ``StateMachine``.

One block per behaviour: registration guards, the deferred request, the
transition relation, the graph walks and the lock. The transition block is
brute force on purpose: the machine holds nothing but a table, so the table is
swept over every ordered pair of states, every simple path of the graph and a
seeded random walk, instead of a few hand picked cases.
"""

from __future__ import annotations

import random
import threading

import pytest

from hex_ros_policy_client.core.state_machine import StateMachine

from .fakes import Capture

STATES = ("S0", "S1", "S2", "S3", "S4", "S5")
EDGES = (
    ("S0", "S1"),
    ("S0", "S2"),
    ("S1", "S3"),
    ("S2", "S3"),
    ("S3", "S4"),
    ("S4", "S5"),
    ("S5", "S0"),
)
REJECT_PREFIX = "[state.transition/rejected]"
THREAD_TIMEOUT_S = 10.0
MONKEY_OPS = 3000
STEADY_STEPS = 2000
REQUESTERS = 6
SEED = 20261125


def _task(calls: list[str], name: str):
    """Build a task that records the state it was called from.

    Args:
        calls (list[str]): List the task appends to.
        name (str): State name the task belongs to.

    Returns:
        Callable[[], None]: Task of that state.
    """
    def task() -> None:
        calls.append(name)
    return task


def build(
    initial_state: str = "S0",
) -> tuple[StateMachine, Capture, list[str]]:
    """Build a machine on the shared graph and record the task calls.

    Args:
        initial_state (str): State the machine starts in.

    Returns:
        tuple: Machine, log capture and the list of the task calls.
    """
    log = Capture()
    calls: list[str] = []
    machine = StateMachine(initial_state=initial_state, logger=log)
    for name in STATES:
        machine.register_state(name, _task(calls, name))
    for source, target in EDGES:
        machine.register_transition(source, target)
    return machine, log, calls


def rejections(log: Capture) -> list[str]:
    """Return the rejection lines of one machine.

    Args:
        log (Capture): Logger double of one machine.

    Returns:
        list[str]: Rejection messages, in emission order.
    """
    return [text for text in log.messages() if text.startswith(REJECT_PREFIX)]


def simple_paths(start: str) -> list[tuple[str, ...]]:
    """Enumerate every simple path that starts at the given state.

    Args:
        start (str): State the paths start at.

    Returns:
        list[tuple[str, ...]]: Paths of two states or more, no repeated state.
    """
    targets: dict[str, list[str]] = {}
    for source, target in EDGES:
        targets.setdefault(source, []).append(target)
    paths: list[tuple[str, ...]] = []

    def walk(path: tuple[str, ...]) -> None:
        if len(path) > 1:
            paths.append(path)
        for nxt in targets.get(path[-1], ()):
            if nxt not in path:
                walk(path + (nxt,))

    walk((start,))
    return paths


# ── 1 registration ────────────────────────────────────────────


def test_a_state_name_cannot_be_registered_twice() -> None:
    machine, _, _ = build()
    with pytest.raises(ValueError):
        machine.register_state("S0", lambda: None)


def test_a_task_has_to_be_callable() -> None:
    machine, _, _ = build()
    with pytest.raises(TypeError):
        machine.register_state("NEW", "not callable")


def test_a_transition_needs_a_registered_source() -> None:
    machine, _, _ = build()
    with pytest.raises(ValueError):
        machine.register_transition("NOPE", "S1")


def test_a_transition_needs_a_registered_target() -> None:
    machine, _, _ = build()
    with pytest.raises(ValueError):
        machine.register_transition("S1", "NOPE")


def test_a_repeated_edge_is_accepted_once() -> None:
    machine, log, calls = build()
    machine.register_transition("S0", "S1")
    machine.transition_to("S1")
    machine.step()
    assert machine.state == "S1"
    assert calls == ["S1"]
    assert rejections(log) == []


def test_an_unregistered_initial_state_still_steps() -> None:
    log = Capture()
    machine = StateMachine(initial_state="ALONE", logger=log)
    machine.register_state("S1", lambda: None)
    assert machine.state == "ALONE"
    machine.step()
    machine.transition_to("S1")
    machine.step()
    assert machine.state == "ALONE"
    assert len(rejections(log)) == 1


# ── 2 deferred request ────────────────────────────────────────


def test_a_request_does_not_change_the_state() -> None:
    machine, _, calls = build()
    machine.transition_to("S1")
    assert machine.state == "S0"
    assert calls == []


def test_one_step_applies_the_request_once() -> None:
    machine, _, calls = build()
    machine.transition_to("S1")
    machine.step()
    assert machine.state == "S1"
    assert calls == ["S1"]
    machine.step()
    assert machine.state == "S1"
    assert calls == ["S1", "S1"]


def test_a_step_runs_the_task_of_the_target_state() -> None:
    machine, _, calls = build()
    machine.transition_to("S2")
    machine.step()
    assert calls == ["S2"]


def test_a_step_without_a_request_runs_the_current_task() -> None:
    machine, _, calls = build()
    machine.step()
    assert machine.state == "S0"
    assert calls == ["S0"]


def test_the_last_request_wins() -> None:
    machine, log, calls = build()
    machine.transition_to("S1")
    machine.transition_to("S2")
    machine.step()
    assert machine.state == "S2"
    assert calls == ["S2"]
    assert rejections(log) == []


def test_a_request_from_a_task_is_applied_by_the_next_step() -> None:
    log = Capture()
    machine = StateMachine(initial_state="A", logger=log)
    seen: list[str] = []

    def task_a() -> None:
        machine.transition_to("B")

    def task_b() -> None:
        seen.append(machine.state)
        machine.transition_to("A")

    machine.register_state("A", task_a)
    machine.register_state("B", task_b)
    machine.register_transition("A", "B")
    machine.register_transition("B", "A")
    machine.step()
    assert machine.state == "A"
    assert seen == []
    machine.step()
    assert machine.state == "B"
    assert seen == ["B"]
    machine.step()
    assert machine.state == "A"
    assert seen == ["B"]


# ── 3 the transition relation, brute force ────────────────────


def test_every_ordered_pair_of_states_is_checked() -> None:
    allowed = set(EDGES)
    checked = 0
    for source in STATES:
        for target in STATES:
            machine, log, calls = build(initial_state=source)
            machine.transition_to(target)
            machine.step()
            if (source, target) in allowed:
                assert machine.state == target, (source, target)
                assert calls == [target], (source, target)
                assert rejections(log) == [], (source, target)
            else:
                assert machine.state == source, (source, target)
                assert calls == [source], (source, target)
                assert len(rejections(log)) == 1, (source, target)
            checked += 1
    assert checked == len(STATES) ** 2


def test_every_simple_path_is_followed() -> None:
    paths = simple_paths("S0")
    assert len(paths) == 8, len(paths)
    for path in paths:
        machine, log, calls = build()
        for target in path[1:]:
            machine.transition_to(target)
            machine.step()
        assert machine.state == path[-1], path
        assert calls == list(path[1:]), path
        assert rejections(log) == [], path


def test_a_random_walk_never_leaves_the_edge_set() -> None:
    machine, _, calls = build()
    allowed = set(EDGES)
    rng = random.Random(SEED)
    moved: list[tuple[str, str]] = []
    steps = 0
    for _ in range(MONKEY_OPS):
        if rng.random() < 0.5:
            machine.transition_to(rng.choice(STATES))
            continue
        before = machine.state
        machine.step()
        steps += 1
        after = machine.state
        assert after in STATES
        if after != before:
            moved.append((before, after))
    assert steps > MONKEY_OPS // 4
    assert len(calls) == steps
    for edge in moved:
        assert edge in allowed, edge


def test_a_target_that_is_not_a_state_is_rejected() -> None:
    machine, log, calls = build()
    foreign = ("", " ", "S0 ", "s0", "S6", "STATE", "S0\n")
    for target in foreign:
        machine.transition_to(target)
    machine.step()
    assert machine.state == "S0"
    assert calls == ["S0"]
    assert len(rejections(log)) == len(foreign)


def test_a_registered_self_loop_keeps_the_state() -> None:
    machine, log, calls = build()
    machine.register_transition("S0", "S0")
    machine.transition_to("S0")
    machine.step()
    assert machine.state == "S0"
    assert calls == ["S0"]
    assert rejections(log) == []


def test_an_unregistered_self_loop_is_rejected() -> None:
    machine, log, calls = build()
    machine.transition_to("S0")
    machine.step()
    assert machine.state == "S0"
    assert calls == ["S0"]
    assert len(rejections(log)) == 1


# ── 4 the lock ────────────────────────────────────────────────


def test_the_state_can_be_read_inside_a_task() -> None:
    log = Capture()
    machine = StateMachine(initial_state="A", logger=log)
    seen: list[str] = []
    machine.register_state("A", lambda: seen.append(machine.state))
    machine.register_state("B", lambda: None)
    machine.register_transition("A", "B")
    machine.step()
    assert seen == ["A"]
    machine.transition_to("B")
    machine.step()
    assert seen == ["A"]
    assert machine.state == "B"


def test_many_threads_may_request_at_once() -> None:
    machine, _, calls = build()
    threads = [
        threading.Thread(target=machine.transition_to, args=(target,))
        for target in ("S1", "S2") * 20
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=THREAD_TIMEOUT_S)
    assert not any(thread.is_alive() for thread in threads)
    machine.step()
    assert machine.state in ("S1", "S2")
    assert calls == [machine.state]


def test_a_stepper_next_to_many_requesters_stays_legal() -> None:
    machine, _, calls = build()
    allowed = set(EDGES)
    moved: list[tuple[str, str]] = []
    stop = threading.Event()

    def requester() -> None:
        rng = random.Random(threading.get_ident())
        while not stop.is_set():
            machine.transition_to(rng.choice(STATES))

    threads = [threading.Thread(target=requester) for _ in range(REQUESTERS)]
    for thread in threads:
        thread.start()
    try:
        for _ in range(STEADY_STEPS):
            before = machine.state
            machine.step()
            after = machine.state
            assert after in STATES
            if after != before:
                moved.append((before, after))
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=THREAD_TIMEOUT_S)
    assert not any(thread.is_alive() for thread in threads)
    assert len(calls) == STEADY_STEPS
    for edge in moved:
        assert edge in allowed, edge


# ── 5 the rejection message ───────────────────────────────────


def test_a_rejection_names_both_ends() -> None:
    machine, log, _ = build()
    machine.transition_to("S5")
    messages = rejections(log)
    assert len(messages) == 1
    assert "S0" in messages[0]
    assert "S5" in messages[0]


def test_a_rejection_of_an_unknown_state_names_it() -> None:
    machine, log, _ = build()
    machine.transition_to("S6")
    messages = rejections(log)
    assert len(messages) == 1
    assert "S6" in messages[0]
    assert "is not registered" in messages[0]
