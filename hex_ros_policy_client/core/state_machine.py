################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Event driven state machine.

The machine stores three tables - event callbacks, per tick behaviours and one
change callback - and applies at most one event per tick. It knows nothing
about the business: it holds no state name of its own (the start state is an
argument) and calls callbacks without arguments, so a caller keeps its own
fields.

Events are queued instead of applied in place, because they arrive from the ROS
callback thread and from the signal handler, while migration must happen in the
main loop only.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable

from ..utils import drop


class StateMachine:
    """Table driven state machine with a single event queue.

    Attributes:
        state (str): Current state; starts at the value given to the
            constructor.
    """

    def __init__(
        self,
        start_state: str,
        logger: Any | None = None,
    ) -> None:
        """Create a machine that starts in the given state.

        Args:
            start_state (str): State the machine starts in; required, because
                the machine holds no business state name of its own.
            logger (Any | None): Logger used to report unknown events.
        """
        self.__on_event: dict[tuple[str, str], Callable] = {}
        self.__on_tick: dict[str, Callable] = {}
        self.__on_change: Callable | None = None
        self.__events: set[str] = set()
        self.__pending: deque[str] = deque()
        self.__lock = threading.Lock()
        self.__logger = logger
        self.__state = start_state

    @property
    def state(self) -> str:
        """Current state.

        Returns:
            str: State name.
        """
        return self.__state

    def create_event(self, name: str) -> None:
        """Register an event name.

        Args:
            name (str): Event name; internal events have to be registered as
                well.

        Raises:
            ValueError: If the name is already registered.
        """
        if name in self.__events:
            raise ValueError(f"event {name!r} already registered")
        self.__events.add(name)

    def register(
        self,
        event: str,
        state: str,
        callback: Callable,
    ) -> None:
        """Register an event callback for one state.

        Args:
            event (str): Event name, previously created.
            state (str): State the callback belongs to.
            callback (Callable): Called with no arguments; it returns the next
                state, or ``None`` to stay.

        Raises:
            ValueError: If the event was not created or the pair is already
                registered.
            TypeError: If the callback is not callable.
        """
        if event not in self.__events:
            raise ValueError(f"event {event!r} was not created")
        if not callable(callback):
            raise TypeError(f"callback for {event!r} is not callable")
        key = (state, event)
        if key in self.__on_event:
            raise ValueError(
                f"({state}, {event}) already registered")
        self.__on_event[key] = callback

    def register_behavior(
        self,
        state: str,
        callback: Callable,
    ) -> None:
        """Register the per tick behaviour of one state.

        Args:
            state (str): State the behaviour belongs to.
            callback (Callable): Called with no arguments every tick; its
                return value is ignored, because a behaviour never migrates.

        Raises:
            ValueError: If the state already has a behaviour.
            TypeError: If the callback is not callable.
        """
        if not callable(callback):
            raise TypeError(f"behaviour for {state!r} is not callable")
        if state in self.__on_tick:
            raise ValueError(f"behaviour for {state!r} already registered")
        self.__on_tick[state] = callback

    def register_change(self, callback: Callable) -> None:
        """Register the migration callback.

        Args:
            callback (Callable): Called as ``callback(old, new)`` on every
                migration.

        Raises:
            TypeError: If the callback is not callable.
        """
        if not callable(callback):
            raise TypeError("change callback is not callable")
        self.__on_change = callback

    def post(self, event: str) -> None:
        """Queue one event; safe from any thread.

        Args:
            event (str): Event name, previously created.

        Raises:
            ValueError: If the event was never created.
        """
        if event not in self.__events:
            raise ValueError(f"event {event!r} was not created")
        with self.__lock:
            self.__pending.append(event)

    def process_events(self) -> None:
        """Apply at most one queued event."""
        with self.__lock:
            event = self.__pending.popleft() if self.__pending else None
        if event is None:
            return
        handler = self.__on_event.get((self.__state, event))
        if handler is None:
            drop(
                logger=self.__logger,
                point="event.dispatch",
                obj="event dropped",
                cause=f"state {self.__state} has no handler for {event}, "
                f"ignored by design",
                reason="unregistered",
            )
            return
        next_state = handler()
        if next_state is not None and next_state != self.__state:
            self.__set_state(next_state=next_state)

    def run_behavior(self) -> None:
        """Run the per tick behaviour of the current state."""
        callback = self.__on_tick.get(self.__state)
        if callback is not None:
            callback()

    def pending(self) -> int:
        """Number of queued events, for tests and diagnostics.

        Returns:
            int: Queue length.
        """
        with self.__lock:
            return len(self.__pending)

    def __set_state(self, next_state: str) -> None:
        """Move to another state and run the change callback.

        Args:
            next_state (str): State to move to.
        """
        previous = self.__state
        self.__state = next_state
        if self.__on_change is not None:
            self.__on_change(previous, next_state)
