################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

import threading
from typing import Any, Callable


class StateMachine:
    """Table driven state machine with a deferred transition request.

    ``transition_to`` only stores a target; ``step`` is the single place that
    changes the state, and it runs the task of the state it just moved to.
    ``step`` has to be called from one thread, ``transition_to`` may be called
    from any thread.

    Attributes:
        state (str): Current state; starts at the value given to the
            constructor.
    """

    def __init__(
        self,
        initial_state: str,
        logger: Any,
    ) -> None:
        """Create a machine that starts in the given state.

        Args:
            initial_state (str): State the machine starts in.
            logger (Any): Logger that provides ``logw``.
        """
        self.__lock = threading.Lock()
        self.__state_table: dict[str, Callable[[], None]] = {}
        self.__state_trans: dict[str, set[str]] = {}
        self.__state = initial_state
        self.__pending_state: str | None = None
        self.__logger = logger

    @property
    def state(self) -> str:
        """Current state.

        Returns:
            str: State name.
        """
        with self.__lock:
            return self.__state

    def register_state(
        self,
        name: str,
        callback: Callable[[], None],
    ) -> None:
        """Register one state and the task it runs every step.

        Args:
            name (str): State name.
            callback (Callable[[], None]): Task of this state; called with no
                argument once per step, its return value is ignored.

        Raises:
            TypeError: If the callback is not callable.
            ValueError: If the name is already registered.
        """
        if not callable(callback):
            raise TypeError(f"task for {name!r} is not callable")

        with self.__lock:
            if name in self.__state_table:
                raise ValueError(f"state {name!r} already registered")
            self.__state_table[name] = callback

    def register_transition(self, source: str, target: str) -> None:
        """Register one allowed transition.

        Args:
            source (str): State the transition leaves; already registered.
            target (str): State the transition reaches; already registered.

        Raises:
            ValueError: If either end was not registered.
        """
        with self.__lock:
            if source not in self.__state_table:
                raise ValueError(f"state {source!r} was not registered")
            if target not in self.__state_table:
                raise ValueError(f"state {target!r} was not registered")

            self.__state_trans.setdefault(source, set()).add(target)

    def transition_to(self, target: str) -> None:
        """Request a transition; the state changes on the next step.

        An unknown target or an unreachable one is dropped with a warning
        instead of raising, because the request may come from a task callback
        or from a signal handler, where raising has no receiver.

        Args:
            target (str): State to move to; registered and reachable from the
                current state.
        """
        with self.__lock:
            source = self.__state
            if target not in self.__state_table:
                reason = f"{target} is not registered"
            elif target not in self.__state_trans.get(source, ()):
                reason = f"{source} -> {target} is not allowed"
            else:
                self.__pending_state = target
                return
        self.__logger.logw(
            f"[state.transition/rejected] request dropped: {reason}")

    def step(self) -> None:
        """Apply the pending transition, then run the task of the state.

        The task runs outside the lock, so it may call ``transition_to`` and
        the request is applied by the next step.
        """
        with self.__lock:
            target = self.__pending_state
            self.__pending_state = None
            if target is not None and target != self.__state:
                self.__state = target
            callback = self.__state_table.get(self.__state)
        if callback is not None:
            callback()
