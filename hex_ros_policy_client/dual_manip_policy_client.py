################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Dual manipulator policy client.

The node declares its parameters, builds the collaborators from the device
table and registers the state machine tables. Everything else - the seven step
loop, the transition rules, the exit path - comes from the base class.

Every state and event is registered by walking a table, so adding a state or an
event means editing a table and not adding a line of assembly code. Device
names never appear here: they come from the device table.

Attributes:
    PARAM_NAMES: Every parameter this node declares.
    EVENT_MAP: Kind received from the model side to the event name.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Any, Callable

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from .core.dual_manip_robot import DualManipRobot
from .core.state_machine import StateMachine
from .process.history_observation import HistoryObservation
from .process.observation_base import (
    SYNC_FIXED_STEP,
    SYNC_STRATEGIES,
    source_stamp,
)
from .process.time_integration_chunk import MERGE_STRATEGIES
from .process.time_integration_chunk import TimeIntegrationChunk
from .protocol.proto_helper import (
    build_observation_frame,
    decode_action_chunk,
)
from .protocol.policy_middleware import PolicyMiddleware
from .struct.cmd_struct import (
    TARGET_ARM,
    TARGET_GRIP,
    ArmCommand,
    GripCommand,
)
from .utils import STAGE_LIMITED, STAGE_RAW, CsvRecorder
from .utils import drop
from .utils import get_interpolator, limit_to_step
from .policy_client_base import (
    EVENT_END,
    EVENT_EPISODE_TIMEOUT,
    EVENT_EXIT,
    EVENT_INIT_SUCCESS,
    EVENT_INIT_TIMEOUT,
    EVENT_START,
    EVENT_STABLE_ARRIVED,
    EVENT_STABLE_TIMEOUT,
    EVENTS,
    STATE_EXIT,
    STATE_INIT,
    STATE_RETURN_TO_STABLE,
    STATE_RUNNING,
    STATE_WAITING,
    REASON_EXIT,
    REASON_INIT_TIMEOUT,
    REASON_KEYBOARD_INTERRUPT,
    REASON_STABLE_TIMEOUT,
    PolicyClientBase,
)

PARAM_NAMES = (
    "control_rate_hz",
    "observation_rate_hz",
    "arm_dof",
    "stable_arm_order",
    "stable_jnt",
    "stable_duration_s",
    "stable_timeout_s",
    "arrive_threshold",
    "init_timeout_s",
    "arm_jnt_threshold",
    "arm_pos_threshold",
    "max_episode_s",
    "data_freshness_ms",
    "action_step_ms",
    "observation_topic",
    "inference_topic",
    "action_topic",
    "interpolation",
    "device_path",
    "action_queue_capacity",
    "weight_decay",
    "action_average",
    "recorder_dir",
)

# Wire name of an event to the event of this package. Written out rather than
# zipped from a tuple, so a reordered tuple cannot shift the mapping.
EVENT_MAP = {
    "start": EVENT_START,
    "end": EVENT_END,
    "exit": EVENT_EXIT,
}

NS_PER_S = 1_000_000_000
JOIN_TIMEOUT_S = 5.0
NS_PER_MS = 1_000_000
GRIP_HOLD_DEFAULT = 0.0


class DualManipPolicyClient(PolicyClientBase):
    """Policy client assembled for manipulators plus cameras."""

    def __init__(
        self,
        ros_node: Any,
        params: dict | None = None,
    ) -> None:
        """Declare the parameters and assemble the collaborators.

        Args:
            ros_node: ROS node that owns the endpoints.
            params (dict | None): Already validated parameters; read from the
                node when omitted, which is what the entry point does.

        Raises:
            RuntimeError: If a parameter is missing or inconsistent.
        """
        values = params if params is not None else read_params(node=ros_node)
        values = validate_params(values=values)
        self.__params = values
        self.__logger = _node_logger(node=ros_node)
        self.__interpolator = get_interpolator(name=values["interpolation"])
        self.__stable_order = tuple(values["stable_arm_order"])
        self.__stable_target: dict[str, np.ndarray] = {}
        self.__stable_start: dict[str, np.ndarray] | None = None
        self.__hold_target: dict | None = None
        self.__state_since_ns = 0
        self.__last_obs_ns = 0
        self.__obs_count = 0
        self.__obs_dropped = 0
        self.__ctrl_dropped = 0
        self.__recorder: CsvRecorder | None = None
        self.__grip_hold: dict[str, float] = {}
        self.__chunk_serial = 0
        self.__name_map: dict[str, str] = {}
        self.__chunk_dropped = 0
        machine = StateMachine(
            start_state=STATE_INIT,
            logger=self.__logger,
        )
        middleware = PolicyMiddleware(
            observation_topic=values["observation_topic"],
            action_topic=values["action_topic"],
            inference_topic=values["inference_topic"],
            capacity=values["action_queue_capacity"],
            on_event=self.__on_policy_message,
            logger=self.__logger,
        )
        robot = DualManipRobot(
            node=ros_node,
            device_path=values["device_path"],
        )
        super().__init__(
            ros_node=ros_node,
            robot=robot,
            state_machine=machine,
            policy_middleware=middleware,
            logger=self.__logger,
        )
        self.__setup_state_machine()

    @property
    def recorder(self) -> CsvRecorder | None:
        """Recorder of the current segment, if recording is enabled.

        Returns:
            CsvRecorder | None: Active recorder.
        """
        return self.__recorder

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
        return self.ros_node.get_clock().now().nanoseconds

    def _build_components(self, hub: Any) -> tuple:
        """Create the observation assembler and the action processor.

        Args:
            hub: Device hub created by the robot on start.

        Returns:
            tuple: Observation assembler and action chunk processor.

        Raises:
            RuntimeError: If the stable arm order does not match the table.
        """
        names = tuple(hub.arm_names)
        if set(self.__stable_order) != set(names):
            raise RuntimeError(
                f"stable_arm_order {list(self.__stable_order)} does not "
                f"match the arms of the device table {list(names)}")
        self.__stable_target = split_stable_jnt(
            order=self.__stable_order,
            flat=self.__params["stable_jnt"],
            arm_dof=self.__params["arm_dof"],
        )
        values = self.__params
        # The device table owns the mapping. It is applied where the wire frame
        # is built and read, so the assembler and the action processor keep
        # working on device names and the transport only sees dictionaries.
        self.__name_map = _field_map(hub=hub)
        observation = HistoryObservation(
            camera_names=hub.camera_names,
            arm_names=names,
            freshness_ns=int(values["data_freshness_ms"] * NS_PER_MS),
            period_ns=int(NS_PER_S / values["observation_rate_hz"]),
            now_fn=self.get_now_timestamp,
            sync_fn=SYNC_STRATEGIES[SYNC_FIXED_STEP],
            logger=self.__logger,
        )
        action_process = TimeIntegrationChunk(
            arm_names=names,
            now_fn=self.get_now_timestamp,
            interpolator=self.__interpolator,
            capacity=values["action_queue_capacity"],
            step_ms_default=values["action_step_ms"],
            average_strategy=values["action_average"],
            decay=values["weight_decay"],
            logger=self.__logger,
        )
        if values["recorder_dir"]:
            self.__recorder = CsvRecorder(
                directory=values["recorder_dir"],
                arm_names=names,
                arm_dof=values["arm_dof"],
            )
        return observation, action_process

    def check_conditions(self, now_ns: int) -> None:
        """Post the internal events that are due in this tick.

        Args:
            now_ns (int): Time of this tick.
        """
        state = self.state_machine.state
        elapsed_ns = now_ns - self.__state_since_ns
        if state == STATE_INIT:
            if self.__devices_ready():
                self.state_machine.post(event=EVENT_INIT_SUCCESS)
            elif elapsed_ns >= self.__seconds(name="init_timeout_s"):
                self.state_machine.post(event=EVENT_INIT_TIMEOUT)
        elif state == STATE_RETURN_TO_STABLE:
            arrived = self.__stable_arrived()
            long_enough = elapsed_ns >= self.__seconds(
                name="stable_duration_s")
            if arrived and long_enough:
                self.state_machine.post(event=EVENT_STABLE_ARRIVED)
            elif elapsed_ns >= self.__seconds(name="stable_timeout_s"):
                self.state_machine.post(event=EVENT_STABLE_TIMEOUT)
        elif state == STATE_RUNNING:
            limit_ns = self.__seconds(name="max_episode_s")
            if limit_ns > 0 and elapsed_ns >= limit_ns:
                self.state_machine.post(event=EVENT_EPISODE_TIMEOUT)

    def __setup_state_machine(self) -> None:
        """Register events, event callbacks, behaviours and the change hook."""
        machine = self.state_machine
        for name in EVENTS:
            machine.create_event(name=name)
        for (state, event), callback in self.__event_table().items():
            machine.register(event=event, state=state, callback=callback)
        for state, callback in self.__behavior_table().items():
            machine.register_behavior(state=state, callback=callback)
        machine.register_change(callback=self.__on_state_change)

    def __event_table(self) -> dict[tuple[str, str], Callable]:
        """Build the transition table.

        Returns:
            dict: ``(state, event)`` to the callback that returns the next
            state.
        """
        return {
            (STATE_INIT, EVENT_INIT_SUCCESS): self.__on_init_success,
            (STATE_INIT, EVENT_INIT_TIMEOUT): self.__on_init_timeout,
            (STATE_INIT, EVENT_EXIT): self.__on_exit,
            (STATE_RETURN_TO_STABLE,
             EVENT_STABLE_ARRIVED): self.__on_stable_arrived,
            (STATE_RETURN_TO_STABLE,
             EVENT_STABLE_TIMEOUT): self.__on_stable_timeout,
            (STATE_RETURN_TO_STABLE, EVENT_EXIT): self.__on_exit,
            (STATE_WAITING, EVENT_START): self.__on_start,
            (STATE_WAITING, EVENT_EXIT): self.__on_exit,
            (STATE_RUNNING, EVENT_END): self.__on_end,
            (STATE_RUNNING, EVENT_EPISODE_TIMEOUT): self.__on_episode_timeout,
            (STATE_RUNNING, EVENT_EXIT): self.__on_exit,
        }

    def __behavior_table(self) -> dict[str, Callable]:
        """Build the per tick behaviour table.

        Returns:
            dict: State to the behaviour of that state.
        """
        return {
            STATE_RETURN_TO_STABLE: self.__stable_behavior,
            STATE_WAITING: self.__waiting_behavior,
            STATE_RUNNING: self.__running_behavior,
            STATE_EXIT: self.noop_behavior,
        }

    def __on_policy_message(self, message: dict) -> None:
        """Turn a message of the model side into an event.

        Args:
            message (dict): Decoded frame with a ``kind``.
        """
        event = EVENT_MAP.get(message.get("kind"))
        if event is None:
            drop(
                logger=self.__logger,
                point="event.map",
                obj="event dropped",
                cause=f"kind={message.get('kind')} is not in the "
                f"mapping table",
                reason="unsupported_kind",
            )
            return
        if event == EVENT_EXIT:
            self.request_exit(reason=REASON_EXIT)
            return
        self.state_machine.post(event=event)

    def __on_init_success(self) -> str:
        """All devices reported at least once.

        Returns:
            str: Next state.
        """
        return STATE_RETURN_TO_STABLE

    def __on_init_timeout(self) -> str:
        """A device did not report in time.

        Returns:
            str: Next state.
        """
        self.request_exit(reason=REASON_INIT_TIMEOUT)
        return STATE_EXIT

    def __on_stable_arrived(self) -> str:
        """The stable pose was reached and held long enough.

        Returns:
            str: Next state.
        """
        return STATE_WAITING

    def __on_stable_timeout(self) -> str:
        """The stable pose was not reached in time.

        Returns:
            str: Next state.
        """
        self.request_exit(reason=REASON_STABLE_TIMEOUT)
        return STATE_EXIT

    def __on_start(self) -> str:
        """The model side starts a task.

        Returns:
            str: Next state.
        """
        self.action_process.reset()
        self.__drain_chunks()
        return STATE_RUNNING

    def __on_end(self) -> str:
        """The model side ends the task.

        Returns:
            str: Next state.
        """
        return STATE_RETURN_TO_STABLE

    def __on_episode_timeout(self) -> str:
        """The task ran longer than allowed.

        Returns:
            str: Next state.
        """
        return STATE_RETURN_TO_STABLE

    def __on_exit(self) -> str:
        """Every non terminal state exits through the same edge.

        Returns:
            str: Next state.
        """
        return STATE_EXIT

    def __on_state_change(
        self,
        old: str,
        new: str,
    ) -> None:
        """Run the leave and enter actions of one migration.

        Args:
            old (str): Previous state.
            new (str): New state.
        """
        self.__note_state(
            old=old,
            new=new,
        )
        leave = self.__leave_actions().get(old)
        if leave is not None:
            leave()
        self.__stable_start = None
        self.__state_since_ns = self.get_now_timestamp()
        enter = self.__enter_actions().get(new)
        if enter is not None:
            enter()

    def __leave_actions(self) -> dict[str, Callable]:
        """Build the leave action table.

        Returns:
            dict: State to the action run when leaving it.
        """
        return {STATE_RUNNING: self.__leave_running}

    def __enter_actions(self) -> dict[str, Callable]:
        """Build the enter action table.

        Returns:
            dict: State to the action run when entering it.
        """
        return {
            STATE_RETURN_TO_STABLE: self.__enter_stable,
            STATE_WAITING: self.__enter_waiting,
            STATE_RUNNING: self.__enter_running,
            STATE_EXIT: self.__enter_exit_guard,
        }

    def __note_state(
        self,
        old: str,
        new: str,
    ) -> None:
        """Record a migration.

        Args:
            old (str): Previous state.
            new (str): New state.
        """
        if self.__recorder is not None:
            self.__recorder.record_state(
                now_ns=self.get_now_timestamp(),
                kind="change",
                old=old,
                new=new,
            )
        self.__logger.info(f"state {old} -> {new}")

    def __enter_stable(self) -> None:
        """Clear the task data before moving to the stable pose."""
        self.action_process.reset()

    def __enter_waiting(self) -> None:
        """Snapshot the pose that the robot holds from now on."""
        self.__hold_target = self.__snapshot_hold(observation=self.current_obs)

    def __enter_running(self) -> None:
        """Clear the task data and start the recording segment.

        The decimation counter starts at zero, so the first observation of the
        segment leaves after one full window; the age reference is reset to the
        entry instant so that window is not reported as a stale observation.
        """
        self.action_process.reset()
        self.__obs_count = 0
        self.__last_obs_ns = self.get_now_timestamp()
        if self.__recorder is not None:
            self.__recorder.start()

    def __leave_running(self) -> None:
        """Close the recording segment."""
        self.close_recorder()

    def __enter_exit_guard(self) -> None:
        """Release every resource once, when the terminal state is entered."""
        self.shutdown()

    def __stable_behavior(self) -> None:
        """Interpolate from the measured pose to the stable target."""
        now_ns = self.get_now_timestamp()
        if self.__stable_start is None:
            self.__stable_start = self.__snapshot_jnt(
                observation=self.current_obs)
        if not self.__stable_start:
            return
        elapsed_ns = now_ns - self.__state_since_ns
        duration_ns = self.__seconds(name="stable_duration_s")
        target: dict = {TARGET_ARM: {}}
        for name, goal in self.__stable_target.items():
            start = self.__stable_start.get(name)
            if start is None:
                continue
            target[TARGET_ARM][name] = ArmCommand(
                pos=self.__interpolator(start, goal, elapsed_ns, duration_ns))
        if target[TARGET_ARM]:
            self.robot.send_ctrl(target=target)

    def __waiting_behavior(self) -> None:
        """Re-send the pose snapshotted when the waiting state was entered."""
        if self.__hold_target is not None:
            self.robot.send_ctrl(target=self.__hold_target)

    def __running_behavior(self) -> None:
        """Publish observations on the decimated ticks and dispatch targets."""
        self.__drain_chunks()
        now_ns = self.get_now_timestamp()
        self.__obs_count += 1
        if self.__obs_count >= self.__obs_divider():
            self.__obs_count = 0
            self.__publish_observation(now_ns=now_ns)
        age_ns = now_ns - self.__last_obs_ns
        limit_ns = self.__millis(name="data_freshness_ms")
        if age_ns >= limit_ns:
            self.__ctrl_dropped += 1
            drop(
                logger=self.__logger,
                point="ctrl.dispatch",
                obj="no command sent this tick",
                cause=f"observation is stale: {age_ns / NS_PER_MS:.1f}ms > "
                 f"{limit_ns / NS_PER_MS:.1f}ms",
                reason="obs_stale",
                count=self.__ctrl_dropped,
            )
            return
        target = self.action_process.get_target()
        if target is None:
            return
        self.__record_weights(now_ns=now_ns)
        self.__record_target(
            now_ns=now_ns,
            stage=STAGE_RAW,
            target=target,
        )
        limited = self.__limit_step(target=target)
        self.__remember_grip(target=limited)
        self.__record_target(
            now_ns=now_ns,
            stage=STAGE_LIMITED,
            target=limited,
        )
        self.robot.send_ctrl(target=limited)

    def __publish_observation(self, now_ns: int) -> None:
        """Assemble and publish one observation frame.

        Args:
            now_ns (int): Time of this tick.
        """
        self.observation.push_obs(observation=self.current_obs)
        frame = self.observation.get_obs()
        if frame is None:
            self.__obs_dropped += 1
            drop(
                logger=self.__logger,
                point="obs.publish",
                obj="observation not published",
                cause="get_obs() returned None, see obs.assemble for the per "
                 "source reason",
                reason="no_frame",
                level=logging.DEBUG,
                count=self.__obs_dropped,
            )
            return
        wire = build_observation_frame(
            observation=frame,
            name_map=self.__name_map,
        )
        if self.policy_middleware.pub_obs(frame=wire):
            self.__last_obs_ns = now_ns

    def __drain_chunks(self) -> None:
        """Interpret every queued frame and hand it to the action processor.

        The transport queues raw frames, so the meaning of a frame is decided
        here, once, before the processor sees it.
        """
        while True:
            frame = self.policy_middleware.request_chunk()
            if frame is None:
                return
            try:
                chunk = decode_action_chunk(frame=frame)
            except ValueError as error:
                self.__chunk_dropped += 1
                drop(
                    logger=self.__logger,
                    point="chunk.decode",
                    obj="action frame dropped",
                    cause=f"action semantics rejected: {error}",
                    reason="schema_invalid",
                    count=self.__chunk_dropped,
                )
                continue
            self.action_process.push_chunk(chunk=chunk)
            self.__record_chunk(chunk=chunk)

    def __limit_step(self, target: dict) -> dict:
        """Limit the step of every arm command.

        Args:
            target (dict): Target of this tick.

        Returns:
            dict: New target, the argument is left untouched.
        """
        threshold = self.__params["arm_jnt_threshold"]
        limited: dict = {TARGET_ARM: {}, TARGET_GRIP: dict(
            target.get(TARGET_GRIP, {}))}
        for name, command in target.get(TARGET_ARM, {}).items():
            measured = self.__measured_jnt(name=name)
            if measured is None:
                drop(
                    logger=self.__logger,
                    point="ctrl.dispatch",
                    obj="no arm command sent this tick",
                    cause=f"device {name} has no measured joints this "
                    f"tick, cannot limit",
                    reason="measured_missing",
                )
                continue
            limited[TARGET_ARM][name] = ArmCommand(
                ctrl_mode=command.ctrl_mode,
                pos=limit_to_step(
                    target=command.pos,
                    measured=measured,
                    threshold=threshold,
                ))
        return limited

    def __measured_jnt(self, name: str) -> np.ndarray | None:
        """Read the measured joint positions of one arm.

        Args:
            name (str): Device name.

        Returns:
            np.ndarray | None: Measured positions of this tick.
        """
        observation = self.current_obs.get("arm", {}).get(name)
        if observation is None or observation.arm_state is None:
            return None
        return observation.arm_state.arm_pos

    def __devices_ready(self) -> bool:
        """Whether every required source reported at least once.

        Returns:
            bool: ``True`` when no source is missing.
        """
        for kind, name in self.observation.sources.values():
            if source_stamp(
                observation=self.current_obs,
                kind=kind,
                name=name,
            ) is None:
                return False
        return True

    def __stable_arrived(self) -> bool:
        """Whether every arm is inside the arrival threshold.

        Returns:
            bool: ``True`` when all arms arrived.
        """
        threshold = self.__params["arrive_threshold"]
        for name, goal in self.__stable_target.items():
            measured = self.__measured_jnt(name=name)
            if measured is None:
                return False
            if float(np.linalg.norm(measured - goal)) > threshold:
                return False
        return True

    def __snapshot_jnt(self, observation: dict) -> dict[str, np.ndarray]:
        """Snapshot the measured joints of every arm.

        Args:
            observation (dict): Grouped observation of this tick.

        Returns:
            dict: Arm name to measured joint positions.
        """
        snapshot = {}
        for name, item in observation.get("arm", {}).items():
            if item is not None and item.arm_state is not None:
                snapshot[name] = np.asarray(item.arm_state.arm_pos,
                                            dtype=np.float64)
        return snapshot

    def __snapshot_hold(self, observation: dict) -> dict | None:
        """Snapshot the pose the robot holds while waiting.

        Args:
            observation (dict): Grouped observation of this tick.

        Returns:
            dict | None: Target to re-send, ``None`` when no arm is known.
        """
        joints = self.__snapshot_jnt(observation=observation)
        if not joints:
            return None
        target: dict = {TARGET_ARM: {}}
        for name, positions in joints.items():
            target[TARGET_ARM][name] = ArmCommand(pos=positions)
        grip = self.__last_grip_command()
        if grip:
            target[TARGET_GRIP] = grip
        return target

    def __last_grip_command(self) -> dict:
        """Return the gripper commands to hold during the waiting state.

        The gripper target is kept in model units: converting the measured
        value back would need the travel range, which only the device layer
        knows. Holding the last commanded value keeps the gripper where the
        model left it.

        Returns:
            dict: Device name to gripper command.
        """
        return {name: GripCommand(pos=[value])
                for name, value in self.__grip_hold.items()}

    def __remember_grip(self, target: dict) -> None:
        """Remember the gripper target that was just dispatched.

        Args:
            target (dict): Target sent in this tick.
        """
        for name, command in target.get(TARGET_GRIP, {}).items():
            values = getattr(command, "pos", None)
            if values:
                self.__grip_hold[name] = float(np.asarray(values).ravel()[0])

    def __record_chunk(self, chunk: dict) -> None:
        """Record the steps of one decoded chunk.

        Args:
            chunk (dict): Chunk as delivered by the transport.
        """
        if self.__recorder is None:
            return
        step_ms = (chunk.get("chunk_step_interval_ms")
                   or self.__params["action_step_ms"])
        arms = chunk.get("arm", {})
        grips = chunk.get("grip", {})
        steps = max((len(values) for values in arms.values()), default=0)
        self.__chunk_serial += 1
        for index in range(steps):
            arm_row = {name: np.asarray(values)[index].tolist()
                       for name, values in arms.items() if len(values) > index}
            grip_row = {name: np.asarray(values)[index].tolist()
                        for name, values in grips.items()
                        if len(values) > index}
            self.__recorder.record_chunk(
                seq=self.__chunk_serial,
                stamp_ns=chunk["stamp_ns"],
                step_ms=step_ms,
                step_index=index,
                arm_row=arm_row,
                grip_row=grip_row,
            )

    def __record_weights(self, now_ns: int) -> None:
        """Record the integration weights of this tick.

        Args:
            now_ns (int): Time of this tick.
        """
        if self.__recorder is None:
            return
        for seq, weight in self.action_process.last_weights.items():
            self.__recorder.record_weight(
                now_ns=now_ns,
                seq=seq,
                weight=weight,
            )

    def __record_target(self, now_ns: int, stage: str,
                        target: dict) -> None:
        """Record one target.

        Args:
            now_ns (int): Time of this tick.
            stage (str): Recorder stage name.
            target (dict): Target to record.
        """
        if self.__recorder is not None:
            self.__recorder.record_target(
                now_ns=now_ns,
                stage=stage,
                target=target,
            )

    def __millis(self, name: str) -> int:
        """Convert a parameter in milliseconds to nanoseconds.

        Args:
            name (str): Parameter name.

        Returns:
            int: Value in nanoseconds.
        """
        return int(self.__params[name] * NS_PER_MS)

    def __seconds(self, name: str) -> int:
        """Convert a parameter in seconds to nanoseconds.

        Args:
            name (str): Parameter name.

        Returns:
            int: Value in nanoseconds.
        """
        return int(self.__params[name] * NS_PER_S)

    def __obs_divider(self) -> int:
        """Number of control ticks between two observation frames.

        Returns:
            int: Decimation factor, at least one.
        """
        ratio = (self.__params["control_rate_hz"]
                 / self.__params["observation_rate_hz"])
        return max(1, int(round(ratio)))

    @property
    def control_period_s(self) -> float:
        """Control period of the main loop.

        Returns:
            float: Period in seconds.
        """
        return 1.0 / self.__params["control_rate_hz"]


def split_stable_jnt(order: tuple[str, ...], flat: list,
                     arm_dof: int) -> dict[str, np.ndarray]:
    """Split the flat stable pose into one array per arm.

    Args:
        order (tuple): Arm names in the order of the flat array.
        flat (list): Concatenated joint targets.
        arm_dof (int): Number of joints of one arm.

    Returns:
        dict: Arm name to joint target.

    Raises:
        RuntimeError: If the length does not match the topology.
    """
    expected = len(order) * arm_dof
    if len(flat) != expected:
        raise RuntimeError(
            f"length of stable_jnt {len(flat)} does not match "
        f"stable_arm_order x arm_dof "
            f"= {len(order)} x {arm_dof} = {expected}")
    values = np.asarray(flat, dtype=np.float64)
    return {name: values[index * arm_dof:(index + 1) * arm_dof].copy()
            for index, name in enumerate(order)}


def read_params(node: Any) -> dict:
    """Declare and read every parameter, refusing unset ones.

    Args:
        node: ROS node that carries the parameters.

    Returns:
        dict: Parameter name to the value found on the node.

    Raises:
        RuntimeError: If a parameter is not provided by the configuration.
    """
    descriptor = ParameterDescriptor(dynamic_typing=True)
    for name in PARAM_NAMES:
        node.declare_parameter(name, descriptor=descriptor)
    values = {}
    missing = []
    for name in PARAM_NAMES:
        value = node.get_parameter(name).value
        if value is None:
            missing.append(name)
        values[name] = value
    if missing:
        raise RuntimeError(f"parameters not set: {missing}")
    return values


def validate_params(values: dict) -> dict:
    """Check the parameters and convert them to their working types.

    Args:
        values (dict): Raw parameter values.

    Returns:
        dict: Validated parameters.

    Raises:
        RuntimeError: If a value is out of range or inconsistent.
    """
    result = dict(values)
    for name in ("control_rate_hz", "observation_rate_hz", "stable_duration_s",
                 "stable_timeout_s", "arrive_threshold", "init_timeout_s",
                 "arm_jnt_threshold", "arm_pos_threshold",
                 "data_freshness_ms", "weight_decay"):
        result[name] = float(values[name])
    for name in ("arm_dof", "action_step_ms", "action_queue_capacity"):
        result[name] = int(values[name])
    result["action_average"] = str(values["action_average"])
    result["max_episode_s"] = float(values["max_episode_s"])
    result["stable_arm_order"] = list(values["stable_arm_order"])
    result["stable_jnt"] = list(values["stable_jnt"])
    problems = []
    if result["arm_dof"] <= 0:
        problems.append("arm_dof must be positive")
    if result["control_rate_hz"] <= 0:
        problems.append("control_rate_hz must be positive")
    if result["observation_rate_hz"] <= 0:
        problems.append("observation_rate_hz must be positive")
    if result["observation_rate_hz"] > result["control_rate_hz"]:
        problems.append("observation_rate_hz must not exceed control_rate_hz")
    if result["action_step_ms"] <= 0:
        problems.append("action_step_ms must be positive")
    if result["action_queue_capacity"] <= 0:
        problems.append("action_queue_capacity must be positive")
    if not np.isfinite(result["weight_decay"]):
        problems.append("weight_decay must be finite")
    if result["action_average"] not in MERGE_STRATEGIES:
        problems.append(
            f"action_average must be one of {MERGE_STRATEGIES}")
    if result["data_freshness_ms"] <= 0:
        problems.append("data_freshness_ms must be positive")
    for name in ("arrive_threshold", "arm_jnt_threshold",
                 "arm_pos_threshold"):
        if result[name] <= 0:
            problems.append(f"{name} must be positive")
    if result["stable_timeout_s"] < result["stable_duration_s"]:
        problems.append("stable_timeout_s must not be smaller than "
                        "stable_duration_s")
    if not result["stable_arm_order"]:
        problems.append("stable_arm_order must not be empty")
    if len(result["stable_jnt"]) != (len(result["stable_arm_order"])
                                     * result["arm_dof"]):
        problems.append(
            "stable_jnt length does not match stable_arm_order x arm_dof")
    if not result["device_path"]:
        problems.append("device_path must not be empty")
    if not result["interpolation"]:
        problems.append("interpolation must not be empty")
    if problems:
        raise RuntimeError("invalid parameters: " + "; ".join(problems))
    return result


def _field_map(hub: Any) -> dict[str, str]:
    """Collect the devices whose wire name differs from their device name.

    Args:
        hub: Device hub built from the device table.

    Returns:
        dict: Device name to model field name, empty when all are identity.
    """
    names: dict[str, str] = {}
    groups = (hub.camera_devices, hub.arm_devices)
    for devices in groups:
        for name, device in devices.items():
            if device.model_field != name:
                names[name] = device.model_field
    return names


def _node_logger(node: Any) -> Any:
    """Pick the logger of the ROS node, falling back to the stdlib one.

    Args:
        node: ROS node or a test double.

    Returns:
        Any: Logger that exposes the logging interface.
    """
    getter = getattr(node, "get_logger", None)
    if callable(getter):
        return getter()
    import logging
    return logging.getLogger(__name__)


def main() -> int:
    """Run the node until it reaches the terminal state.

    Returns:
        int: Process exit code.
    """
    rclpy.init()
    client = DualManipPolicyClient(ros_node=Node("policy_client"))
    executor = SingleThreadedExecutor()
    executor.add_node(client.ros_node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    signal.signal(signal.SIGINT, _on_sigint(client=client))
    client.start()
    period_s = client.control_period_s
    try:
        while rclpy.ok():
            if not client.step():
                break
            _pace(period_s=period_s)
    except KeyboardInterrupt:
        client.request_exit(reason=REASON_KEYBOARD_INTERRUPT)
        client.step()
    finally:
        executor.shutdown()
        thread.join(timeout=JOIN_TIMEOUT_S)
        client.shutdown()
        client.ros_node.destroy_node()
        rclpy.try_shutdown()
    return client.exit_code


def _on_sigint(client: "DualManipPolicyClient") -> Callable:
    """Build the SIGINT handler that only queues an exit request.

    Args:
        client (DualManipPolicyClient): Node to notify.

    Returns:
        Callable: Signal handler.
    """
    def handler(signum: int, frame: Any) -> None:
        client.request_exit(reason=REASON_KEYBOARD_INTERRUPT)

    return handler


def _pace(period_s: float) -> None:
    """Sleep until the next control period.

    The wall clock is used for pacing only; every data timestamp still comes
    from the ROS clock through the node.

    Args:
        period_s (float): Control period in seconds.
    """
    global _next_deadline_s
    _next_deadline_s += period_s
    delay = _next_deadline_s - time.monotonic()
    if delay > 0.0:
        time.sleep(delay)
    else:
        _next_deadline_s = time.monotonic()


_next_deadline_s = time.monotonic()
