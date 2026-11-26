################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

import signal
from typing import Any, Callable

from .core.dual_manip_robot import (
    SECTION_ARMS,
    SECTION_CAMERAS,
    DualManipRobot,
)
from .core.state_machine import StateMachine
from .process.action.action_window_query import (
    ActionWindowQuery,
    WindowQueryParam,
)
from .process.observation.primary_observation_assembly import (
    PrimaryAssemblyParam,
    PrimaryObservationAssembly,
)
from .protocol.policy_middleware import PolicyMiddleware
from .protocol.proto_helper import (
    build_inference_frame,
    build_observation_frame,
    decode_action_chunk,
)
from .utils import DataInterface

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000

STATE_INIT = "INIT"
STATE_RETURN_TO_STABLE = "RETURN_TO_STABLE"
STATE_WAITING = "WAITING"
STATE_RUNNING = "RUNNING"
STATE_EXIT = "EXIT"

REASON_EXIT = "exit"
REASON_INIT_TIMEOUT = "init_timeout"
REASON_KEYBOARD_INTERRUPT = "keyboard_interrupt"


class DualManipPolicyClient:
    """Policy client assembled for manipulators plus cameras.

    The node owns the loop and the state machine, every other object is a
    collaborator built on start: the robot, the two assemblers and the
    transport. Nothing below the node knows about states.

    Attributes:
        SECTION_ARMS (str): Device table section holding the arms.
        SECTION_CAMERAS (str): Device table section holding the cameras.
    """

    _MIDDLEWARE_QUEUE_MAX = 100
    _STOP_KEY = "key_q"

    def __init__(
        self,
    ) -> None:
        """Build every collaborator that does not need the device table.

        The two assemblers are built on start instead, because they need the
        device names the table declares. A failure here propagates, so nothing
        is left half built and nothing has to be released.
        """
        self.__ros_interface = DataInterface(name="dual_manip_policy_client")
        self.__logger = self.__ros_interface.getlogger()
        self.__params = self.__ros_interface.get_client_param()
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
            logger=self.__logger,
            create_subscription=self.__ros_interface.create_subscription,
            create_publisher=self.__ros_interface.create_publisher,
            now_fn=self.get_now_timestamp,
        )
        self.__observation: PrimaryObservationAssembly | None = None
        self.__action_process: ActionWindowQuery | None = None
        self.__current_obs: dict = {"camera": {}, "arm": {}}
        self.__started = False
        self.__shutdown_done = False
        self.__exit_reason = ""
        self.__state_since_ns = 0
        self.__last_state = STATE_INIT
        self.__last_obs_ns = 0
        self.__obs_count = 0
        self.__chunk_seen = False
        self.__anchor_given = False
        self.__prev_stop_key = False
        self.__read_scalars()

    def start(self) -> None:
        """Start the devices, build the assemblers and register the states.

        The call is idempotent. A failure inside the block releases what was
        already started and leaves the node unstarted, so the loop never runs
        over half built collaborators.
        """
        if self.__started:
            return
        try:
            self.__robot.start()
            self.__observation, self.__action_process = (
                self._build_components(robot=self.__robot))
            self.__register_states()
            self.__register_transitions()
            self.__policy_middleware.start()
            self.__state_since_ns = self.get_now_timestamp()
            self.__last_state = self.__state_machine.state
            self.__started = True
        except Exception as error:  # noqa: BLE001
            self.__logger.logw(
                "[node.start/not_ready] node cannot start: "
                f"collaborator assembly failed: {error}")
            self.__exit_reason = REASON_EXIT
            self.shutdown()

    def run(self) -> None:
        """Drive the loop until the node reaches the terminal state."""
        if not self.__started:
            return
        while self.__ros_interface.ok():
            self.__step()
            self.__state_machine.step()
            if self.__state_machine.state == STATE_EXIT:
                break
            self.__ros_interface.sleep()

    def publish_inference_state(self, now_ns: int) -> None:
        """Publish the inference enable flag of this tick.

        The flag follows the current state and is published every tick, so the
        model side sees ``False`` even while no observation is sent.

        Args:
            now_ns (int): Time of this tick, in nanoseconds.
        """
        enabled = self.__state_machine.state == STATE_RUNNING
        self.__policy_middleware.pub_inference_state(
            frame=build_inference_frame(enabled=enabled))

    def request_exit(self, reason: str) -> None:
        """Ask for the terminal state without releasing anything.

        Callable from the signal handler and from the callback thread, because
        it only records the reason and leaves a deferred transition.

        Args:
            reason (str): Reason kept for the log line of the exit.
        """
        self.__exit_reason = reason
        if self.__state_machine is None:
            return
        if self.__state_machine.state == STATE_EXIT:
            return
        self.__state_machine.transition_to(STATE_EXIT)

    def shutdown(self) -> None:
        """Release every resource once.

        The order is fixed and every step is isolated, so a failure in one of
        them does not keep the others from running.
        """
        if self.__shutdown_done:
            return
        self.__shutdown_done = True
        steps = (
            ("robot", self.__release_robot),
            ("middleware", self.__release_middleware),
            ("interface", self.__release_interface),
        )
        for name, action in steps:
            try:
                action()
            except Exception as error:  # noqa: BLE001
                self.__logw(
                    f"[node.shutdown/not_ready] release of {name} failed: "
                    f"{error}")
        self.__started = False

    def get_now_timestamp(self) -> int:
        """Return the clock of the node.

        Returns:
            int: Current time in nanoseconds.
        """
        stamp = self.__ros_interface.get_nowstamp()
        return int(stamp.sec) * NS_PER_S + int(stamp.nanosec)

    def _build_components(self, robot: Any) -> tuple:
        """Build the two assemblers on the topology of the robot.

        Args:
            robot: Robot whose devices were created on start.

        Returns:
            tuple: Observation assembler and action chunk processor.

        Raises:
            ValueError: If the table declares no camera to anchor a frame.
        """
        arm_names = robot.device_names[SECTION_ARMS]
        camera_names = robot.device_names[SECTION_CAMERAS]
        if not camera_names:
            raise ValueError(
                "device table declares no camera, so no frame can be anchored")
        freshness_ns = int(self.__params["data_freshness_ms"] * NS_PER_MS)
        observation = PrimaryObservationAssembly(
            params=PrimaryAssemblyParam(
                now_fn=self.get_now_timestamp,
                logger=self.__logger,
                primary_camera=camera_names[0],
                camera_skew_ns=freshness_ns,
            ))
        action_process = ActionWindowQuery(
            params=WindowQueryParam(
                arm_names=arm_names,
                now_fn=self.get_now_timestamp,
                anchor_tolerance_ns=freshness_ns,
                logger=self.__logger,
            ))
        return observation, action_process

    def __read_scalars(self) -> None:
        """Convert the time limits and the decimation once, at start."""
        self.__init_timeout_ns = int(
            self.__params["init_timeout_s"] * NS_PER_S)
        self.__stable_duration_ns = int(
            self.__params["stable_duration_s"] * NS_PER_S)
        self.__arrive_threshold = float(self.__params["arrive_threshold"])
        self.__episode_ns = int(self.__params["max_episode_s"] * NS_PER_S)
        self.__freshness_ns = int(
            self.__params["data_freshness_ms"] * NS_PER_MS)
        rate = float(self.__params["observation_rate_hz"])
        control_rate = float(self.__params["rate_ros"])
        self.__obs_divider = max(1, int(round(control_rate / rate)))

    def __register_states(self) -> None:
        """Register every state with the task it runs each step."""
        states = (
            (STATE_INIT, self.__init_behavior),
            (STATE_RETURN_TO_STABLE, self.__stable_behavior),
            (STATE_WAITING, self.__waiting_behavior),
            (STATE_RUNNING, self.__running_behavior),
            (STATE_EXIT, self.__exit_behavior),
        )
        for name, callback in states:
            self.__state_machine.register_state(
                name=name, callback=callback)

    def __register_transitions(self) -> None:
        """Register every edge the node may take."""
        edges = (
            (STATE_INIT, STATE_RETURN_TO_STABLE),
            (STATE_INIT, STATE_EXIT),
            (STATE_RETURN_TO_STABLE, STATE_WAITING),
            (STATE_RETURN_TO_STABLE, STATE_EXIT),
            (STATE_WAITING, STATE_RUNNING),
            (STATE_WAITING, STATE_EXIT),
            (STATE_RUNNING, STATE_RETURN_TO_STABLE),
            (STATE_RUNNING, STATE_EXIT),
        )
        for source, target in edges:
            self.__state_machine.register_transition(
                source=source, target=target)

    def __init_behavior(self) -> None:
        """Wait until every declared source and the keyboard reported once.

        The judgement is presence and elapsed time only; the frame itself is
        not consumed here, because no policy runs before the stable pose.
        """
        self.__note_state_entry()
        keyboard = self.__ros_interface.get_keyboard_state(latest=True)
        stop_pressed = (keyboard is not None
                        and bool(getattr(keyboard, self._STOP_KEY, False)))
        if stop_pressed and not self.__prev_stop_key:
            self.__logw(
                f"[node.init/keyboard_stop] stop and exit: "
                f"key {self._STOP_KEY} went down")
            self.__prev_stop_key = stop_pressed
            self.request_exit(reason=REASON_KEYBOARD_INTERRUPT)
            return
        self.__prev_stop_key = stop_pressed
        if self.__devices_ready() and keyboard is not None:
            self.__state_machine.transition_to(STATE_RETURN_TO_STABLE)
            return
        if self.__elapsed_ns() >= self.__init_timeout_ns:
            self.__logw(
                "[node.init/timeout] exit requested: sources did not report "
                f"within {self.__params['init_timeout_s']}s, "
                f"devices_ready={self.__devices_ready()}, "
                f"keyboard_ready={keyboard is not None}")
            self.request_exit(reason=REASON_INIT_TIMEOUT)

    def __stable_behavior(self) -> None:
        """Ramp every arm to its stable pose and wait for the arrival."""
        self.__note_state_entry()
        arrived = self.__robot.move_to_stable(
            duration_ns=self.__stable_duration_ns,
            arrive_threshold=self.__arrive_threshold,
        )
        elapsed_ns = self.__elapsed_ns()
        if arrived or elapsed_ns >= self.__stable_duration_ns:
            self.__state_machine.transition_to(STATE_WAITING)

    def __waiting_behavior(self) -> None:
        """Hold the stable pose until the model sends the first chunk."""
        self.__note_state_entry()
        self.__robot.hold_position()
        if self.__chunk_seen:
            self.__chunk_seen = False
            self.__last_obs_ns = self.get_now_timestamp()
            self.__state_machine.transition_to(STATE_RUNNING)

    def __running_behavior(self) -> None:
        """Publish observations on the decimated ticks and dispatch targets."""
        self.__note_state_entry()
        now_ns = self.get_now_timestamp()
        self.__obs_count += 1
        if self.__obs_count >= self.__obs_divider:
            self.__obs_count = 0
            self.__publish_observation(now_ns=now_ns)
        age_ns = now_ns - self.__last_obs_ns
        if age_ns >= self.__freshness_ns:
            self.__logw(
                "[node.ctrl/stale] command dropped: "
                f"observation age {age_ns / NS_PER_MS:.1f}ms "
                f"> {self.__freshness_ns / NS_PER_MS:.1f}ms")
            return
        action_process = self.__action_process
        if action_process is None:
            return
        target = action_process.get_target()
        if target is None:
            return
        self.__robot.send_ctrl(target=target)
        if self.__elapsed_ns() >= self.__episode_ns:
            self.__logw(
                "[node.task/timeout] back to stable: "
                f"task ran {self.__params['max_episode_s']}s")
            self.__state_machine.transition_to(STATE_RETURN_TO_STABLE)

    def __exit_behavior(self) -> None:
        """Release every resource once the terminal state is entered."""
        self.shutdown()

    def __publish_observation(self, now_ns: int) -> None:
        """Push the newest observation and publish the assembled frame.

        Args:
            now_ns (int): Time of this tick, in nanoseconds.
        """
        observation = self.__observation
        action_process = self.__action_process
        if observation is None or action_process is None:
            return
        observation.push_obs(observation=self.__current_obs)
        frame = observation.get_obs()
        if frame is None:
            self.__logw(
                "[node.obs/dropped] frame skipped: "
                "assembly returned no frame for this instant")
            return
        wire = build_observation_frame(observation=frame)
        if self.__policy_middleware.pub_obs(frame=wire):
            action_process.push_obs(observation=frame)
            self.__last_obs_ns = now_ns

    def __drain_chunks(self) -> None:
        """Take every queued chunk and hand the valid ones to the action."""
        frame = self.__policy_middleware.request_chunk()
        if frame is None:
            return
        try:
            chunk = decode_action_chunk(frame=frame)
        except ValueError as error:
            self.__logw(
                "[node.chunk/dropped] chunk skipped: "
                f"decode failed: {error}")
            return
        action_process = self.__action_process
        if action_process is None:
            return
        if not self.__anchor_given:
            self.__observation.push_obs(observation=self.__current_obs)
            frame = self.__observation.get_obs()
            if frame is None:
                self.__logw(
                    "[node.chunk/no_anchor] chunk held back: "
                    "no observation frame is available yet")
                return
            action_process.push_obs(observation=frame)
            self.__anchor_given = True
        action_process.push_chunk(chunk=chunk)
        self.__chunk_seen = True

    def __devices_ready(self) -> bool:
        """Whether every declared device reported at least once.

        The robot hands out an empty group before a device sampled anything, so
        a missing source shows up either as a group without the name or as a
        name whose sample is ``None``; both are caught by comparing the names.

        Returns:
            bool: ``True`` when no declared source is still missing.
        """
        declared = set()
        for section in (SECTION_ARMS, SECTION_CAMERAS):
            declared.update(self.__robot.device_names[section])
        present = set()
        for samples in self.__current_obs.values():
            for name, sample in (samples or {}).items():
                if sample is not None:
                    present.add(name)
        return declared <= present

    def __note_state_entry(self) -> None:
        """Record the instant the running state was entered.

        The machine has no entry hook and runs the task of a new state inside
        the same step that migrates, so the first task call of a state would
        otherwise measure the elapsed time of the state it came from.
        """
        state = self.__state_machine.state
        if state == self.__last_state:
            return
        self.__last_state = state
        self.__state_since_ns = self.get_now_timestamp()
        self.__obs_count = 0
        self.__logw(f"[node.state/changed] entering state {state}")

    def __elapsed_ns(self) -> int:
        """Return how long the current state has been running.

        Returns:
            int: Elapsed time in nanoseconds.
        """
        return self.get_now_timestamp() - self.__state_since_ns

    def __release_robot(self) -> None:
        """Stop the devices."""
        self.__robot.stop()

    def __release_middleware(self) -> None:
        """Close the transport."""
        self.__policy_middleware.close()

    def __release_interface(self) -> None:
        """Shut the ROS interface down last."""
        self.__ros_interface.shutdown()

    def __step(self) -> None:
        """Gather the data of this tick.

        The flow itself, the state machine step and the pacing stay in
        :meth:`run`; this method only refreshes what the behaviours read.
        """
        now_ns = self.get_now_timestamp()
        self.__current_obs = self.__robot.read_obs()
        self.__drain_chunks()
        self.publish_inference_state(now_ns=now_ns)

    def __logw(self, message: str) -> None:
        """Write one warning line through the logger when it exists.

        Args:
            message (str): Text to log.
        """
        self.__logger.logw(message)


def main() -> int:
    """Run the node until it reaches the terminal state.

    Returns:
        int: Process exit code.
    """
    client = DualManipPolicyClient()
    client.start()
    signal.signal(signal.SIGINT, _on_sigint(client=client))
    try:
        client.run()
    finally:
        client.shutdown()
    return 0


def _on_sigint(client: DualManipPolicyClient) -> Callable:
    """Build the handler that only leaves a deferred exit request.

    Args:
        client (DualManipPolicyClient): Node to notify.

    Returns:
        Callable: Signal handler.
    """
    def handler(signum: int, frame: Any) -> None:
        client.request_exit(reason=REASON_KEYBOARD_INTERRUPT)
    return handler
