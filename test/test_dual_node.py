################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the assembled dual manipulator node."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hex_ros_policy_client import dual_manip_policy_client as dual
from hex_ros_policy_client.policy_client_base import (
    EVENT_END,
    EVENT_EPISODE_TIMEOUT,
    EVENT_EXIT,
    EVENT_INIT_SUCCESS,
    EVENT_INIT_TIMEOUT,
    EVENT_STABLE_ARRIVED,
    EVENT_STABLE_TIMEOUT,
    EVENT_START,
    STATE_EXIT,
    STATE_INIT,
    STATE_RETURN_TO_STABLE,
    STATE_RUNNING,
    STATE_WAITING,
)
from hex_ros_policy_client.struct.device_struct import (
    ArmObservation,
    ArmState,
    CameraObservation,
    GripState,
    ImageFrame,
)
from hex_ros_policy_client.struct.geom_struct import Pose
from hex_util_zenoh.proto_util import policy_proto_parser
from hex_ros_policy_client.utils.common_util.logkit import (
    _reset_state as reset_log,
)

from .fakes import FakeRosNode
from .test_device_hub import DUAL_TABLE, SINGLE_TABLE, write_table

NS_PER_MS = 1_000_000

BASE_PARAMS = {
    "control_rate_hz": 1000.0,
    "observation_rate_hz": 30.0,
    "arm_dof": 7,
    "stable_arm_order": ["left", "right"],
    "stable_jnt": [0.0] * 14,
    "stable_duration_s": 2.0,
    "stable_timeout_s": 15.0,
    "arrive_threshold": 0.05,
    "init_timeout_s": 5.0,
    "arm_jnt_threshold": 0.1,
    "arm_pos_threshold": 0.2,
    "max_episode_s": 120.0,
    "data_freshness_ms": 40.0,
    "action_step_ms": 33,
    "observation_topic": "policy/observation",
    "inference_topic": "policy/inference",
    "action_topic": "policy/action",
    "interpolation": "linear",
    "device_path": "",
    "action_queue_capacity": 100,
    "weight_decay": 0.5,
    "action_average": "none",
    "recorder_dir": "",
}

SINGLE_PARAMS = {
    "stable_arm_order": ["main"],
    "stable_jnt": [0.0] * 7,
}


def build_client(tmp_path, overrides: dict | None = None,
                 table: str = DUAL_TABLE,
                 now_ns: int = 1_000_000_000) -> tuple:
    """Create a node bound to a fake ROS node.

    Args:
        tmp_path: Pytest temporary directory.
        overrides (dict | None): Parameter overrides.
        table (str): Device table contents.
        now_ns (int): Initial clock reading.

    Returns:
        tuple: Client and its fake ROS node.
    """
    params = dict(BASE_PARAMS)
    params.update(overrides or {})
    directory = Path(tmp_path) if tmp_path is not None else Path(
        tempfile.mkdtemp(prefix="policy_client_"))
    params["device_path"] = write_table(directory, table)
    node = FakeRosNode(params=params, now_ns=now_ns)
    client = dual.DualManipPolicyClient(node)
    return client, node


def fill_obs(client, stamp_ns: int, arms: tuple = (), cameras: tuple = ()):
    """Fill the observation of the current tick.

    Args:
        client: Node under test.
        stamp_ns (int): Stamp used for every source.
        arms (tuple): Arm names to fill, defaults to the device table.
        cameras (tuple): Camera names to fill, defaults to the device table.
    """
    hub = client.robot.device_hub
    arm_names = arms or hub.arm_names
    camera_names = cameras or hub.camera_names
    frame = ImageFrame(ts_ns=stamp_ns, encoding="jpeg", width=1, height=1,
                       data=np.zeros(1, dtype=np.uint8))
    client.current_obs = {
        "camera": {name: CameraObservation(rgb=frame)
                   for name in camera_names},
        "arm": {name: ArmObservation(stamp_ns=stamp_ns,
                                     arm_state=_arm_state(stamp_ns),
                                     grip_state=_grip_state())
                for name in arm_names},
    }


def _arm_state(stamp_ns: int) -> ArmState:
    """Build an arm state with zero joints.

    Args:
        stamp_ns (int): Unused stamp kept for readability.

    Returns:
        ArmState: Zero joint state.
    """
    return ArmState(arm_pos=np.zeros(7), arm_vel=np.zeros(7),
                    arm_eff=np.zeros(7), ee_base=Pose())


def _grip_state() -> GripState:
    """Build a gripper state with zero joints.

    Returns:
        GripState: Zero gripper state.
    """
    return GripState(grip_pos=np.zeros(1), grip_vel=np.zeros(1),
                     grip_eff=np.zeros(1))


def stub_devices(client, jnt: np.ndarray | None = None) -> None:
    """Replace the device read with a synthetic observation.

    Args:
        client: Node under test.
        jnt (np.ndarray | None): Joint positions per arm, zeros by default.
    """
    hub = client.robot.device_hub
    positions = np.zeros(7) if jnt is None else np.asarray(jnt)

    def read_obs() -> dict:
        stamp_ns = client.get_now_timestamp()
        frame = ImageFrame(ts_ns=stamp_ns, encoding="jpeg", width=1,
                           height=1, data=np.zeros(1, dtype=np.uint8))
        state = ArmState(arm_pos=positions.copy(), arm_vel=np.zeros(7),
                         arm_eff=np.zeros(7), ee_base=Pose())
        grip = GripState(grip_pos=np.zeros(1), grip_vel=np.zeros(1),
                         grip_eff=np.zeros(1))
        return {
            "camera": {name: CameraObservation(rgb=frame)
                       for name in hub.camera_names},
            "arm": {name: ArmObservation(stamp_ns=stamp_ns,
                                         arm_state=state, grip_state=grip)
                    for name in hub.arm_names},
        }

    client.robot.read_obs = read_obs


def goto(client, state: str) -> None:
    """Drive the machine to a state along the happy path.

    Args:
        client: Node under test.
        state (str): Target state.
    """
    path = {
        STATE_RETURN_TO_STABLE: [EVENT_INIT_SUCCESS],
        STATE_WAITING: [EVENT_INIT_SUCCESS, EVENT_STABLE_ARRIVED],
        STATE_RUNNING: [EVENT_INIT_SUCCESS, EVENT_STABLE_ARRIVED,
                        EVENT_START],
        STATE_EXIT: [EVENT_EXIT],
    }
    client.start()
    for event in path[state]:
        client.state_machine.post(event)
        client.state_machine.process_events()


def test_parameters_are_all_declared(wire) -> None:
    """Every parameter is declared and read from the node."""
    client, node = build_client(None)
    assert tuple(node.declared) == dual.PARAM_NAMES


def test_missing_parameter_is_refused() -> None:
    """A parameter that the configuration omits stops the start up."""
    params = dict(BASE_PARAMS)
    params.pop("device_path")
    node = FakeRosNode(params=params)
    with pytest.raises(RuntimeError):
        dual.DualManipPolicyClient(node, None)


def test_stable_length_mismatch_is_refused() -> None:
    """stable_jnt must match stable_arm_order times arm_dof."""
    params = dict(BASE_PARAMS)
    params["stable_jnt"] = [0.0] * 7
    with pytest.raises(RuntimeError):
        dual.validate_params(params)


def test_stable_timeout_below_duration_is_refused() -> None:
    """A timeout shorter than the motion itself is inconsistent."""
    params = dict(BASE_PARAMS)
    params["stable_timeout_s"] = 1.0
    with pytest.raises(RuntimeError):
        dual.validate_params(params)


def test_unknown_interpolation_is_refused() -> None:
    """The interpolation name is looked up in the table."""
    with pytest.raises(ValueError):
        dual.get_interpolator("cubic")


def test_registration_matches_the_tables(wire) -> None:
    """Every table entry is registered exactly once."""
    client, _ = build_client(None)
    machine = client.state_machine
    assert len(machine._StateMachine__on_event) == 11
    assert len(machine._StateMachine__on_tick) == 4
    assert len(machine._StateMachine__events) == 8


@pytest.mark.parametrize("start, event, expected", [
    (STATE_INIT, EVENT_INIT_SUCCESS, STATE_RETURN_TO_STABLE),
    (STATE_INIT, EVENT_INIT_TIMEOUT, STATE_EXIT),
    (STATE_INIT, EVENT_EXIT, STATE_EXIT),
    (STATE_RETURN_TO_STABLE, EVENT_STABLE_ARRIVED, STATE_WAITING),
    (STATE_RETURN_TO_STABLE, EVENT_STABLE_TIMEOUT, STATE_EXIT),
    (STATE_RETURN_TO_STABLE, EVENT_EXIT, STATE_EXIT),
    (STATE_WAITING, EVENT_START, STATE_RUNNING),
    (STATE_WAITING, EVENT_EXIT, STATE_EXIT),
    (STATE_RUNNING, EVENT_END, STATE_RETURN_TO_STABLE),
    (STATE_RUNNING, EVENT_EPISODE_TIMEOUT, STATE_RETURN_TO_STABLE),
    (STATE_RUNNING, EVENT_EXIT, STATE_EXIT),
])
def test_every_transition(wire, start: str, event: str,
                          expected: str) -> None:
    """All eleven edges of the transition table behave as designed."""
    client, _ = build_client(None)
    client.start()
    if start != STATE_INIT:
        goto(client, start)
    assert client.state_machine.state == start
    client.state_machine.post(event)
    client.state_machine.process_events()
    assert client.state_machine.state == expected


def test_observation_is_decimated(wire) -> None:
    """Thirty three ticks produce exactly one observation frame."""
    client, node = build_client(None)
    client.start()
    goto(client, STATE_RUNNING)
    stub_devices(client)
    reset_log()
    for _ in range(33):
        node.clock.now_ns += 1_000_000
        client.step()
    assert len(wire["obs"]) == 1


def test_stale_observation_blocks_dispatch(wire) -> None:
    """A target is not sent once the published observation is too old."""
    reset_log()
    client, node = build_client(None)
    client.start()
    goto(client, STATE_RUNNING)
    stub_devices(client)
    sent = []
    client.robot.send_ctrl = lambda target: sent.append(target)
    client.step()
    node.clock.now_ns += 200 * NS_PER_MS
    client.step()
    assert sent == []
    lines = node.logger.messages()
    assert any(line.startswith("[ctrl.dispatch/obs_stale] ")
               for line in lines)


def test_target_is_limited_against_the_measured_pose(wire) -> None:
    """The dispatched joint target moves at most one threshold per tick."""
    client, node = build_client(None)
    client.start()
    goto(client, STATE_RUNNING)
    stub_devices(client)
    sent = []
    client.robot.send_ctrl = lambda target: sent.append(target)
    for _ in range(33):
        node.clock.now_ns += 1_000_000
        client.step()
    chunk = {
        "header": {"secs": 0, "nsecs": node.clock.now_ns},
        "chunk_step_interval_ms": 33,
        "action_mode": "joint",
        "arm_ctrl_mode": "pos",
        "arm": {"left": [[1.0] * 7], "right": [[1.0] * 7]},
        "grip": {"left": [[0.0]], "right": [[0.0]]},
    }
    client.policy_middleware._PolicyMiddleware__on_action_payload(
        policy_proto_parser.pack(chunk))
    node.clock.now_ns += 1_000_000
    client.step()
    assert sent
    for command in sent[-1]["arm"].values():
        assert np.max(np.abs(np.asarray(command.pos))) == pytest.approx(0.1)


def test_policy_message_is_mapped_to_an_event(wire) -> None:
    """A start frame of the model side ends up as the start event."""
    client, _ = build_client(None)
    goto(client, STATE_WAITING)
    payload = policy_proto_parser.pack({"kind": "start"})
    client.policy_middleware._PolicyMiddleware__on_inference_payload(payload)
    assert client.state_machine.pending() == 1
    client.state_machine.process_events()
    assert client.state_machine.state == STATE_RUNNING


def test_unknown_kind_is_dropped(wire) -> None:
    """A kind without a mapping is reported and ignored."""
    reset_log()
    client, node = build_client(None)
    payload = policy_proto_parser.pack({"kind": "resume"})
    client.policy_middleware._PolicyMiddleware__on_inference_payload(payload)
    assert client.state_machine.pending() == 0
    assert "event.map/unsupported_kind" in node.logger.messages()[0]


@pytest.mark.parametrize("state, reason, event", [
    (STATE_RUNNING, "exit", EVENT_EXIT),
    (STATE_RUNNING, "keyboard_interrupt", EVENT_EXIT),
    (STATE_INIT, "init_timeout", EVENT_INIT_TIMEOUT),
    (STATE_RETURN_TO_STABLE, "stable_timeout", EVENT_STABLE_TIMEOUT),
])
def test_all_termination_paths_release_once(wire, state: str, reason: str,
                                            event) -> None:
    """Every ending reaches the terminal state and releases once."""
    client, _ = build_client(None)
    client.start()
    if state != STATE_INIT:
        goto(client, state)
    calls = []
    client.robot.stop = lambda: calls.append("stop")
    client.policy_middleware.close = lambda: calls.append("close")
    if reason in ("exit", "keyboard_interrupt"):
        client.request_exit(reason)
        client.step()
    else:
        client.state_machine.post(event)
        client.state_machine.process_events()
    assert client.state_machine.state == STATE_EXIT
    assert calls == ["stop", "close"]


def test_init_condition_posts_success_when_every_source_arrived(wire) -> None:
    """The initial state leaves as soon as every device reported."""
    client, node = build_client(None)
    client.start()
    fill_obs(client, node.clock.now_ns)
    client.check_conditions(node.clock.now_ns)
    client.state_machine.process_events()
    assert client.state_machine.state == STATE_RETURN_TO_STABLE


def test_init_condition_times_out(wire) -> None:
    """A device that never reports ends the node after the timeout."""
    client, node = build_client(None)
    client.start()
    node.clock.now_ns += 6_000_000_000
    client.check_conditions(node.clock.now_ns)
    client.state_machine.process_events()
    assert client.state_machine.state == STATE_EXIT
    assert client.exit_reason == "init_timeout"


def test_stable_condition_requires_arrival(wire) -> None:
    """The stable state leaves only after the arrival criterion is met."""
    client, node = build_client(None)
    client.start()
    goto(client, STATE_RETURN_TO_STABLE)
    client.check_conditions(node.clock.now_ns)
    assert client.state_machine.pending() == 0
    node.clock.now_ns += 3_000_000_000
    fill_obs(client, node.clock.now_ns)
    client.check_conditions(node.clock.now_ns)
    client.state_machine.process_events()
    assert client.state_machine.state == STATE_WAITING


def test_single_arm_table_reuses_the_same_class(wire) -> None:
    """One arm needs a table and a parameter change only."""
    client, _ = build_client(None, overrides=SINGLE_PARAMS,
                             table=SINGLE_TABLE)
    client.start()
    fill_obs(client, client.get_now_timestamp())
    client.check_conditions(client.get_now_timestamp())
    client.state_machine.process_events()
    assert client.state_machine.state == STATE_RETURN_TO_STABLE
    assert client.robot.device_hub.arm_names == ("main",)


def test_stable_order_must_match_the_table(wire) -> None:
    """A pose order that disagrees with the table stops the start up."""
    client, _ = build_client(None, overrides=SINGLE_PARAMS,
                             table=DUAL_TABLE)
    with pytest.raises(RuntimeError):
        client.start()
