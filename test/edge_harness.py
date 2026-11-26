#!/usr/bin/env python3
"""Scenario harness with the client in a process of its own.

Three roles take part. The device node and the model side live in this
process and share one rclpy context; the client under test runs in a
separate process, so its context, its spin thread and its shutdown stay its
own. That split is what makes the scenarios meaningful: sharing one context
made the client's ``rclpy.spin`` die whenever this side was torn down, and
every scenario then failed for a reason that had nothing to do with the
client.

Every check is taken from the wire or from the lines the client printed:
the control commands the device node received, the frames the model side
received, and the state transitions and warnings the client wrote.

Usage, from the package directory with the workspace sourced::

    python3 test/edge_harness.py --scenario all --budget 45
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from hex_ros_msgs.msg import (
    HexRosRoboArmState,
    HexRosRoboGripState,
    HexRosRoboManipCtrlStamped,
    HexRosRoboManipState,
    HexRosRoboManipStateStamped,
    HexRosTeleopKeyboardState,
    HexRosTeleopKeyboardStateStamped,
)
from hex_util_zenoh import HexZenohNode
from hex_util_zenoh.proto_util import pack

NS_PER_SEC = 1_000_000_000
DOF = 6
ARMS = ("left", "right")
CAMERA = "overhead"
KEYBOARD_TOPIC = "teleop_keyboard_state"
STABLE_POSITION = [0.0, -1.5, 3.0, 0.0, 0.0, 0.0]
ARM_VALUE = 0.3
STEP_INTERVAL_MS = 20
ROW_COUNT = 8
TOLERANCE = 1e-6
CHUNK_PERIOD_S = 0.1
DEVICE_TABLE = {
    "cameras": {
        CAMERA: {
            "ros_cfg": {
                "namespace": CAMERA,
                "subscribers": {
                    "color_topic": "image_color_raw",
                    "depth_topic": None,
                },
            },
            "crop_mode": "cover",
            "resize_w": 128,
            "resize_h": 128,
            "jpeg_quality": 90,
        },
    },
    "arms": {
        name: {
            "ros_cfg": {
                "namespace": name,
                "subscribers": {"state_topic": "manip_state"},
                "publishers": {"ctrl_topic": "manip_ctrl"},
            },
            "arm_cfg": {
                "lim_vel": 0.5,
                "lim_acc": 50.0,
                "grav": [0.0, 0.0, -9.81],
                "stable_position": list(STABLE_POSITION),
                "kp": [50.0] * DOF,
                "kd": [2.5] * DOF,
            },
            "grip_cfg": {
                "lim_pos": [0.0, 0.70],
                "lim_vel": 1.0,
                "lim_eff": 1.0,
            },
        }
        for name in ARMS
    },
}
BASE_PARAMS = {
    "rate_ros": 200.0,
    "observation_rate_hz": 30.0,
    "stable_duration_s": 0.5,
    "arrive_threshold": 0.05,
    "init_timeout_s": 3.0,
    "max_episode_s": 20.0,
    "data_freshness_ms": 120.0,
    "observation_topic": "",
    "inference_topic": "",
    "action_topic": "",
    "recorder_dir": "",
    "device_path": "",
}


def base_params(device_path: str, **overrides) -> dict:
    """Return the parameter set of one scenario.

    Args:
        device_path (str): Path of the device table of this scenario.
        **overrides: Parameters this scenario replaces.

    Returns:
        dict: Parameters the client is built with.
    """
    tag = "harness_{}".format(int(time.time() * 1000) % 1000000)
    params = dict(BASE_PARAMS)
    params["device_path"] = device_path
    params["observation_topic"] = "policy/{}/observation".format(tag)
    params["inference_topic"] = "policy/{}/inference".format(tag)
    params["action_topic"] = "policy/{}/action".format(tag)
    params.update(overrides)
    return params


def write_table(table: dict) -> str:
    """Write a device table into a temporary file.

    Args:
        table (dict): Device table as plain data.

    Returns:
        str: Path of the written file.
    """
    handle, path = tempfile.mkstemp(suffix=".yaml", prefix="harness_devices_")
    os.close(handle)
    import yaml
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(table, stream, sort_keys=False)
    return path


def fake_image(nanoseconds: int) -> Image:
    """Build one standard colour image, as a camera would publish.

    Args:
        nanoseconds (int): Stamp of the image.

    Returns:
        Image: A ``bgr8`` gradient, so a decoder sees real content.
    """
    message = Image()
    message.header.stamp.sec = nanoseconds // NS_PER_SEC
    message.header.stamp.nanosec = nanoseconds % NS_PER_SEC
    message.height = 48
    message.width = 64
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = message.width * 3
    row = bytes((index * 4) % 256 for index in range(message.step))
    message.data = row * message.height
    return message


class Devices(Node):
    """Publish device messages and record the control commands received."""

    def __init__(self, joint_count: int = DOF) -> None:
        super().__init__("harness_devices")
        self.joint_count = joint_count
        self.ctrl: list = []
        self.publishing = True
        self.keyboard_sent = False
        self.keys = {"key_q": False}
        self.stamp_ns = 0
        self.__state_pubs = {
            name: self.create_publisher(
                HexRosRoboManipStateStamped, "{}/manip_state".format(name), 10)
            for name in ARMS
        }
        self.__image_pub = self.create_publisher(
            Image, "{}/image_color_raw".format(CAMERA), 10)
        self.__keyboard_pub = self.create_publisher(
            HexRosTeleopKeyboardStateStamped, KEYBOARD_TOPIC, 10)
        self.__ctrl_subs = [
            self.create_subscription(
                HexRosRoboManipCtrlStamped, "{}/manip_ctrl".format(name),
                self.__on_ctrl, 10)
            for name in ARMS
        ]
        self.__arm_timer = self.create_timer(1.0 / 100.0, self.__tick_arms)
        self.__camera_timer = self.create_timer(
            1.0 / 30.0, self.__tick_camera)

    def hold_key(self, name: str) -> None:
        """Press one keyboard key for every later message.

        Args:
            name (str): Key field, ``key_q`` for instance.
        """
        self.keys[name] = True

    def __on_ctrl(self, message) -> None:
        self.ctrl.append(message)

    def __tick_arms(self) -> None:
        if not self.publishing:
            return
        stamp = self.get_clock().now().to_msg()
        self.stamp_ns = stamp.sec * NS_PER_SEC + stamp.nanosec
        for name in ARMS:
            state = HexRosRoboManipStateStamped()
            state.header.stamp = stamp
            arm = HexRosRoboArmState()
            joint = JointState()
            joint.position = [float(v)
                              for v in STABLE_POSITION[:self.joint_count]]
            joint.velocity = [0.0] * self.joint_count
            joint.effort = [0.0] * self.joint_count
            arm.jnt = joint
            grip = HexRosRoboGripState()
            grip_joint = JointState()
            grip_joint.position = [0.0]
            grip_joint.velocity = [0.0]
            grip_joint.effort = [0.0]
            grip.jnt = grip_joint
            manip = HexRosRoboManipState()
            manip.arm_state = arm
            manip.grip_state = grip
            state.manip_state = manip
            self.__state_pubs[name].publish(state)
        keyboard = HexRosTeleopKeyboardStateStamped()
        keyboard.header.stamp = stamp
        state = HexRosTeleopKeyboardState()
        for name, pressed in self.keys.items():
            setattr(state, name, pressed)
        keyboard.keyboard_state = state
        self.__keyboard_pub.publish(keyboard)
        self.keyboard_sent = True

    def __tick_camera(self) -> None:
        if not self.publishing:
            return
        stamp = self.get_clock().now().to_msg()
        self.__image_pub.publish(fake_image(
            nanoseconds=stamp.sec * NS_PER_SEC + stamp.nanosec))

    def action_positions(self) -> list:
        """Return the joint positions of every action command received.

        A stable command carries the stable pose and an action command the
        value the policy asked for, so the two are told apart by value.

        Returns:
            list: One list of floats per command that is not the stable pose.
        """
        found = []
        for message in self.ctrl:
            positions = [float(v)
                         for v in message.manip_ctrl.arm_ctrl.jnt.pos]
            if np.allclose(positions, STABLE_POSITION, atol=TOLERANCE):
                continue
            found.append(positions)
        return found

    def stop(self) -> None:
        """Stop publishing and drop the node."""
        self.publishing = False
        try:
            self.destroy_node()
        except Exception:  # noqa: BLE001
            pass


class Model:
    """Play the model side over zenoh."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self.observations: list = []
        self.inference: list = []
        self.sending = False
        self.bad_only = False
        self.running = True
        self.node = HexZenohNode()
        self.node.start()
        deadline = time.time() + 5.0
        while not self.node.is_working and time.time() < deadline:
            time.sleep(0.02)
        self.node.create_sub(params["observation_topic"],
                             self.__on_observation)
        self.node.create_sub(params["inference_topic"], self.__on_inference)
        self.node.create_pub(params["action_topic"])
        self.__sender = threading.Thread(target=self.__send_loop, daemon=True)
        self.__sender.start()

    def __on_observation(self, payload: bytes) -> None:
        self.observations.append(payload)

    def __on_inference(self, payload: bytes) -> None:
        self.inference.append(payload)

    def __send_loop(self) -> None:
        while self.running:
            if self.bad_only:
                self.send_bad_chunk()
            elif self.sending:
                self.send_chunk()
            time.sleep(CHUNK_PERIOD_S)

    def send_chunk(self) -> None:
        """Publish one action chunk that asks for a constant target."""
        stamp_ns = int(time.time() * NS_PER_SEC)
        frame = {
            "header": {"secs": stamp_ns // NS_PER_SEC,
                       "nsecs": stamp_ns % NS_PER_SEC},
            "chunk_step_interval_ms": STEP_INTERVAL_MS,
            "action_mode": "abs",
            "arm_ctrl_mode": "pos",
            "arm": {name: [[ARM_VALUE] * DOF for _ in range(ROW_COUNT)]
                    for name in ARMS},
            "grip": {name: [[ARM_VALUE] for _ in range(ROW_COUNT)]
                     for name in ARMS},
        }
        self.node.pub(self.params["action_topic"], pack(frame=frame))

    def send_bad_chunk(self) -> None:
        """Publish a payload that is not a frame at all."""
        self.node.pub(self.params["action_topic"], b"not-a-frame")

    def close(self) -> None:
        """Stop the sender and close the session."""
        self.running = False
        time.sleep(2 * CHUNK_PERIOD_S)
        try:
            self.node.close()
        except Exception:  # noqa: BLE001
            pass


class LineReader:
    """Collect the output lines of the client process."""

    def __init__(self, stream) -> None:
        self.lines: list = []
        self.__stream = stream
        self.__thread = threading.Thread(target=self.__read, daemon=True)
        self.__thread.start()

    def __read(self) -> None:
        for line in self.__stream:
            self.lines.append(line.rstrip())

    def has(self, needle: str) -> bool:
        """Whether any collected line contains a text.

        Args:
            needle (str): Text to find.

        Returns:
            bool: ``True`` when a line contains it.
        """
        return any(needle in line for line in self.lines)

    def state(self) -> str:
        """Return the last state the client announced.

        Returns:
            str: State name, empty before the first announcement.
        """
        seen = ""
        for line in self.lines:
            if "entering state" in line:
                seen = line.split("entering state", 1)[1].strip()
        return seen

    def crashed(self) -> bool:
        """Whether the client printed a traceback.

        Returns:
            bool: ``True`` when a traceback was seen.
        """
        return self.has("Traceback (most recent call last)")


class Watchdog:
    """Kill the run when one scenario exceeds its budget."""

    def __init__(self, budget_s: float) -> None:
        self.budget_s = budget_s
        self.__timer: threading.Timer | None = None

    def start(self) -> None:
        """Arm the watchdog."""
        self.__timer = threading.Timer(self.budget_s, self.__bark)
        self.__timer.daemon = True
        self.__timer.start()

    def cancel(self) -> None:
        """Disarm the watchdog."""
        if self.__timer is not None:
            self.__timer.cancel()

    def __bark(self) -> None:
        print("  TIMEOUT after {}s".format(self.budget_s))
        sys.stdout.flush()
        os._exit(3)


def run_client(params: dict) -> int:
    """Run the real client in the current process.

    Args:
        params (dict): Parameters of this scenario.

    Returns:
        int: ``0`` when the client stopped on its own.
    """
    from hex_ros_policy_client import dual_manip_policy_client as node_mod
    node_mod.DataInterface.get_client_param = lambda self: params
    client = node_mod.DualManipPolicyClient()
    client.start()
    client.run()
    client.shutdown()
    return 0


class ClientProcess:
    """The client under test, running in its own process."""

    def __init__(self, params: dict, script: str) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-u", script, "--as-node",
             "--params-json", json.dumps(params)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1"))
        self.reader = LineReader(self.process.stdout)

    def state(self) -> str:
        """Return the state the client announced last.

        Returns:
            str: State name, empty before the first announcement.
        """
        return self.reader.state()

    def has(self, needle: str) -> bool:
        """Whether the client printed a text.

        Args:
            needle (str): Text to find.

        Returns:
            bool: ``True`` when a line contains it.
        """
        return self.reader.has(needle)

    def alive(self) -> bool:
        """Whether the client process still runs.

        Returns:
            bool: ``True`` while it has not exited.
        """
        return self.process.poll() is None

    def stop(self) -> None:
        """Terminate the client and collect what is left."""
        if os.environ.get("HARNESS_VERBOSE"):
            for line in self.reader.lines:
                print("      | {}".format(line[:120]))
        self.process.terminate()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
        time.sleep(0.2)


def drive(client: ClientProcess, devices: Devices, model: Model, predicate,
          timeout_s: float = 15.0) -> bool:
    """Spin the device node until the predicate holds.

    Args:
        client (ClientProcess): Client under test, already running.
        devices (Devices): Device side of this harness.
        model (Model): Model side of this harness.
        predicate (callable): Condition to reach.
        timeout_s (float): Longest wait in seconds.

    Returns:
        bool: Whether the predicate held in time.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        if not client.alive():
            return predicate()
        try:
            rclpy.spin_once(devices, timeout_sec=0.02)
        except Exception:  # noqa: BLE001
            return predicate()
    return predicate()


def run_isolated(name: str, script: str) -> dict:
    """Run one in-process scenario in a process of its own.

    Those scenarios shut the rclpy context down, so they cannot share the
    process with the ones that still need to spin a device node.

    Args:
        name (str): Scenario to run.
        script (str): Path of this file.

    Returns:
        dict: One check, true when the child exited cleanly.
    """
    finished = subprocess.run(
        [sys.executable, "-u", script, "--scenario", name],
        capture_output=True, text=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1"))
    for line in finished.stdout.splitlines():
        print(line)
    return {"child scenario passed": finished.returncode == 0}


def resolve_paths() -> tuple:
    """Return a device table path and the path of this file.

    Returns:
        tuple: ``(device_path, script_path)``.
    """
    return write_table(DEVICE_TABLE), os.path.abspath(__file__)


def scenario_happy(path: str) -> dict:
    """Full path: devices and keyboard arrive, the model sends chunks."""
    params = base_params(device_path=path)
    devices = Devices()
    model = Model(params)
    model.sending = True
    client = ClientProcess(params=params, script=os.path.abspath(__file__))
    reached = drive(
        client, devices, model,
        lambda: (client.state() == "RUNNING"
                 and len(model.observations) >= 5
                 and len(devices.action_positions()) >= 3), timeout_s=20.0)
    crashed = client.reader.crashed()
    client.stop()
    positions = devices.action_positions()
    expected = [ARM_VALUE] * DOF
    weighted = bool(positions) and all(
        np.allclose(values, expected, atol=TOLERANCE) for values in positions)
    result = {
        "reached RUNNING": client.state() == "RUNNING",
        "observations reached the model": len(model.observations) >= 5,
        "action target weighted as asked": weighted,
        "loop stayed consistent": reached,
        "client did not crash": not crashed,
    }
    model.close()
    devices.stop()
    return result


def scenario_init_timeout(path: str) -> dict:
    """No device message at all: the client must leave for EXIT."""
    params = base_params(device_path=path, init_timeout_s=2.0)
    devices = Devices()
    devices.publishing = False
    model = Model(params)
    client = ClientProcess(params=params, script=os.path.abspath(__file__))
    drive(client, devices, model,
          lambda: client.state() == "EXIT", timeout_s=10.0)
    crashed = client.reader.crashed()
    client.stop()
    result = {
        "goes to EXIT": client.state() == "EXIT",
        "init timeout logged": client.has("node.init/timeout"),
        "no command sent": len(devices.ctrl) == 0,
        "client did not crash": not crashed,
    }
    model.close()
    devices.stop()
    return result


def scenario_key_stop(path: str) -> dict:
    """The stop key goes down during INIT."""
    params = base_params(device_path=path, init_timeout_s=10.0)
    devices = Devices()
    devices.hold_key("key_q")
    model = Model(params)
    client = ClientProcess(params=params, script=os.path.abspath(__file__))
    drive(client, devices, model,
          lambda: client.state() == "EXIT", timeout_s=10.0)
    crashed = client.reader.crashed()
    client.stop()
    result = {
        "goes to EXIT": client.state() == "EXIT",
        "stop key logged": client.has("node.init/keyboard_stop"),
        "keyboard arrived": devices.keyboard_sent,
        "client did not crash": not crashed,
    }
    model.close()
    devices.stop()
    return result


def scenario_bad_chunk(path: str) -> dict:
    """A malformed chunk must be dropped, the loop must survive."""
    params = base_params(device_path=path)
    devices = Devices()
    model = Model(params)
    model.bad_only = True
    client = ClientProcess(params=params, script=os.path.abspath(__file__))
    drive(client, devices, model,
          lambda: client.has("frame.schema/schema_invalid"), timeout_s=15.0)
    crashed = client.reader.crashed()
    alive = client.alive()
    client.stop()
    result = {
        "malformed payload dropped by the link":
            client.has("frame.schema/schema_invalid"),
        "client still alive": alive,
        "client did not crash": not crashed,
    }
    model.close()
    devices.stop()
    return result


def scenario_stale(path: str) -> dict:
    """Observations stop arriving: commands must be dropped."""
    params = base_params(device_path=path, data_freshness_ms=60.0)
    devices = Devices()
    model = Model(params)
    model.sending = True
    client = ClientProcess(params=params, script=os.path.abspath(__file__))

    def predicate():
        if client.state() == "RUNNING" and devices.publishing:
            devices.publishing = False
        return client.has("node.ctrl/stale")

    drive(client, devices, model, predicate, timeout_s=20.0)
    crashed = client.reader.crashed()
    client.stop()
    result = {
        "stale command dropped": client.has("node.ctrl/stale"),
        "client did not crash": not crashed,
    }
    model.close()
    devices.stop()
    return result


def scenario_joint_mismatch(path: str) -> dict:
    """The arm reports fewer joints than the stable pose declares."""
    params = base_params(device_path=path, init_timeout_s=10.0)
    devices = Devices(joint_count=5)
    model = Model(params)
    client = ClientProcess(params=params, script=os.path.abspath(__file__))
    drive(client, devices, model,
          lambda: (client.has("stable_shape")
                   and client.state() in ("WAITING", "EXIT")),
          timeout_s=20.0)
    crashed = client.reader.crashed()
    client.stop()
    result = {
        "mismatch is reported": client.has("stable_shape"),
        "left INIT": client.state() in ("WAITING", "EXIT"),
        "client did not crash": not crashed,
    }
    model.close()
    devices.stop()
    return result


def in_process_client(params: dict):
    """Build the client in this process, without running it.

    Args:
        params (dict): Parameters of this scenario.

    Returns:
        object: Client that was started or tried to start.
    """
    from hex_ros_policy_client import dual_manip_policy_client as node_mod
    node_mod.DataInterface.get_client_param = lambda self: params
    return node_mod.DualManipPolicyClient()


def scenario_empty_device_path(given=None) -> dict:
    """An empty device path must fail cleanly, not crash the process."""
    client = in_process_client(base_params(device_path=""))
    client.start()
    logger = client._DualManipPolicyClient__logger
    started = client._DualManipPolicyClient__started
    client.shutdown()
    lines = list(getattr(logger, "lines", []))
    return {
        "start failure logged": any(
            "node.start/not_ready" in line for line in lines
        ) or started is False,
        "not started": started is False,
    }


def scenario_no_camera(given=None) -> dict:
    """A table without cameras cannot anchor a frame."""
    table = {key: value for key, value in DEVICE_TABLE.items()
             if key != "cameras"}
    client = in_process_client(base_params(device_path=write_table(table)))
    client.start()
    started = client._DualManipPolicyClient__started
    client.shutdown()
    return {
        "start refused": started is False,
    }


def scenario_idempotent(given=None) -> dict:
    """Starting and shutting down twice must be harmless."""
    client = in_process_client(base_params(device_path=write_table(
        DEVICE_TABLE)))
    client.start()
    client.start()
    client.shutdown()
    client.shutdown()
    return {"double start and double shutdown": True}


PROCESS_SCENARIOS = {
    "happy": scenario_happy,
    "init_timeout": scenario_init_timeout,
    "key_stop": scenario_key_stop,
    "bad_chunk": scenario_bad_chunk,
    "stale": scenario_stale,
    "joint_mismatch": scenario_joint_mismatch,
}
IN_PROCESS_SCENARIOS = {
    "empty_device_path": scenario_empty_device_path,
    "no_camera": scenario_no_camera,
    "idempotent": scenario_idempotent,
}
SCENARIOS = dict(PROCESS_SCENARIOS, **IN_PROCESS_SCENARIOS)
ORDER = list(PROCESS_SCENARIOS) + list(IN_PROCESS_SCENARIOS)


def main() -> int:
    """Run one scenario or every scenario.

    Returns:
        int: ``0`` when every check passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-node", action="store_true")
    parser.add_argument("--params-json", default="{}")
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--budget", type=float, default=45.0)
    args = parser.parse_args()
    if args.as_node:
        return run_client(json.loads(args.params_json))
    names = ORDER if args.scenario == "all" else [args.scenario]
    if any(name in PROCESS_SCENARIOS for name in names):
        rclpy.init()
    failed = []
    for name in names:
        watchdog = Watchdog(budget_s=args.budget)
        watchdog.start()
        try:
            if name in PROCESS_SCENARIOS:
                table_path, _ = resolve_paths()
                result = SCENARIOS[name](table_path)
                os.unlink(table_path)
            elif args.scenario == "all":
                result = run_isolated(name, os.path.abspath(__file__))
            else:
                result = SCENARIOS[name]()
        except Exception as error:  # noqa: BLE001
            print("  {:<20} ERROR {}".format(name, error))
            failed.append(name)
            continue
        finally:
            watchdog.cancel()
        for check, ok in result.items():
            print("  {:<18} {:<42} {}".format(
                name, check, "OK" if ok else "FAIL"))
            if not ok:
                failed.append(name)
    try:
        rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    print("failed:", ", ".join(sorted(set(failed))) or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
