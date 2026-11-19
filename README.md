# hex_ros_policy_client

Policy client for imitation-learning deployment on ROS 2. Devices become
topics: the node subscribes `manip_state` and `image_color_raw`, publishes
`manip_ctrl`, ships observations to the model side and executes the returned
action chunks.

## Layout

| Layer | Directory | Holds | May import |
|---|---|---|---|
| Node (`policy_client`) | package root | ROS node, parameter checks, event and behaviour tables | everything below |
| Runtime (`core`) | `core/` | robot composition, state machine | `process`, adapters, `utils`, `struct` |
| Processing (`process`) | `process/` | observation assembly, action chunk integration | adapters, `utils`, `struct` |
| Adapter (`protocol`) | `protocol/` | wire contract, frame codec, Zenoh transport | `utils`, `struct` |
| Adapter (`device`) | `device/` | ROS device wrappers, device table and factory | `utils`, `struct` |
| Helper (`utils`) | `utils/` | logging grammar, math tables, CSV sink | `struct` |
| Data (`struct`) | `struct/` | data classes and the wire frame contracts | standard library and `numpy` |

`protocol` and `device` are **two adapters of the same height**, one towards the
model side and one towards ROS; neither may import the other, so replacing the
carrier never reaches the device layer. Below them the direction is one way
only - `policy_client -> core -> process -> {protocol, device} -> utils ->
struct` - and is enforced by `test/test_import_direction.py`. `struct` is a pure
data layer and is checked by `test/test_struct_purity.py`.

The Zenoh side is not implemented here: `PolicyMiddleware` holds a
`hex_util_zenoh.HexZenohNode` (composition, not inheritance) and only adds the
three topics, the frame calls and the chunk queue. Install the util package with
`pip install -e <workspace>/pkg/hex_util_zenoh`.

A frame is an agreement with the model side, so its shape is declared in
`protocol/policy_protocol.py`; turning a frame into bytes is
`protocol/frame_codec.py`; moving the bytes is `protocol/policy_middleware.py`.
Field lists are never repeated: the codec reads `__required_keys__` of the frame
type. Changing the wire format therefore touches the first two files only.

Two decisions are tables with a fixed signature, so they can be replaced
without touching their caller: `process/action_restore.py` turns increments into
absolute positions per `action_mode`, and `process/action_average.py` weights the
samples of overlapping chunks (`none`, `exponential`, `plain`).

## Run

```bash
ros2 launch hex_ros_policy_client policy_client.launch.py
```

The device side must be running first: `hex_ros_robot_arm` provides
`<arm>/manip_state` and consumes `<arm>/manip_ctrl`, `hex_ros_camera` provides
`<camera>/image_color_raw`.

## States

| State | Meaning | Leaves on |
|---|---|---|
| `INIT` | Waiting for the first sample of every device | `init_success`, `init_timeout` |
| `RETURN_TO_STABLE` | Interpolating to the stable pose | `stable_arrived`, `stable_timeout` |
| `WAITING` | Holding the pose, ready for a task | `start` |
| `RUNNING` | Publishing observations, executing chunks | `end`, `episode_timeout` |
| `EXIT` | Terminal state, resources released | — |

Every ending converges on `EXIT`, where the resources are released once:
model side `exit`, Ctrl+C, start up timeout and a blocked return to stable
differ only in the recorded reason and the process exit code.

## Exit codes

| Reason | Code |
|---|---|
| `exit` | 0 |
| `init_timeout` | 1 |
| `stable_timeout` | 1 |
| `keyboard_interrupt` | 130 |
| anything else | 1 |

## Extension points

The table lists what has to be touched for each kind of change. Nothing outside
that row needs editing.

| Goal | Files to change |
|---|---|
| Add or remove a device, change a topic, image size, `grip_limit`, `lim_vel`, `lim_acc` or `grav` | `config/ros2/devices.yaml` only |
| Change a scalar (rates, thresholds, topics, interpolation, recorder directory) | `config/ros2/policy_client.yaml` only |
| Support another interpolation | `utils/math_util.py`: one entry in `INTERPOLATORS` |
| Change how observations are assembled | `process/observation_base.py` and `process/history_observation.py` |
| Change how action chunks are integrated | `process/action_process_base.py` and `process/time_integration_chunk.py` |
| Change the wire format of a frame | `protocol/policy_protocol.py` (shape) and `protocol/frame_codec.py` (bytes) |
| Support another action encoding | one entry in `ACTION_RESTORERS` (`process/action_restore.py`) |
| Merge overlapping chunks differently | one entry in `AVERAGERS` (`process/action_average.py`) |
| Change the carrier (Zenoh to something else) | `protocol/policy_middleware.py` only |
| Support another control mode | `struct/cmd_struct.py`: one constant; `utils/arm_device.py`: one row in the mode table |
| Add a device type (new sensor) | new `device/<name>_device.py` subclassing `DeviceBase()` + one entry in `core/device_hub.py: DEVICE_BUILDERS` |
| Add a node for another topology | new `<name>_policy_client.py` in the package root subclassing `PolicyClientBase` (override `_build_components` and `check_conditions`, register its own tables), its YAML and one entry point in `setup.py` |

A single arm needs no code change at all: declare one arm in the device table,
then set `stable_arm_order` and `stable_jnt` accordingly. `test/test_extension.py`
holds a runnable example of both cases.

## Tests

```bash
pytest hex_ros2_ws/src/hex_ros_policy_client/test -v
```

The suite never starts ROS and never touches hardware: the node, the devices and
the transport are replaced by doubles.

## Relation to `hex_policy_bridge`

The old package `hex_policy_bridge` stays in place and keeps working. This
package is the single entry point for new deployments: the device layer talks
ROS topics instead of vendor SDKs, and the node is split into one class per
responsibility.
