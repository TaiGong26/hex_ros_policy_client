################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..struct.device_struct import DeviceObservationFrame
from hex_util_zenoh.proto_util import (
    ArmPayload,
    DualArmActionManipFrame,
    ImagePayload,
    InferenceFrame,
    ObservationFrame,
    check_frame,
    ns_to_stamp,
    stamp_to_ns,
)


def model_name(device_name: str,
               name_map: Mapping[str, str] | None = None) -> str:
    """Return the field name the model uses for a device.

    Args:
        device_name (str): Name declared in the device table.
        name_map (Mapping | None): Device name to model field name; ``None`` or
            a missing entry means identity.

    Returns:
        str: Mapped name, or the device name itself.
    """
    if name_map is None:
        return device_name
    return name_map.get(device_name, device_name)


def build_observation_frame(
        observation: DeviceObservationFrame,
        name_map: Mapping[str, str] | None = None) -> ObservationFrame:
    """Build one observation frame from the output of the assembler.

    Args:
        observation (DeviceObservationFrame): Output of
            ``ObservationBase.get_obs()``.
        name_map (Mapping | None): Device name to model field name.

    Returns:
        ObservationFrame: Frame ready for packing.
    """
    cameras = {}
    for name, camera in observation["camera"].items():
        cameras[model_name(
            device_name=name,
            name_map=name_map,
        )] = {
            "rgb": _image_payload(frame=camera.rgb),
            "depth": (None if camera.depth is None
                      else _image_payload(frame=camera.depth)),
        }
    arms = {}
    for name, arm in observation["arm"].items():
        arms[model_name(
            device_name=name,
            name_map=name_map,
        )] = _arm_payload(observation=arm)
    return {
        "header": ns_to_stamp(stamp_ns=observation["stamp_ns"]),
        "camera": cameras,
        "arm": arms,
    }


def build_inference_frame(enabled: bool) -> InferenceFrame:
    """Build the inference enable frame.

    Args:
        enabled (bool): Whether the robot accepts actions.

    Returns:
        InferenceFrame: Frame ready for packing.
    """
    return {"inference_enabled": bool(enabled)}


def decode_action_chunk(frame: dict) -> dict:
    """Convert one action chunk frame.

    Only two things happen here: the keys are checked against the frame type,
    and the scalar fields are converted to the types of this package. Both mode
    strings and both device groups travel as the sender wrote them, because
    whether a mode is supported and whether a step matrix has a usable shape
    are questions of the action layer, the only place that knows what it can do
    with them.

    Args:
        frame (dict): Decoded frame received from the model side.

    Returns:
        dict: ``stamp_ns``, ``chunk_step_interval_ms``, ``action_mode``,
        ``arm_ctrl_mode`` and the ``arm`` / ``grip`` mappings as received.

    Raises:
        ValueError: If a key the frame type declares is missing.
    """
    check_frame(
        frame=frame,
        frame_type=DualArmActionManipFrame,
    )
    return {
        "stamp_ns": stamp_to_ns(stamp=frame["header"]),
        "chunk_step_interval_ms": int(frame["chunk_step_interval_ms"]),
        "action_mode": str(frame["action_mode"]),
        "arm_ctrl_mode": str(frame["arm_ctrl_mode"]),
        "arm": frame["arm"],
        "grip": frame["grip"],
    }


def _image_payload(frame: Any) -> ImagePayload:
    """Convert one image to its wire payload.

    Args:
        frame: ``ImageFrame`` of the data layer.

    Returns:
        ImagePayload: Wire payload.
    """
    return {
        "encoding": frame.encoding,
        "width": int(frame.width),
        "height": int(frame.height),
        "data": np.asarray(frame.data, dtype=np.uint8).tobytes(),
    }


# Wire key to the data layer attribute that fills it, one table per state
# object, so adding a vector needs one row and nothing else.


ARM_VECTOR_FIELDS = {
    "pos": "arm_pos",
    "vel": "arm_vel",
    "eff": "arm_eff",
}


GRIP_VECTOR_FIELDS = {
    "grip_pos": "grip_pos",
    "grip_vel": "grip_vel",
    "grip_eff": "grip_eff",
}


def _arm_payload(observation: Any) -> ArmPayload:
    """Convert one arm observation to its wire payload.

    Args:
        observation: ``ArmObservation`` of the data layer.

    Returns:
        ArmPayload: Wire payload; a part the device never received becomes an
        empty list, which the field type allows.
    """
    arm = observation.arm_state
    grip = observation.grip_state
    payload = {
        "pose": _pose_vector(pose=None if arm is None else arm.ee_base),
    }
    for key, attribute in ARM_VECTOR_FIELDS.items():
        payload[key] = _vector(
            state=arm,
            attribute=attribute,
        )
    for key, attribute in GRIP_VECTOR_FIELDS.items():
        payload[key] = _vector(
            state=grip,
            attribute=attribute,
        )
    return payload


def _vector(state: Any, attribute: str) -> list[float]:
    """Read one vector of a state object, tolerating a missing state.

    Args:
        state: ``ArmState``, ``GripState`` or ``None``.
        attribute (str): Name of the vector on that state.

    Returns:
        list: Values as floats, empty when the state or the vector is missing.
    """
    if state is None:
        return []
    values = getattr(state, attribute)
    if values is None:
        return []
    return np.asarray(values).ravel().tolist()


def _pose_vector(pose: Any) -> list[float]:
    """Flatten a pose into position plus quaternion.

    Args:
        pose: ``Pose`` of the data layer, or ``None``.

    Returns:
        list: Seven floats ``(px, py, pz, qw, qx, qy, qz)``, empty when the
        pose is missing.
    """
    if pose is None:
        return []
    orientation = pose.orientation
    return [
        pose.position.x,
        pose.position.y,
        pose.position.z,
        orientation.w,
        orientation.x,
        orientation.y,
        orientation.z,
    ]
