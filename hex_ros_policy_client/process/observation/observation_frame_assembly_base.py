################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ...struct.device_struct import (
    ArmObservation,
    ArmState,
    CameraObservation,
    DeviceObservationFrame,
    GripState,
    ImageFrame,
)
from ...struct.geom_struct import Pose, Quaternion, Vector3
from ...utils import linear_interpolate, pose_interpolate


@dataclass(frozen=True)
class ObservationFrameAssemblyParam:
    """Hold what every observation frame assembler needs at run time.

    Attributes:
        now_fn (Callable[[], int]): Clock injected by the node, returning the
            current time in nanoseconds.
        queue_capacity (int): How many pushed observations are kept for the
            blending; the oldest one leaves first, and two are needed to
            bracket one instant.
        logger (Any | None): Logger that receives the refusal lines.
    """

    now_fn: Callable[[], int]
    queue_capacity: int = 50
    logger: Any | None = None

    def __post_init__(self) -> None:
        """Reject a history that cannot hold a pair of observations.

        Raises:
            ValueError: If the capacity is smaller than two.
        """
        if self.queue_capacity < 2:
            raise ValueError(
                "queue_capacity must be at least 2, "
                f"got {self.queue_capacity}")


class ObservationFrameAssemblyBase(ABC):
    """Assemble the frame that the protocol layer publishes.

    A frame is one stamp plus one sample of every declared device at that
    instant. The node pushes the newest observations and asks for a frame at
    its own rate; this class keeps those observations and places every
    device at the instant a subclass asks for.

    Samples are copied on the way in and on the way out, so neither the device
    that produced one nor the caller that received a frame can change the kept
    history.

    Choosing the instant and refusing a frame are policy and stay in the
    subclass; keeping the observations and blending them happens once, here.
    """

    def __init__(self, params: ObservationFrameAssemblyParam) -> None:
        """Store the clock, the logger and an empty history.

        Args:
            params (ObservationFrameAssemblyParam): Parameters of the
                assembler, already checked when the parameter object was
                built.
        """
        self._now_ns_fn = params.now_fn
        self._logger = params.logger
        self.__history: deque[dict] = deque(maxlen=int(params.queue_capacity))

    def push_obs(self, observation: dict) -> None:
        """Keep one observation so a later frame can be blended from it.

        Every sample is copied on the way in, so a device that keeps writing
        into its own buffers cannot change the kept history afterwards.

        Args:
            observation (dict): Output of ``Robot.read_obs()``, grouped under
                ``camera`` and ``arm`` and keyed by device name.
        """
        kept_observation: dict = {"camera": {}, "arm": {}}
        for device_name, camera_sample in observation["camera"].items():
            kept_observation["camera"][device_name] = _copy_camera(
                camera_sample=camera_sample)
        for device_name, arm_sample in observation["arm"].items():
            kept_observation["arm"][device_name] = _copy_arm(
                arm_sample=arm_sample)
        self.__history.append(kept_observation)

    def reset(self) -> None:
        """Forget the kept observations so a new task starts empty."""
        self.__history.clear()

    @abstractmethod
    def get_obs(self) -> DeviceObservationFrame | None:
        """Assemble the frame of the instant this assembler works on.

        Returns:
            DeviceObservationFrame | None: ``{"stamp_ns", "camera", "arm"}``
            when every device can be placed at that instant, ``None`` when the
            frame is refused.
        """

    def _latest_observation(self) -> dict | None:
        """Return the newest kept observation.

        Returns:
            dict | None: Output of ``Robot.read_obs()``, ``None`` before the
            first one arrives.
        """
        return self.__history[-1] if self.__history else None

    def _blend_frame_at(self, target_ns: int) -> DeviceObservationFrame | None:
        """Place every device at one instant and return the frame.

        Every array of the frame is new, so a caller cannot reach the kept
        observations through it. Images are not blended, a picture cannot be
        interpolated, so the newest one is copied. Arm positions and the end
        effector pose are blended; velocities and efforts are copied from the
        older sample.

        Args:
            target_ns (int): Instant the frame describes, in nanoseconds.

        Returns:
            DeviceObservationFrame | None: ``{"stamp_ns", "camera", "arm"}``
            at ``target_ns``,
            ``None`` when an arm has no sample, or no pair of samples that
            brackets the instant.
        """
        latest_observation = self._latest_observation()
        if latest_observation is None:
            return None
        assembled_frame = DeviceObservationFrame(
            stamp_ns=target_ns,
            camera={},
            arm={},
        )
        for device_name, camera_sample in latest_observation[
                "camera"].items():
            assembled_frame["camera"][device_name] = _copy_camera(
                camera_sample=camera_sample)
        for device_name, arm_sample in latest_observation["arm"].items():
            if arm_sample is None or arm_sample.arm_state is None:
                self._logger.logw(
                    f"[obs.assemble/source_missing] whole frame dropped: "
                    f"arm {device_name} has no sample")
                return None
            bracketing_pair = self.__find_bracketing_pair(
                device_name=device_name,
                target_ns=target_ns,
            )
            if bracketing_pair is None:
                self._logger.logw(
                    f"[obs.assemble/source_missing] whole frame dropped: "
                    f"arm {device_name} has no pair of samples around "
                    f"target_ns={target_ns}")
                return None
            assembled_frame["arm"][device_name] = _blend_arm(
                bracketing_pair=bracketing_pair,
                target_ns=target_ns,
            )
        return assembled_frame

    def __find_bracketing_pair(
            self, device_name: str,
            target_ns: int) -> tuple[ArmObservation,
                                     ArmObservation] | None:
        """Find the two kept samples that the instant falls between.

        Args:
            device_name (str): Arm name of the topology.
            target_ns (int): Instant the frame describes, in nanoseconds.

        Returns:
            tuple | None: Older and newer ``ArmObservation`` of that arm, the
            older one stamped at or before ``target_ns`` and the newer one
            after it; ``None`` when no such pair was kept.
        """
        older_sample = None
        newer_sample = None
        for kept_observation in self.__history:
            arm_sample = kept_observation["arm"].get(device_name)
            if arm_sample is None or arm_sample.stamp_ns is None:
                continue
            if arm_sample.stamp_ns <= target_ns:
                older_sample = arm_sample
            elif newer_sample is None:
                newer_sample = arm_sample
        if older_sample is None or newer_sample is None:
            return None
        return older_sample, newer_sample


def _blend_arm(bracketing_pair: tuple[ArmObservation, ArmObservation],
               target_ns: int) -> ArmObservation:
    """Place one arm of the topology at the instant.

    Args:
        bracketing_pair (tuple): Older and newer ``ArmObservation`` of the
            same arm, the older one stamped at or before ``target_ns``.
        target_ns (int): Instant the frame describes, in nanoseconds.

    Returns:
        ArmObservation: Sample stamped ``target_ns``; positions and the end
        effector pose are blended, velocities and efforts are copied from the
        older sample.
    """
    older_sample, newer_sample = bracketing_pair
    elapsed_ns = target_ns - older_sample.stamp_ns
    duration_ns = newer_sample.stamp_ns - older_sample.stamp_ns
    blended_arm_state = None
    if newer_sample.arm_state is not None:
        blended_arm_state = ArmState(
            arm_pos=linear_interpolate(
                start=older_sample.arm_state.arm_pos,
                target=newer_sample.arm_state.arm_pos,
                elapsed_ns=elapsed_ns,
                duration_ns=duration_ns,
            ),
            arm_vel=older_sample.arm_state.arm_vel.copy(),
            arm_eff=older_sample.arm_state.arm_eff.copy(),
            ee_base=_blend_pose(
                older_pose=older_sample.arm_state.ee_base,
                newer_pose=newer_sample.arm_state.ee_base,
                elapsed_ns=elapsed_ns,
                duration_ns=duration_ns,
            ),
        )
    blended_grip_state = None
    if (older_sample.grip_state is not None
            and newer_sample.grip_state is not None):
        blended_grip_state = GripState(
            grip_pos=linear_interpolate(
                start=older_sample.grip_state.grip_pos,
                target=newer_sample.grip_state.grip_pos,
                elapsed_ns=elapsed_ns,
                duration_ns=duration_ns,
            ),
            grip_vel=older_sample.grip_state.grip_vel.copy(),
            grip_eff=older_sample.grip_state.grip_eff.copy(),
        )
    return ArmObservation(
        stamp_ns=target_ns,
        arm_state=blended_arm_state,
        grip_state=blended_grip_state,
    )


def _copy_camera(
        camera_sample: CameraObservation | None
) -> CameraObservation | None:
    """Copy one camera sample so the frame owns its buffers.

    Args:
        camera_sample (CameraObservation | None): Sample kept in the history.

    Returns:
        CameraObservation | None: Sample whose images are new arrays, ``None``
        when there was no sample.
    """
    if camera_sample is None:
        return None
    return CameraObservation(
        rgb=_copy_image(image=camera_sample.rgb),
        depth=None if camera_sample.depth is None
        else _copy_image(image=camera_sample.depth),
    )


def _copy_arm(arm_sample: ArmObservation | None) -> ArmObservation | None:
    """Copy one arm sample so the history owns its arrays.

    The end effector pose is shared, because it holds plain floats and no
    array that a later write could reach.

    Args:
        arm_sample (ArmObservation | None): Sample of one arm.

    Returns:
        ArmObservation | None: Sample whose arrays are new, ``None`` when there
        was no sample.
    """
    if arm_sample is None:
        return None
    arm_state = None
    if arm_sample.arm_state is not None:
        arm_state = ArmState(
            arm_pos=arm_sample.arm_state.arm_pos.copy(),
            arm_vel=arm_sample.arm_state.arm_vel.copy(),
            arm_eff=arm_sample.arm_state.arm_eff.copy(),
            ee_base=arm_sample.arm_state.ee_base,
        )
    grip_state = None
    if arm_sample.grip_state is not None:
        grip_state = GripState(
            grip_pos=arm_sample.grip_state.grip_pos.copy(),
            grip_vel=arm_sample.grip_state.grip_vel.copy(),
            grip_eff=arm_sample.grip_state.grip_eff.copy(),
        )
    return ArmObservation(
        stamp_ns=arm_sample.stamp_ns,
        arm_state=arm_state,
        grip_state=grip_state,
    )


def _copy_image(image: ImageFrame) -> ImageFrame:
    """Copy one image so the frame owns its pixel buffer.

    Args:
        image (ImageFrame): Image kept in the history.

    Returns:
        ImageFrame: Same stamp, encoding and geometry, with a new buffer.
    """
    return ImageFrame(
        ts_ns=image.ts_ns,
        encoding=image.encoding,
        width=image.width,
        height=image.height,
        data=image.data.copy(),
    )


def _blend_pose(older_pose: Pose, newer_pose: Pose, elapsed_ns: int,
                duration_ns: int) -> Pose:
    """Place one end effector pose at the instant.

    The position moves in proportion; the orientation follows the shortest
    arc, so the result is a unit quaternion in ``(w, x, y, z)`` order.

    Args:
        older_pose (Pose): Pose of the older sample.
        newer_pose (Pose): Pose of the newer sample.
        elapsed_ns (int): Nanoseconds from the older sample to the instant.
        duration_ns (int): Nanoseconds between the two samples.

    Returns:
        Pose: Pose at the instant.
    """
    blended_values = pose_interpolate(
        start=_flatten_pose(pose=older_pose),
        target=_flatten_pose(pose=newer_pose),
        elapsed_ns=elapsed_ns,
        duration_ns=duration_ns,
    )
    return Pose(
        position=Vector3(
            x=float(blended_values[0]),
            y=float(blended_values[1]),
            z=float(blended_values[2]),
        ),
        orientation=Quaternion(
            w=float(blended_values[3]),
            x=float(blended_values[4]),
            y=float(blended_values[5]),
            z=float(blended_values[6]),
        ),
    )


def _flatten_pose(pose: Pose) -> np.ndarray:
    """Read one pose as the seven numbers the interpolation works on.

    Args:
        pose (Pose): Position and orientation of the end effector.

    Returns:
        np.ndarray: Seven values ordered ``(x, y, z, qw, qx, qy, qz)``.
    """
    position = pose.position
    orientation = pose.orientation
    return np.array([position.x, position.y, position.z,
                     orientation.w, orientation.x,
                     orientation.y, orientation.z], dtype=np.float64)
