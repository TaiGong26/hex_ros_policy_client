################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from dataclasses import dataclass

from ...struct.device_struct import DeviceObservationFrame
from .observation_frame_assembly_base import (
    ObservationFrameAssemblyBase,
    ObservationFrameAssemblyParam,
)

NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class PrimaryAssemblyParam(ObservationFrameAssemblyParam):
    """Parameters of the assembler that follows the primary camera.

    Attributes:
        primary_camera (str): Camera name whose stamp becomes the frame
            instant; it has to be one of the declared cameras.
        camera_skew_ns (int): Largest distance in nanoseconds between another
            camera stamp and the primary one; beyond it the frame is
            refused, because the pictures would not describe one instant.
    """

    primary_camera: str = ""
    camera_skew_ns: int = 50 * NS_PER_MS

    def __post_init__(self) -> None:
        """Reject a missing primary camera or a bad skew limit.

        Raises:
            ValueError: If the primary camera name is empty or the skew
                limit is not positive.
        """
        super().__post_init__()
        if not self.primary_camera:
            raise ValueError("primary_camera must be a camera name")
        if self.camera_skew_ns <= 0:
            raise ValueError(
                f"camera_skew_ns must be positive, got {self.camera_skew_ns}")


class PrimaryObservationAssembly(ObservationFrameAssemblyBase):
    """Place every device at the instant the primary camera reports.

    The primary picture is used as it is, so its stamp is the instant of the
    frame and no clock is read. Every other picture has to sit within
    ``camera_skew_ns`` of it, and the arm positions and the end effector are
    blended from the kept observations.
    """

    def __init__(self, params: PrimaryAssemblyParam) -> None:
        """Store the primary camera and the camera skew limit.

        Args:
            params (PrimaryAssemblyParam): Parameters of the assembler,
                already checked when the parameter object was built.
        """
        super().__init__(params=params)
        self.__primary_camera = params.primary_camera
        self.__camera_skew_ns = int(params.camera_skew_ns)

    def get_obs(self) -> DeviceObservationFrame | None:
        """Assemble the frame of the primary camera instant.

        Returns:
            DeviceObservationFrame | None: ``{"stamp_ns", "camera", "arm"}``
            at the instant of the primary picture, ``None`` when a camera is
            missing, another camera is too far from the primary one, or a
            device cannot be blended to that instant.
        """
        latest_observation = self._latest_observation()
        if latest_observation is None:
            return None
        
        primary_sample = latest_observation["camera"].get(
            self.__primary_camera)
        if primary_sample is None:
            self._logger.logw(
                f"[obs.assemble/source_missing] whole frame dropped: "
                f"camera {self.__primary_camera} has no sample")
            return None
        
        target_ns = primary_sample.rgb.ts_ns
        for device_name, camera_sample in latest_observation[
                "camera"].items():
            if device_name == self.__primary_camera:
                continue
            
            if camera_sample is None:
                self._logger.logw(
                    f"[obs.assemble/source_missing] whole frame dropped: "
                    f"camera {device_name} has no sample")
                return None
            
            skew_ns = abs(camera_sample.rgb.ts_ns - target_ns)
            if skew_ns > self.__camera_skew_ns:
                self._logger.logw(
                    f"[obs.assemble/camera_skew] whole frame dropped: "
                    f"camera {device_name} is "
                    f"{skew_ns / NS_PER_MS:.1f}ms from "
                    f"primary {self.__primary_camera}, "
                    f"limit {self.__camera_skew_ns / NS_PER_MS:.1f}ms")
                return None
        
        return self._blend_frame_at(target_ns=target_ns)
