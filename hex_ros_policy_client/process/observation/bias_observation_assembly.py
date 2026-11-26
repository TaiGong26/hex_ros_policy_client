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
class BiasAssemblyParam(ObservationFrameAssemblyParam):
    """Parameters of the assembler that follows the biased clock.

    Attributes:
        bias_ns (int): Offset added to the clock to choose the frame instant,
            in nanoseconds; zero asks for the current instant, a negative
            one asks for an instant the samples already cover.
        camera_tolerance_ns (int): Largest distance in nanoseconds between a
            camera stamp and the frame instant; beyond it the frame is
            refused, because the picture would not describe that instant.
    """

    bias_ns: int = 0
    camera_tolerance_ns: int = 50 * NS_PER_MS

    def __post_init__(self) -> None:
        """Reject a camera tolerance that cannot describe a frame.

        Raises:
            ValueError: If the camera tolerance is not positive.
        """
        super().__post_init__()
        if self.camera_tolerance_ns <= 0:
            raise ValueError(
                "camera_tolerance_ns must be positive, "
                f"got {self.camera_tolerance_ns}")


class BiasObservationAssembly(ObservationFrameAssemblyBase):
    """Place every device at the instant the clock reports plus a bias.

    The instant comes from the injected clock, so it does not depend on when
    the observations arrived. A picture cannot be blended, so each one has to
    sit within ``camera_tolerance_ns`` of that instant; the arm positions and
    the end effector are blended from the kept observations.
    """

    def __init__(self, params: BiasAssemblyParam) -> None:
        """Store the bias and the camera tolerance.

        Args:
            params (BiasAssemblyParam): Parameters of the assembler, already
                checked when the parameter object was built.
        """
        super().__init__(params=params)
        self.__bias_ns = int(params.bias_ns)
        self.__camera_tolerance_ns = int(params.camera_tolerance_ns)

    def get_obs(self) -> DeviceObservationFrame | None:
        """Assemble the frame of the biased instant.

        Returns:
            DeviceObservationFrame | None: ``{"stamp_ns", "camera", "arm"}``
            at the biased instant, ``None`` when a camera is missing, a camera
            is too far from that instant, or a device cannot be blended to it.
        """
        latest_observation = self._latest_observation()
        if latest_observation is None:
            return None
        
        target_ns = self._now_ns_fn() + self.__bias_ns
        
        for device_name, camera_sample in latest_observation[
                "camera"].items():
            if camera_sample is None:
                self._logger.logw(
                    f"[obs.assemble/source_missing] whole frame dropped: "
                    f"camera {device_name} has no sample")
                return None
            
            distance_ns = abs(camera_sample.rgb.ts_ns - target_ns)
            if distance_ns > self.__camera_tolerance_ns:
                self._logger.logw(
                    f"[obs.assemble/camera_stale] whole frame dropped: "
                    f"camera {device_name} is "
                    f"{distance_ns / NS_PER_MS:.1f}ms from "
                    f"the frame instant, "
                    f"limit {self.__camera_tolerance_ns / NS_PER_MS:.1f}ms")
                return None
        
        return self._blend_frame_at(target_ns=target_ns)
