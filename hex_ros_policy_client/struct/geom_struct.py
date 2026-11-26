################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Vector3:
    """A point or a vector in three dimensions.

    Attributes:
        x (float): First axis.
        y (float): Second axis.
        z (float): Third axis.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class Quaternion:
    """An orientation as a unit quaternion in ``(w, x, y, z)`` order.

    Attributes:
        w (float): Scalar component, 1.0 for the identity rotation.
        x (float): First vector component.
        y (float): Second vector component.
        z (float): Third vector component.
    """

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class Pose:
    """A position and an orientation.

    Attributes:
        position (Vector3): Translation part.
        orientation (Quaternion): Rotation part.
    """

    position: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)
