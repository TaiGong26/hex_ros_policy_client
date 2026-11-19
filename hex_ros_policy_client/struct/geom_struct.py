################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Geometric primitives shared by the device layer and the action layer.

The primitives carry no behaviour and no dependency: they are plain frozen
dataclasses, so they can be used by every layer without breaking the single
dependency direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Vector3:
    """A point or a vector in three dimensions.

    Attributes:
        x (float): First axis, metres for positions.
        y (float): Second axis, metres for positions.
        z (float): Third axis, metres for positions.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class Quaternion:
    """An orientation as a unit quaternion.

    Attributes:
        x (float): First vector component.
        y (float): Second vector component.
        z (float): Third vector component.
        w (float): Scalar component, 1.0 for the identity rotation.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass(frozen=True)
class Pose:
    """A position and an orientation.

    The default factory keeps every instance independent, so mutating a nested
    value of one pose cannot leak into another.

    Attributes:
        position (Vector3): Translation part.
        orientation (Quaternion): Rotation part.
    """

    position: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)
