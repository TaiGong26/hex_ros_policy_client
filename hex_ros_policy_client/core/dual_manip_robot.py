################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Composite robot for the two arm topology.

No topology is coded here: the arm and camera names come from the device table,
so this class drives one arm, two arms or more without a branch. All behaviour
lives in the base class, which is why this file only names the concrete type.
"""

from __future__ import annotations

from .composite_robot_base import CompositeRobotBase


class DualManipRobot(CompositeRobotBase):
    """Robot assembled from a device table with manipulators and cameras.

    Attributes:
        device_path (str): Path of the device table.
    """
