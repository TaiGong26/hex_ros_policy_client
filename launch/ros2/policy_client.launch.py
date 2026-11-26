#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Launch file of the policy client.

Only the parameter file and the device table path are passed here; everything
else comes from the two YAML files.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Build the launch description of the node.

    Returns:
        LaunchDescription: Node with its parameter file and device table.
    """
    package_name = "hex_ros_policy_client"
    pkg_share = FindPackageShare(package_name).find(package_name)
    param_path = pkg_share + "/config/ros2/dual_manip_policy_client.yaml"
    device_path = pkg_share + "/config/ros2/dual_manip_devices.yaml"

    device_arg = DeclareLaunchArgument(
        name="device_path",
        default_value=device_path,
        description="Path of the device table declared as YAML")

    recorder_arg = DeclareLaunchArgument(
        name="recorder_dir",
        default_value="",
        description="Directory of the recorder files, empty disables it")

    policy_client = Node(
        package=package_name,
        executable="dual_manip_policy_client",
        name="dual_manip_policy_client",
        output="screen",
        parameters=[
            param_path,
            {
                "device_path": LaunchConfiguration("device_path"),
                "recorder_dir": ParameterValue(
                    LaunchConfiguration("recorder_dir"), value_type=str),
            },
        ],
    )

    return LaunchDescription([device_arg, recorder_arg, policy_client])
