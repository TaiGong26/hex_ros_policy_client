#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-26
################################################################
"""One command for the whole dual-arm deployment.

Two arm drivers, the keyboard node and the policy client. Each driver is
pushed into its own namespace, so ``manip_state`` and ``manip_ctrl`` become
``/left/...`` and ``/right/...`` and match the namespaces of the device table.

Cameras are not started here: a RealSense driver publishes the colour stream
the device table expects, so its own launch runs beside this one.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import PythonExpression
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Build the launch description of the whole deployment.

    Returns:
        LaunchDescription: Both arm drivers, the keyboard node and the client.
    """
    arm_pkg_path = FindPackageShare("hex_ros_robot_arm")
    keyboard_pkg_path = FindPackageShare("hex_ros_teleop_keyboard")
    client_pkg_path = FindPackageShare("hex_ros_policy_client")

    robot_type_arg = DeclareLaunchArgument(
        name="robot_type",
        default_value="firefly",
        choices=["archer", "firefly"],
        description="Robot arm type: archer or firefly")
    left_host_arg = DeclareLaunchArgument(
        name="left_robot_host",
        default_value="172.25.0.81",
        description="Left robot controller IP address")
    left_port_arg = DeclareLaunchArgument(
        name="left_robot_port",
        default_value="8439",
        description="Left robot controller WebSocket port")
    left_grip_arg = DeclareLaunchArgument(
        name="left_robot_grip_type",
        default_value="gr100",
        choices=["gp100", "gp80", "gr100", "empty"],
        description="Left robot grip type")
    right_host_arg = DeclareLaunchArgument(
        name="right_robot_host",
        default_value="172.25.0.81",
        description="Right robot controller IP address")
    right_port_arg = DeclareLaunchArgument(
        name="right_robot_port",
        default_value="9439",
        description="Right robot controller WebSocket port")
    right_grip_arg = DeclareLaunchArgument(
        name="right_robot_grip_type",
        default_value="gr100",
        choices=["gp100", "gp80", "gr100", "empty"],
        description="Right robot grip type")
    device_arg = DeclareLaunchArgument(
        name="device_path",
        default_value=PathJoinSubstitution(
            [client_pkg_path, "config/ros2/dual_manip_devices.yaml"]),
        description="Device table path, defaults to the packaged table")
    recorder_arg = DeclareLaunchArgument(
        name="recorder_dir",
        default_value="",
        description="Directory of the recorder files, empty disables it")

    robot_launch_file = PythonExpression(
        ['"', LaunchConfiguration("robot_type"), '.launch.py"'])

    def arm_driver(host, port, grip):
        """Include one arm driver.

        Args:
            host (LaunchConfiguration): Controller address of this arm.
            port (LaunchConfiguration): Controller port of this arm.
            grip (LaunchConfiguration): Gripper model of this arm.

        Returns:
            IncludeLaunchDescription: Driver launch of one arm.
        """
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([arm_pkg_path, robot_launch_file])),
            launch_arguments={
                "robot_host": host,
                "robot_port": port,
                "robot_grip_type": grip,
                "test": "false",
            }.items(),
        )

    keyboard_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [keyboard_pkg_path, "teleop_keyboard.launch.py"])),
    )
    client_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [client_pkg_path, "policy_client.launch.py"])),
        launch_arguments={
            "device_path": LaunchConfiguration("device_path"),
            "recorder_dir": LaunchConfiguration("recorder_dir"),
        }.items(),
    )

    return LaunchDescription([
        robot_type_arg,
        left_host_arg,
        left_port_arg,
        left_grip_arg,
        right_host_arg,
        right_port_arg,
        right_grip_arg,
        device_arg,
        recorder_arg,
        GroupAction([
            PushRosNamespace("left"),
            arm_driver(
                host=LaunchConfiguration("left_robot_host"),
                port=LaunchConfiguration("left_robot_port"),
                grip=LaunchConfiguration("left_robot_grip_type"),
            ),
        ]),
        GroupAction([
            PushRosNamespace("right"),
            arm_driver(
                host=LaunchConfiguration("right_robot_host"),
                port=LaunchConfiguration("right_robot_port"),
                grip=LaunchConfiguration("right_robot_grip_type"),
            ),
        ]),
        keyboard_launch,
        client_launch,
    ])
