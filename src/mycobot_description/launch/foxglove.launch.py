#!/usr/bin/env python3
"""
Launch file to display MyCobotPro 630 robot in Foxglove Studio.

Usage:
    ros2 launch mycobot_description foxglove.launch.py

Then open Foxglove Studio and connect to ws://localhost:8765
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('mycobot_description')

    # Paths
    urdf_file = os.path.join(pkg_dir, 'urdf', 'mycobot_pro_630.urdf')

    # Read URDF file
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    port = LaunchConfiguration('port')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock'
    )

    declare_port = DeclareLaunchArgument(
        'port',
        default_value='8765',
        description='Foxglove bridge WebSocket port'
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time
        }]
    )

    # Joint State Publisher GUI
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    # Foxglove Bridge
    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': 8765,
            'address': '0.0.0.0',
            'tls': False,
            'send_buffer_limit': 10000000,
            'use_sim_time': use_sim_time,
        }]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_port,
        robot_state_publisher,
        joint_state_publisher_gui,
        foxglove_bridge
    ])
