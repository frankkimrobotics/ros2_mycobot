#!/usr/bin/env python3
"""
Launch file to display MyCobotPro 630 robot in RViz2.

Usage:
    ros2 launch mycobot_description display.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('mycobot_description')

    # Paths
    urdf_file = os.path.join(pkg_dir, 'urdf', 'mycobot_pro_630.urdf')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'display.rviz')

    # Read URDF file
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gui = LaunchConfiguration('use_gui')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock'
    )

    declare_use_gui = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Use joint_state_publisher_gui'
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

    # Joint State Publisher GUI (disabled when use_gui:=false, e.g. when using external planner)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
        condition=IfCondition(use_gui),
    )

    # RViz2
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_use_gui,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2
    ])
