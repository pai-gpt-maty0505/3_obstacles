#!/usr/bin/env python3
"""Launch the project world, a clean SO-101 model, and optional MoveIt.

This is a project-owned composition of public ROS 2 packages.  It does not
modify or include launch code from any prior project.  Starting this launch
only creates simulation processes; it never sends a trajectory goal.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_project_root() -> str:
    """Find the checkout when installed with ``--symlink-install``."""
    return str(Path(__file__).resolve().parents[4])


def generate_launch_description() -> LaunchDescription:
    start_moveit = LaunchConfiguration("start_moveit")
    moveit_delay = LaunchConfiguration("moveit_delay")
    controller_delay = LaunchConfiguration("controller_delay")
    moveit_use_gazebo_geometry = LaunchConfiguration("moveit_use_gazebo_geometry")

    project_root_value = os.environ.get("THREE_OBSTACLES_PROJECT_ROOT", _default_project_root())
    model_path = os.path.join(project_root_value, "three_obstacles", "gazebo", "models")
    world_path = os.path.join(project_root_value, "three_obstacles", "gazebo", "worlds", "three_obstacles_baseline.world")
    inherited_resources = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_path = model_path + (os.pathsep + inherited_resources if inherited_resources else "")

    description_share = get_package_share_directory("so101_description")
    moveit_share = get_package_share_directory("so101_moveit_config")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(description_share, "launch", "robot_state_publisher.launch.py")
        ),
        launch_arguments={
            "robot_name": "so101",
            "use_camera": "false",
            "use_gazebo": "true",
            "use_rviz": "false",
            "use_jsp": "false",
            "jsp_gui": "false",
            "use_sim_time": "true",
        }.items(),
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": [" -r -s -v 3 ", world_path]}.items(),
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "/robot_description", "-name", "so101", "-allow_renaming",
            "-x", "0.0", "-y", "0.0", "-z", "0.0", "-R", "0.0", "-P", "0.0", "-Y", "0.0",
        ],
        output="screen",
    )

    controller_loader = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit_share, "launch", "load_ros2_controllers.launch.py")),
        # The upstream controller loader owns the wait.  Keeping its delay
        # inside that launch avoids starting controller CLI calls before the
        # spawned robot's ros2_control manager has initialized.
        launch_arguments={"controller_load_delay": controller_delay}.items(),
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit_share, "launch", "move_group.launch.py")),
        launch_arguments={
            "use_rviz": "false",
            "use_camera": "false",
            # Gazebo retains its fast physical contact geometry through the
            # robot_state_publisher above. MoveIt instead defaults to the
            # upstream detailed mesh collision geometry. This prevents a
            # false self-collision from the coarse Gazebo boxes; it is not an
            # AllowedCollisionMatrix or collision-disable workaround.
            "use_gazebo": moveit_use_gazebo_geometry,
            "use_sim_time": "true",
        }.items(),
        condition=IfCondition(start_moveit),
    )

    return LaunchDescription([
        DeclareLaunchArgument("start_moveit", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("controller_delay", default_value="8.0"),
        DeclareLaunchArgument("moveit_delay", default_value="14.0"),
        DeclareLaunchArgument(
            "moveit_use_gazebo_geometry",
            default_value="false",
            choices=["true", "false"],
            description="Use coarse Gazebo collision boxes in MoveIt; false uses detailed upstream meshes.",
        ),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),
        robot_state_publisher,
        gazebo,
        clock_bridge,
        TimerAction(period=2.0, actions=[spawn_robot]),
        controller_loader,
        TimerAction(period=moveit_delay, actions=[moveit]),
    ])
