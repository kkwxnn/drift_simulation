#!/usr/bin/env python3


from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os
import xacro    
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    
    pkg = get_package_share_directory('robot_gazebo')
    rviz_path = os.path.join(pkg,'rviz','display.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_path],
        output='screen')
    
    f1tenth_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("robot_description"),
                    "launch",
                    "f1tenth_robot.launch.py"
                )
            ]
        )
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("gazebo_ros"),
                    "launch",
                    "gazebo.launch.py"
                )
            ]
        ),
        launch_arguments={
            'pause': 'true'
        }.items()
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "f1tenth",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.12" 
        ],
        output = "screen"
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["velocity_controllers", "--controller-manager", "/controller_manager"],
    )

    launch_description = LaunchDescription()
    launch_description.add_action(rviz)
    launch_description.add_action(f1tenth_robot)
    launch_description.add_action(gazebo)
    launch_description.add_action(spawn_entity)
    launch_description.add_action(joint_state_broadcaster_spawner)
    launch_description.add_action(robot_controller_spawner)
    
    return launch_description