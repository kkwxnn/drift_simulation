#!/usr/bin/env python3


from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os
import xacro    
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess

def generate_launch_description():
    
    pkg_description = get_package_share_directory('robot_gazebo')
    pkg_controller = get_package_share_directory('nmpc_controller')

    world_file_name = 'sample.world'
    world = os.path.join(get_package_share_directory(
        'robot_gazebo'), 'world', world_file_name)
    
    declare_world_fname = DeclareLaunchArgument(
        'world_fname', default_value=world, description='absolute path of gazebo world file')
    
    world_fname = LaunchConfiguration('world_fname')

    rviz_path = os.path.join(pkg_description,'rviz','display.rviz')
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
            'world': world_fname,
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

    steering_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["steering_controller", "--controller-manager", "/controller_manager"],
    )

    joint_group_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_group_position_controller", "--controller-manager", "/controller_manager"],
    )

    activate_joint_group_position_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_group_position_controller'], 
        output='screen'
    )

    effort_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["effort_controllers", "--controller-manager", "/controller_manager"],
    )
    

    # velocity_controller_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=["velocity_controllers", "--controller-manager", "/controller_manager"],
    # )

    launch_description = LaunchDescription()
    launch_description.add_action(rviz)
    launch_description.add_action(f1tenth_robot)
    launch_description.add_action(declare_world_fname)
    launch_description.add_action(gazebo)
    launch_description.add_action(spawn_entity)
    launch_description.add_action(joint_state_broadcaster_spawner)
    launch_description.add_action(steering_controller_spawner)
    # launch_description.add_action(joint_group_position_controller_spawner)
    # launch_description.add_action(activate_joint_group_position_controller)
    launch_description.add_action(effort_controller_spawner)
    # launch_description.add_action(velocity_controller_spawner)
    
    return launch_description