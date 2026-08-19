from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    world_file_path = PathJoinSubstitution([
        FindPackageShare("b2w_sim_worlds"),
        "worlds",
        LaunchConfiguration("world_file"),
    ])

    bridge_config_path = PathJoinSubstitution([
        FindPackageShare("scout_description"),
        "launch",
        "scout_gz_bridge.yaml",
    ])

    declared_arguments = [
        DeclareLaunchArgument("world_file", default_value="ISAACLAB_TRAIN.world"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.5"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("paused", default_value="false"),
        DeclareLaunchArgument("verbose", default_value="false"),
    ]

    load_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("scout_description"), "launch", "scout_base_description.launch.py"
            ])
        ]),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])
        ),
        launch_arguments={
            "gz_args": [
                IfElseSubstitution(LaunchConfiguration("paused"), if_value="", else_value="-r "),
                IfElseSubstitution(LaunchConfiguration("verbose"), if_value="-v4 ", else_value=""),
                "--headless-rendering ",
                world_file_path,
            ],
            "on_exit_shutdown": "true",
        }.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "robot_description",
            "-name", "scout_mini",
            "-x", LaunchConfiguration("x"),
            "-y", LaunchConfiguration("y"),
            "-z", LaunchConfiguration("z"),
            "-Y", LaunchConfiguration("yaw"),
        ],
        output="screen",
    )

    ros_gz_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ros_gz_bridge"), "launch", "ros_gz_bridge.launch.py"
            ])
        ),
        launch_arguments={
            "bridge_name": "scout_gz_bridge",
            "config_file": bridge_config_path,
        }.items(),
    )

    return LaunchDescription(declared_arguments + [load_launch, gz_sim, spawn_robot, ros_gz_bridge])
