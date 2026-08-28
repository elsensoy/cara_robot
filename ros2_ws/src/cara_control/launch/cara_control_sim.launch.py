# cara_control_sim.launch.py
#   ros2 launch cara_control cara_control_sim.launch.py
#   ros2 launch cara_control cara_control_sim.launch.py source:=hw
#
# Publishes the actuator-health signal path. Observation-only (no servo commands),
# safe to run alongside cara_stack.launch.py.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="sim",
                              description="sim | hw"),
        # This launch is for watching the sim demo, so it defaults to the
        # internal gait. Set to /joint_commands to track a real JointTrajectory.
        DeclareLaunchArgument("setpoint_topic", default_value="",
                              description="JointTrajectory topic to track; "
                                          "'' = internal demo gait."),
        Node(
            package="cara_control",
            executable="cara_control_node",
            name="cara_control_node",
            output="screen",
            parameters=[{
                "source": LaunchConfiguration("source"),
                "rate_hz": 50.0,
                "i2c_bus": 1,
                "ina_addr": 0x45,   # servo rail, per tests/cara_power_monitor.py
                "imu_addr": 0x28,
                "setpoint_topic": LaunchConfiguration("setpoint_topic"),
                "setpoint_timeout_s": 0.5,
            }],
        ),
    ])
