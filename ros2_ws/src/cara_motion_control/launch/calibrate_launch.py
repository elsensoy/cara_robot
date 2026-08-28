import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('cara_motion_control')
    urdf_file = os.path.join(pkg_share, 'urdf', 'cara.urdf')

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        # 1. State Publisher (to see the robot in RViz)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}]
        ),
        # 2. Actuator Node (Directly talks to PCA9685)
        Node(
            package='cara_motion_control',
            executable='cara_actuator_node',
            name='cara_actuator_node'
        ),
        # 3. Head Node (Talks to Arduino)
        Node(
            package='cara_motion_control',
            executable='head_expression_node',
            name='head_expression_node',
            parameters=[{'arduino_port': '/dev/ttyAMA0'}]
        )
    ])
