import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Paths and Files
    pkg_share = get_package_share_directory('cara_motion_control')
    urdf_file = os.path.join(pkg_share, 'urdf', 'cara.urdf')

    # Read URDF content to pass to Robot State Publisher
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # 2. Node: Robot State Publisher (Publishes Cara's Skeleton)
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    # 3. Node: Body Node (The RL Policy "Brain")
    body_node = Node(
        package='cara_motion_control',
        executable='cara_body_node',
        name='cara_body_node'
    )

    # 4. Node: Actuator Node (The PCA9685 "Nervous System")
    actuator_node = Node(
        package='cara_motion_control',
        executable='cara_actuator_node',
        name='cara_actuator_node'
    )

    # 5. Node: Head Expression Node (The Arduino "Face")
    head_node = Node(
        package='cara_motion_control',
        executable='head_expression_node',
        name='head_expression_node',
        parameters=[{'use_jetson_gpio': False, 'arduino_port': '/dev/ttyAMA0', 'arduino_baud':'9600'}]
    )
# Inside generate_launch_description in cara_launch.py

    health_monitor_node = Node(
    	package='cara_motion_control',
    	executable='cara_health_monitor',
   	 name='cara_health_monitor'
    )

    return LaunchDescription([
        robot_state_pub,
        body_node,
        actuator_node,
        head_node,
        health_monitor_node
    ])
