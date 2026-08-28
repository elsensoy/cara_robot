import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('cara_motion_control')
    # Make sure this matches your actual URDF filename
    urdf_file = os.path.join(pkg_share, 'urdf', 'cara.urdf')

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        # 1. Publishes the URDF to the /robot_description topic
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}]
        ),
        # 2. Creates the Slider GUI to move the joints
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui'
        ),
        # 3. Opens RViz2 to visualize the robot
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            # Add -d flag later with .rviz config file path
        )
    ])
