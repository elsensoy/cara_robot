from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    imu_node = Node(
        package='cara_motion_control',
        executable='cara_imu_node',
        name='cara_imu_node',
        output='screen',
    )

    actuator_node = Node(
        package='cara_motion_control',
        executable='cara_actuator_node',
        name='cara_actuator_node',
        output='screen',
        parameters=[{'serial_port': '/dev/ttyCH341USB0', 'baud_rate': 115200}],
    )

    servo_test_node = Node(
        package='cara_motion_control',
        executable='cara_servo_test_node',
        name='cara_servo_test_node',
        output='screen',
    )

    return LaunchDescription([imu_node, actuator_node, servo_test_node])
