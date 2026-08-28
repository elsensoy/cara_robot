from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    camera_id = LaunchConfiguration('camera_id')
    img_w = LaunchConfiguration('img_w')
    img_h = LaunchConfiguration('img_h')
    use_servos = LaunchConfiguration('use_servos')

    share = get_package_share_directory('cara_vision_control')
    yunet = os.path.join(share, 'models', 'yunet.onnx')
    arc = os.path.join(share, 'models', 'arcface.onnx')
    db = os.path.expanduser('~/.cara/face_db.yaml')

    video_dev = PythonExpression(["'/dev/video' + str(", camera_id, ")"])

    return LaunchDescription([
        DeclareLaunchArgument('camera_id', default_value='0'),
        DeclareLaunchArgument('img_w', default_value='640'),
        DeclareLaunchArgument('img_h', default_value='480'),
        DeclareLaunchArgument('use_servos', default_value='false'),

        # 1) Camera
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='camera',
            output='screen',
            parameters=[{
                'video_device': video_dev,
                'image_size': PythonExpression(['[', 'int(', img_w, '), ', 'int(', img_h, ')]']),
                'pixel_format': 'YUYV',
                'output_encoding': 'rgb8',
            }],
            additional_env={'QT_QPA_PLATFORM': 'offscreen', 'DISPLAY': ''},
        ),

        # 2) Face detection / recognition
        Node(
            package='cara_vision_control',    # <-- Changed to python package
            executable='face_yunet_node', # <-- Changed to python script
            name='face_id_node',
            output='screen',
            parameters=[{
                'camera_topic': '/image_raw',
                'detector_model': yunet,
                'score_threshold': 0.6,
                'nms_threshold': 0.3,
                'top_k': 5000,
                'use_gpu': False,
            }],
        ),

        # 3) Emotion node
        Node(
            package='cara_vision_control',
            executable='emotion_node',
            name='emotion_node',
            output='screen',
            parameters=[{
                'camera_id': camera_id,
                'use_gstreamer': True,
                'img_w': img_w,
                'img_h': img_h
            }]
        ),

        # 4) Behavior node
        Node(
            package='cara_vision_control',
            executable='behavior_node',
            name='behavior_node',
            output='screen',
        ),

        # 5) Arduino bridge node
        Node(
            package='cara_vision_control',
            executable='arduino_bridge_node',
            name='arduino_bridge_node',
            output='screen',
        ),

        # 6) Optional PCA9685 servo node
        Node(
            package='cara_vision_control',
            executable='servo_pca9685',
            name='servo_driver',
            output='screen',
            parameters=[{
                'i2c_addr': 0x40,
                'channels': [0, 1],
                'startup_deg': [90.0, 90.0],
                'neutral_deg': [90.0, 90.0],
                'min_deg': [30.0, 40.0],
                'max_deg': [150.0, 130.0],
            }],
            condition=IfCondition(use_servos),
        ),
    ])
