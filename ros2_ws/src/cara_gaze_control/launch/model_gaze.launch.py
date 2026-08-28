from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    camera_id = LaunchConfiguration('camera_id')
    img_w = LaunchConfiguration('img_w')
    img_h = LaunchConfiguration('img_h')
    use_servos = LaunchConfiguration('use_servos')

    share = get_package_share_directory('cara_vision_control')
    yunet = os.path.join(share, 'models', 'yunet.onnx')
    arc   = os.path.join(share, 'models', 'arcface.onnx')
    db    = os.path.expanduser('~/.cara/face_db.yaml')

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
                'camera_info_url': 'file:///root/.ros/camera_info/my_c270.yaml',
                'video_device': video_dev,
                'image_size': PythonExpression([
                    '[', 'int(', img_w, '), ', 'int(', img_h, ')]'
                ]),
                'pixel_format': 'YUYV',
                'output_encoding': 'yuv422_yuy2',
            }],
            additional_env={'QT_QPA_PLATFORM': 'offscreen', 'DISPLAY': ''},
        ),

        # 2) YuNet face detector node (publishes /faces/primary_center)
        Node(
            package='cara_vision_control',
            executable='face_yunet',
            name='face_yunet',
            output='screen',
            parameters=[{
                'camera_topic': '/image_raw',
                'detector_model': yunet,
                'score_threshold': 0.6,
                'nms_threshold': 0.3,
                'top_k': 5000,
            }],
        ),

        # 3) Model-based gaze mapper
# 3) Model-based gaze mapper
        Node(
            package='cara_gaze_control',
            executable='model_gaze_mapper',
            name='model_gaze_mapper',
            output='screen',
            parameters=[{
                'image_width': img_w,
                'image_height': img_h,
                'fov_deg': [62.0, 49.0],
                'neutral_yaw_rad': 0.0,
                'neutral_pitch_rad': 0.0,
                'yaw_limits': [-1.0, 1.0],
                'pitch_limits': [-0.17, 0.61],

                # --- STABILITY FIX: RE-ENABLE MILD PRE-SMOOTHING ---
                # Instead of 0.0, use a small value (0.1) to filter 
                # pixel noise before it even reaches the driver.
                'tau_yaw': 0.1,    
                'tau_pitch': 0.1,
                
                'max_rate_rad_s': 4.0, 
                'no_face_timeout': 2.8,
                'gh_g': 0.25,
                'gh_h': 0.05,
                'deadband_norm': 0.01,
                # --- NEW PARAMETER HERE ---
                # 1.0 = Centering is "perfect" (might oscillate if latency is high)
                # 0.5 = Centering is "soft" (slower to center, but more stable)
                'track_gain': 0.1,
            }],
        ),
         
        # 4) Servo driver (PCA9685 → pan/tilt) – optional
# 4) Servo driver
        Node(
            package='cara_vision_control',
            executable='servo_pca9685',
            name='servo_driver',
            output='screen',
            parameters=[{
                'i2c_addr': 0x40,
                'channels': [0, 1],
                
                'control_tau': 0.6,
                'max_speed_dps': [90.0, 90.0],
                
                # --- FIX FOR INVERTED MOTION ---
                'pan_scale': 1.0,   # Inverts Left/Right
                'tilt_scale': -1.0,   # Keep Up/Down normal (change to -1.0 if inverted)
                'startup_deg': [90.0, 120.0],
                'neutral_deg': [90.0, 120.0],
                'min_deg': [0.0, 0.0],
                'max_deg': [180.0, 180.0]
            }],
            condition=IfCondition(use_servos),
        ),
    ])

