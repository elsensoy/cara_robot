# cara_stack.launch.py
# run: ros2 launch cara_bringup cara_stack.launch.py use_servos:=true
#
# Node ownership:
#   Neck pan/tilt servos  → model_gaze_mapper → servo_pca9685_node  (face tracking)
#   Blink / LED           → behavior_node → arduino_bridge_node     (emotion/mode)
#   No test code touches servos directly.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_servos    = LaunchConfiguration("use_servos")
    camera_id     = LaunchConfiguration("camera_id")
    img_w         = LaunchConfiguration("img_w")
    img_h         = LaunchConfiguration("img_h")
    enable_health = LaunchConfiguration("enable_health")
    health_source = LaunchConfiguration("health_source")
    health_setpoint_topic = LaunchConfiguration("health_setpoint_topic")

    vision_pkg_share = get_package_share_directory("cara_vision_control")
    vision_launch    = os.path.join(vision_pkg_share, "launch", "cara_runtime.launch.py")

    return LaunchDescription([
        DeclareLaunchArgument("use_servos", default_value="false"),
        DeclareLaunchArgument("camera_id",  default_value="0"),
        DeclareLaunchArgument("img_w",      default_value="640"),
        DeclareLaunchArgument("img_h",      default_value="480"),

        # Actuator-health signal path (jetson/control via cara_control pkg).
        # enable_health:=false to leave it out. health_source:=hw once the
        # BNO055 + servo-rail INA219 are wired.
        DeclareLaunchArgument("enable_health", default_value="true"),
        DeclareLaunchArgument("health_source", default_value="sim"),
        # Joint-target source for the health controller. Defaults to the
        # actuator node's own input so the controller sees Cara's real commanded
        # motion; set to "" for the internal demo gait.
        DeclareLaunchArgument("health_setpoint_topic", default_value="/joint_commands"),

        # ── 1. Full vision pipeline ───────────────────────────────────────
        # camera → face_yunet_node → emotion_node → behavior_node
        # → arduino_bridge_node (blink only)
        # → servo_pca9685_node  (only when use_servos:=true, driven by /head_cmd)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(vision_launch),
            launch_arguments={
                "use_servos": use_servos,
                "camera_id":  camera_id,
                "img_w":      img_w,
                "img_h":      img_h,
            }.items(),
        ),

        # ── 2. Gaze tracker ───────────────────────────────────────────────
        # Subscribes to /faces/primary_center (from face_yunet_node above).
        # Publishes /head_cmd → consumed by servo_pca9685_node when servos enabled.
        # This node is the ONLY thing that should command neck pan/tilt.
        Node(
            package="cara_gaze_control",
            executable="model_gaze_mapper",
            name="model_gaze_mapper",
            output="screen",
            parameters=[{
                "image_width":           640,
                "image_height":          480,
                "tau_yaw":               0.1,
                "tau_pitch":             0.1,
                "max_rate_rad_s":        4.0,
                "no_face_timeout":       2.8,
                "track_gain":            0.1,
                "invert_yaw_tracking":   False,
                "invert_pitch_tracking": False,
                "enable_scanning":       True,
                "scan_speed":            0.5,
                "scan_range":            0.5,
                "yaw_limits":            [-1.0, 1.0],
                "pitch_limits":          [-0.17, 0.61],
            }],
        ),

        # ── 3. Actuator-health / FDIR signal path ─────────────────────────
        # Reads servo-rail power + IMU, estimates a health scalar, feeds it
        # into the policy observation, runs the hand-written controller.
        # Observation-only: publishes /cara/health/* and /cara/control/*
        # but sends NO servo commands, so it can't collide with the gaze or
        # behavior pipelines above.
        Node(
            package="cara_control",
            executable="cara_control_node",
            name="cara_control_node",
            output="screen",
            condition=IfCondition(enable_health),
            parameters=[{
                "source":         health_source,   # "sim" | "hw"
                "rate_hz":        50.0,
                "i2c_bus":        1,
                "ina_addr":       0x45,            # servo rail, per tests/cara_power_monitor.py
                "imu_addr":       0x28,
                "setpoint_topic": health_setpoint_topic,
                "setpoint_timeout_s": 0.5,
            }],
        ),
    ])
