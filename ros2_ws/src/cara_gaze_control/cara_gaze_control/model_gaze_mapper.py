#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray

class ModelGazeMapper(Node):
    """
    Smart Gaze Mapper with Scanning & Direction Correction
    """
    def __init__(self):
        super().__init__("model_gaze_mapper")

        # --- Parameters 
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        
        # --- TUNING FOR ROBUSTNESS ---
        # 1.0 = Standard Tracking
        # 0.08 = Very fast servo response time (from your tuning)
        self.declare_parameter("tau_yaw", 0.05)    
        self.declare_parameter("tau_pitch", 0.05)
        self.declare_parameter("max_rate_rad_s", 6.0) # Fast!
        
        # --- THE FIX FOR "RUNAWAY" MOTION ---
        # If the robot spins AWAY from the face, change this to True!
        self.declare_parameter("invert_yaw_tracking", False)
        self.declare_parameter("invert_pitch_tracking", False)
        
        # --- SCANNING BEHAVIOR ---
        # If true, robot sweeps left/right when searching
        self.declare_parameter("enable_scanning", True)
        self.declare_parameter("scan_speed", 0.5)  # Rad/s
        self.declare_parameter("scan_range", 0.5)  # Radians (~30 deg)

        self.declare_parameter("no_face_timeout", 1.0) # Wait 1s before scanning
        self.declare_parameter("yaw_limits", [-1.0, 1.0])
        self.declare_parameter("pitch_limits", [-0.2, 0.6])
        
        self.declare_parameter("track_gain", 1.0) # Aggressiveness

        # Internal Variables
        self.W = self.get_parameter("image_width").value
        self.H = self.get_parameter("image_height").value
        self.yaw_min, self.yaw_max = self.get_parameter("yaw_limits").value
        self.pitch_min, self.pitch_max = self.get_parameter("pitch_limits").value
        
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0

        # State
        self.yaw = 0.0
        self.pitch = 0.0
        self.yaw_target = 0.0
        self.pitch_target = 0.0
        
        self.face_x = None
        self.face_y = None
        self.last_face_time = 0.0
        
        # Scanning State
        self.scan_phase = 0.0

        # ROS Setup
        self.sub_face = self.create_subscription(
            Point, "/faces/primary_center", self.face_callback, 10
        )
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, "/head_cmd", 10
        )
        
        # Run loop at 30Hz
        self.timer = self.create_timer(1.0/30.0, self.update)
        
        self.get_logger().info("Smart Gaze Mapper Initialized.")
        self.get_logger().info("TIP: If robot runs away from face, set 'invert_yaw_tracking' to True!")

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def face_callback(self, msg: Point):
        # Update "last seen" time
        self.last_face_time = time.time()
        
        # Raw Pixel Coordinates
        self.face_x = msg.x
        self.face_y = msg.y

    def update(self):
        # 1. Update Parameters Live (So you we tune without restarting!)
        # This lets toggle the invert switch instantly
        invert_yaw = self.get_parameter("invert_yaw_tracking").value
        invert_pitch = self.get_parameter("invert_pitch_tracking").value
        track_gain = self.get_parameter("track_gain").value
        do_scan = self.get_parameter("enable_scanning").value
        
        now = time.time()
        dt = 1.0 / 30.0 # Fixed step for simplicity
        
        time_since_face = now - self.last_face_time
        has_face = time_since_face < self.get_parameter("no_face_timeout").value

        # ================================
        # STATE 1: FACE DETECTED (TRACK)
        # ================================
        if has_face and self.face_x is not None:
            # 1. Calculate Error (Pixel Distance from Center)
            cx = self.W / 2.0
            cy = self.H / 2.0
            
            # Approximate intrinsics
            fx = 1064.0
            fy = 1068.0
            
            # Error = How far is the face from the center?
            # Standard: Left of center (x < cx) -> Negative Error
            x_err_px = self.face_x - cx
            y_err_px = self.face_y - cy
            
            # Convert to Angles (Small angle approximation is fine here)
            yaw_err_rad = -math.atan(x_err_px / fx)
            pitch_err_rad = math.atan(y_err_px / fy)
            
            # 2. APPLY DIRECTION CORRECTION (The Fix!)
            if invert_yaw:
                yaw_err_rad = -yaw_err_rad
            if invert_pitch:
                pitch_err_rad = -pitch_err_rad

            # 3. VISUAL SERVOING
            # "Move from where I am now (self.yaw) by the amount of the error"
            self.yaw_target = self.yaw + (yaw_err_rad * track_gain)
            self.pitch_target = self.pitch + (pitch_err_rad * track_gain)
            
            # Reset Scan Phase so we don't jump when we lose face later
            self.scan_phase = 0.0

        # ================================
        # STATE 2: SEARCHING (SCAN)
        # ================================
        elif do_scan:
            # Create a gentle Sine Wave to look for faces
            scan_speed = self.get_parameter("scan_speed").value
            scan_range = self.get_parameter("scan_range").value
            
            self.scan_phase += scan_speed * dt
            
            # Target is the sine wave
            self.yaw_target = self.neutral_yaw + math.sin(self.scan_phase) * scan_range
            self.pitch_target = self.neutral_pitch # Stay level
            
        # ================================
        # STATE 3: IDLE (CENTER)
        # ================================
        else:
            self.yaw_target = self.neutral_yaw
            self.pitch_target = self.neutral_pitch

        # ================================
        # CONTROL LAW (SMOOTHING)
        # ================================
        tau_yaw = self.get_parameter("tau_yaw").value
        tau_pitch = self.get_parameter("tau_pitch").value
        max_rate = self.get_parameter("max_rate_rad_s").value

        # Yaw Update
        dyaw = (self.yaw_target - self.yaw) / max(1e-3, tau_yaw)
        dyaw = self.clamp(dyaw, -max_rate, max_rate)
        self.yaw += dyaw * dt
        
        # Pitch Update
        dpitch = (self.pitch_target - self.pitch) / max(1e-3, tau_pitch)
        dpitch = self.clamp(dpitch, -max_rate, max_rate)
        self.pitch += dpitch * dt

        # Mechanical Limits
        self.yaw = self.clamp(self.yaw, self.yaw_min, self.yaw_max)
        self.pitch = self.clamp(self.pitch, self.pitch_min, self.pitch_max)

        # Publish
        msg = Float64MultiArray()
        msg.data = [self.yaw, self.pitch]
        self.pub_cmd.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ModelGazeMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
