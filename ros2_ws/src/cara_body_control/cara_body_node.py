#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import onnxruntime as ort
import numpy as np
import time

class CaraBodyNode(Node):
    def __init__(self):
        super().__init__('cara_body_node')

        # 1. Load the "Brain" exported from your lab's RTX 5060
        # Ensure the path points to your actual exported ONNX file
        self.policy = ort.InferenceSession("cara_policy.onnx")
        
        # 2. State Variables
        self.current_joint_pos = np.zeros(20)  # Fed back from servos or last command
        self.last_imu_data = np.zeros(6)       # [roll, pitch, yaw, gx, gy, gz]
        
        # 3. ROS Interfaces
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self._imu_cb, 10)
        self.affect_pub = self.create_publisher(String, '/cara/affect_state', 10)
        self.joint_pub = self.create_publisher(JointTrajectory, '/joint_commands', 10)

        # 4. Control Loop (50Hz to match Isaac Lab training)
        self.timer = self.create_timer(0.02, self._control_tick)
        self.get_logger().info("Cara's Body Node is active and monitoring homeostasis.")

    def _imu_cb(self, msg):
        # Store orientation and angular velocity for the observation vector
        self.last_imu_data = np.array([
            msg.orientation.x, msg.orientation.y, msg.orientation.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        ])
        
        # Homeostasis check: If tipping too far, tell the head node!
        self.check_homeostasis(msg)

    def check_homeostasis(self, imu_msg):
        # 0.35 radians is roughly 20 degrees
        if abs(imu_msg.orientation.x) > 0.35 or abs(imu_msg.orientation.y) > 0.35:
            msg = String()
            msg.data = "sad"  # This triggers 'sad_blink' in your head node
            self.affect_pub.publish(msg)

    def _control_tick(self):
        # Create the Observation Vector (must match Isaac Lab's obs exactly)
        # Typically: [Joint Positions] + [IMU Data] + [Previous Actions]
        obs = np.concatenate([self.current_joint_pos, self.last_imu_data]).astype(np.float32)
        obs = obs.reshape(1, -1)

        # Run Inference
        actions = self.policy.run(None, {"obs": obs})[0][0]

        # Publish Joint Commands to your Actuator node (PCA9685 driver)
        traj_msg = JointTrajectory()
        point = JointTrajectoryPoint()
        point.positions = actions.tolist()
        traj_msg.points.append(point)
        self.joint_pub.publish(traj_msg)
        
        # Update internal state
        self.current_joint_pos = actions

def main(args=None):
    rclpy.init(args=args)
    node = CaraBodyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
