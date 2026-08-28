#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from adafruit_servokit import ServoKit

class CaraActuatorNode(Node):
    def __init__(self):
        super().__init__('cara_actuator_node')

        # 1. Initialize Boards (Standard addresses 0x40 and 0x41)
        # Board 0: Pins 0-15 | Board 1: Pins 0-3 (used for the remaining 4 joints)
        self.kit0 = ServoKit(channels=16, address=0x40)
        self.kit1 = ServoKit(channels=16, address=0x41)

        # 2. Setup Servo Limits (Match these to your 3D printed mechanical limits!)
        # Example: kit0.servo[0].set_pulse_width_range(500, 2500) 

        # 3. Subscribe to the commands from your Body Policy Node
        self.subscription = self.create_subscription(
            JointTrajectory,
            '/joint_commands',
            self.listener_callback,
            10)
        
        self.get_logger().info("Actuator Node online: Driving 20 servos across 2 PCA9685 boards.")

    def listener_callback(self, msg):
        # We expect 20 position values in the JointTrajectoryPoint
        if not msg.points:
            return
            
        positions = msg.points[0].positions

        # Safety check for joint count
        if len(positions) < 20:
            self.get_logger().warn(f"Received only {len(positions)} joints. Need 20.")
            return

        # Map actions to physical pins
        for i in range(20):
            # Convert RL output (-1 to 1) to servo degrees (0 to 180)
            # You can tune the center offset here
            target_angle = (positions[i] + 1.0) * 90.0
            target_angle = max(0, min(180, target_angle)) # Hard clamp for safety

            if i < 16:
                self.kit0.servo[i].angle = target_angle
            else:
                self.kit1.servo[i-16].angle = target_angle

def main(args=None):
    rclpy.init(args=args)
    node = CaraActuatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
