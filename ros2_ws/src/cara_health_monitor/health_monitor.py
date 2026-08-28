#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import os

class CaraHealthMonitor(Node):
    def __init__(self):
        super().__init__('cara_health_monitor')
        
        # Publisher for the 'Health Dashboard'
        self.health_pub = self.create_publisher(Float32MultiArray, '/cara/health_metrics', 10)
        self.affect_pub = self.create_publisher(String, '/cara/affect_state', 10)
        
        # Check every 2 seconds (Homeostasis doesn't need 50Hz)
        self.timer = self.create_timer(2.0, self.check_health)
        
        self.get_logger().info("Cara's Homeostasis Health Monitor is active.")

    def get_temp(self):
        # Reads the Jetson's thermal zone (usually zone 0 is CPU/GPU package)
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read()) / 1000.0
        except:
            return 0.0

    def check_health(self):
        temp = self.get_temp()
        
        # 1. Create Data Message [Temperature, CPU_Usage_Placeholder]
        msg = Float32MultiArray()
        msg.data = [temp]
        self.health_pub.publish(msg)

        # 2. Homeostasis Logic: Trigger expressions based on "Health"
        if temp > 65.0: # If Orin Nano is getting hot
            affect_msg = String()
            affect_msg.data = "sad" # Make Cara look tired/exhausted
            self.affect_pub.publish(affect_msg)
            self.get_logger().warn(f"Homeostasis Warning: Cara is overheating ({temp}C)!")

def main(args=None):
    rclpy.init(args=args)
    node = CaraHealthMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
