#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from cara_vision_control.serial_control import (
    initialize_arduino, send_behavior_command, close_arduino,
)


class CaraArduinoBridgeNode(Node):
    def __init__(self):
        super().__init__('cara_arduino_bridge_node')
        try:
            initialize_arduino()
        except Exception as e:
            self.get_logger().fatal(f"Could not open Arduino serial: {e}")
            raise

        self.last_cmd = None  # for dedup
        self.sub = self.create_subscription(
            String, '/cara/behavior_cmd', self.cmd_callback, 10)
        self.get_logger().info(
            "Arduino bridge ready. Listening on /cara/behavior_cmd")

    def cmd_callback(self, msg):
        cmd = msg.data
        # Skip duplicate HEAD-only commands to reduce serial traffic
        # (but always send blinks)
        if cmd == self.last_cmd and 'BLINK:1' not in cmd:
            return
        self.last_cmd = cmd

        try:
            send_behavior_command(cmd)
            self.get_logger().debug(f"Executed: {cmd}")  # debug, not info
        except Exception as e:
            self.get_logger().error(f"Failed: {cmd} -> {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CaraArduinoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            close_arduino()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
