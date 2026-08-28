#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray


class GazeDebugger(Node):
    def __init__(self):
        super().__init__("gaze_debugger")

        # Image size parameters (must match your launch, or we can pull from params)
        self.declare_parameter("img_w", 640.0)
        self.declare_parameter("img_h", 480.0)
        self.img_w = float(self.get_parameter("img_w").value)
        self.img_h = float(self.get_parameter("img_h").value)

        self.face_sub = self.create_subscription(
            Point, "/faces/primary_center", self.face_cb, 10
        )
        self.head_sub = self.create_subscription(
            Float64MultiArray, "/head_cmd", self.head_cb, 10
        )

        self.last_face = None
        self.last_head = None

        self.get_logger().info(
            f"GazeDebugger started (img_w={self.img_w}, img_h={self.img_h})"
        )

    def face_cb(self, msg: Point):
        self.last_face = msg
        cx, cy, conf = msg.x, msg.y, msg.z

        # normalize to [-1, 1] around the center of the image
        nx = (cx - self.img_w / 2.0) / (self.img_w / 2.0)
        ny = (cy - self.img_h / 2.0) / (self.img_h / 2.0)
        err_norm = math.sqrt(nx * nx + ny * ny)

        self.get_logger().info(
            f"[FACE] center=({cx:.1f}, {cy:.1f}) norm_err={err_norm:.3f} conf={conf:.2f}"
        )

    def head_cb(self, msg: Float64MultiArray):
        self.last_head = msg
        if len(msg.data) != 2:
            self.get_logger().warn(f"/head_cmd has unexpected size: {len(msg.data)}")
            return

        yaw_rad, pitch_rad = msg.data
        yaw_deg = yaw_rad * 180.0 / math.pi
        pitch_deg = pitch_rad * 180.0 / math.pi

        self.get_logger().info(
            f"[HEAD] yaw={yaw_rad:.3f} rad ({yaw_deg:.1f} deg), "
            f"pitch={pitch_rad:.3f} rad ({pitch_deg:.1f} deg)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GazeDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
