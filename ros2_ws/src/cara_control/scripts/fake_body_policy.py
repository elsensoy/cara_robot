#!/usr/bin/env python3
"""Stand-in for cara_body_node, for testing cara_control_node without an ONNX policy.

Publishes the SAME message shape cara_body_node emits: an unnamed
trajectory_msgs/JointTrajectory on /joint_commands with positions in [-1, 1],
at 50 Hz (cara_body_node: `point.positions = actions.tolist()`, no joint_names).

cara_control_node maps unnamed positions p -> p * pi/2 rad, so with system_health
near 1.0 you should see /cara/control/action track roughly `positions * pi/2`,
shrinking as you drive /cara/sim/fault up.

Usage:
    ros2 run cara_control fake_body_policy
    ros2 run cara_control fake_body_policy --ros-args -p rate_hz:=50.0 -p gait_hz:=0.5
"""
import math

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Amplitudes in [-1, 1] policy space, servo index order (arduino/main.cpp):
# left_shoulder, right_shoulder, left_arm, right_arm, hip, neck_yaw, neck_pitch
_AMPL = [0.6, -0.6, 0.4, -0.4, 0.25, 0.2, 0.15]
_PHASE = [0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0]
_FREQ_MULT = [1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0]


class FakeBodyPolicy(Node):
    def __init__(self):
        super().__init__("fake_body_policy")
        rate_hz = self.declare_parameter("rate_hz", 50.0).value
        self.gait_hz = self.declare_parameter("gait_hz", 0.5).value
        self.pub = self.create_publisher(JointTrajectory, "/joint_commands", 10)
        self.t0 = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(
            f"fake body policy -> /joint_commands (unnamed, [-1,1]) at {rate_hz:.0f} Hz"
        )

    def tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        base = 2 * math.pi * self.gait_hz * t
        acts = [
            a * math.sin(base * f + p)
            for a, p, f in zip(_AMPL, _PHASE, _FREQ_MULT)
        ]
        msg = JointTrajectory()
        point = JointTrajectoryPoint()
        point.positions = acts
        msg.points.append(point)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FakeBodyPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
