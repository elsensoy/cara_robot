#!/usr/bin/env python3
"""
Replicates the Arduino main.cpp test sequences over ROS2.
Publishes named JointTrajectory commands to /joint_commands.
Loops: neck → hip → arms → neutral → look left/right
"""
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

_JOINTS = ['left_shoulder', 'right_shoulder', 'left_arm', 'right_arm',
           'hip', 'neck_yaw', 'neck_pitch']

_N = 90.0  # neutral angle for all joints

# (joint overrides, hold_seconds, label)
_SEQUENCES = [
    # neutral → everything centered
    ({}, 1.5, 'neutral'),

    # --- testNeckOnly ---
    ({'neck_yaw': 80.0, 'neck_pitch': 90.0}, 1.5, 'neck: yaw left'),
    ({'neck_yaw': 90.0, 'neck_pitch': 90.0}, 1.5, 'neck: center'),
    ({'neck_yaw': 100.0, 'neck_pitch': 90.0}, 1.5, 'neck: yaw right'),
    ({'neck_yaw': 90.0, 'neck_pitch': 90.0}, 1.5, 'neck: center'),
    ({'neck_yaw': 90.0, 'neck_pitch': 85.0}, 1.5, 'neck: pitch up'),
    ({'neck_yaw': 90.0, 'neck_pitch': 90.0}, 1.5, 'neck: center'),
    ({'neck_yaw': 90.0, 'neck_pitch': 95.0}, 1.5, 'neck: pitch down'),
    ({'neck_yaw': 90.0, 'neck_pitch': 90.0}, 1.0, 'neck: done'),

    # --- testHipOnly ---
    ({'hip': 80.0}, 1.5, 'hip: left'),
    ({'hip': 90.0}, 1.5, 'hip: center'),
    ({'hip': 100.0}, 1.5, 'hip: right'),
    ({'hip': 90.0}, 1.0, 'hip: done'),

    # --- testArmsOnly ---
    ({'left_shoulder': 80.0, 'right_shoulder': 100.0,
      'left_arm': 80.0, 'right_arm': 100.0}, 1.5, 'arms: pose A'),
    ({'left_shoulder': 100.0, 'right_shoulder': 80.0,
      'left_arm': 100.0, 'right_arm': 80.0}, 1.5, 'arms: pose B'),
    ({}, 1.0, 'arms: done'),

    # --- poseNeutralUpperBody ---
    ({}, 1.5, 'full neutral'),

    # --- poseLookLeftRight ---
    ({'neck_yaw': 80.0, 'neck_pitch': 90.0}, 1.2, 'look: left'),
    ({'neck_yaw': 100.0, 'neck_pitch': 90.0}, 1.2, 'look: right'),
    ({'neck_yaw': 90.0, 'neck_pitch': 90.0}, 2.0, 'look: done'),
]


class CaraServoTestNode(Node):
    def __init__(self):
        super().__init__('cara_servo_test_node')

        self._pub = self.create_publisher(JointTrajectory, '/joint_commands', 10)
        self._step = 0
        self._step_start = time.monotonic()
        self._ready = False

        # Move to neutral immediately, wait 1s before cycling
        self._send({})
        self.get_logger().info(
            f"Servo test ready — {len(_SEQUENCES)} steps, looping continuously.")

        self.create_timer(0.05, self._tick)  # 20 Hz state machine

    def _send(self, overrides: dict):
        angles = {j: _N for j in _JOINTS}
        angles.update(overrides)
        msg = JointTrajectory()
        msg.joint_names = _JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [angles[j] for j in _JOINTS]
        msg.points.append(pt)
        self._pub.publish(msg)

    def _tick(self):
        now = time.monotonic()

        if not self._ready:
            if now - self._step_start >= 1.0:
                self._ready = True
                self._step = 0
                self._step_start = now
                overrides, _, label = _SEQUENCES[0]
                self._send(overrides)
                self.get_logger().info(f"[step 0] {label}")
            return

        _, hold, _ = _SEQUENCES[self._step]
        if now - self._step_start >= hold:
            self._step = (self._step + 1) % len(_SEQUENCES)
            self._step_start = now
            overrides, _, label = _SEQUENCES[self._step]
            self._send(overrides)
            self.get_logger().info(f"[step {self._step}] {label}")


def main(args=None):
    rclpy.init(args=args)
    node = CaraServoTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
