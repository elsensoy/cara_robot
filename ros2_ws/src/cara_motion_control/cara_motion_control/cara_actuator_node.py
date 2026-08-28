#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

HAVE_SERIAL = False
try:
    import serial
    HAVE_SERIAL = True
except ImportError:
    pass

# (channel, min_deg, neutral_deg, max_deg) — must match Arduino joint table
JOINT_MAP = {
    'left_shoulder':  (0,  75, 90, 105),
    'right_shoulder': (1,  75, 90, 105),
    'left_arm':       (2,  75, 90, 105),
    'right_arm':      (3,  75, 90, 105),
    'hip':            (4,  80, 90, 100),
    'neck_yaw':       (5,  80, 90, 100),
    'neck_pitch':     (6,  85, 90,  95),
}


class CaraActuatorNode(Node):
    def __init__(self):
        super().__init__('cara_actuator_node')
        self.declare_parameter('serial_port', '/dev/ttyCH341USB0')
        self.declare_parameter('baud_rate', 115200)

        self._ser = None
        if HAVE_SERIAL:
            port = self.get_parameter('serial_port').value
            baud = self.get_parameter('baud_rate').value
            try:
                self._ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2.0)  # wait for Arduino reset after serial open
                self.get_logger().info(
                    f"Arduino serial bridge ready: {port} @ {baud} baud.")
            except Exception as e:
                self.get_logger().error(f"Serial init failed: {e}")
                self._ser = None
        else:
            self.get_logger().warn("pyserial unavailable — running in dummy mode.")

        self.create_subscription(JointTrajectory, '/joint_commands', self._on_cmd, 10)
        self.get_logger().info("Actuator node ready.")

    def _send(self, channel: int, angle: float):
        cmd = f"S{channel},{int(angle)}\n"
        if self._ser and self._ser.is_open:
            try:
                self._ser.write(cmd.encode())
            except Exception as e:
                self.get_logger().warn(f"Serial write failed: {e}")

    def _on_cmd(self, msg: JointTrajectory):
        if not msg.points:
            return
        positions = list(msg.points[0].positions)

        if msg.joint_names:
            self._drive_named(msg.joint_names, positions)
        else:
            self._drive_indexed(positions)

    def _drive_named(self, names, positions):
        for name, deg in zip(names, positions):
            if name not in JOINT_MAP:
                self.get_logger().warn(f"Unknown joint: {name}")
                continue
            ch, lo, _, hi = JOINT_MAP[name]
            safe = max(float(lo), min(float(hi), float(deg)))
            self.get_logger().debug(f"{name} ch{ch}: {deg:.1f}° → {safe:.1f}°")
            self._send(ch, safe)

    def _drive_indexed(self, positions):
        # Backward-compat: unnamed joints from RL policy, positions in [-1, 1]
        if len(positions) < 7:
            self.get_logger().warn(
                f"Indexed mode: expected ≥7 joints, got {len(positions)}")
            return
        for i, pos in enumerate(positions[:7]):
            angle = max(0.0, min(180.0, (pos + 1.0) * 90.0))
            self._send(i, angle)

    def destroy_node(self):
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CaraActuatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
