#!/usr/bin/env python3
import math
import struct
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

HAVE_HARDWARE = False
try:
    from smbus2 import SMBus
    HAVE_HARDWARE = True
except ImportError:
    pass

BNO055_ADDR = 0x28
BUS = 1

# Registers
_CHIP_ID_REG    = 0x00
_OPR_MODE_REG   = 0x3D
_PWR_MODE_REG   = 0x3E
_SYS_TRIGGER    = 0x3F
_QUA_W_LSB      = 0x20  # Quaternion w/x/y/z, 8 bytes, Q14 (1/16384 per unit)
_GYRO_X_LSB     = 0x14  # Gyro x/y/z, 6 bytes, 1/16 deg/s per unit
_LIA_X_LSB      = 0x28  # Linear accel x/y/z, 6 bytes, 1/100 m/s² per unit

_CONFIG_MODE    = 0x00
_NDOF_MODE      = 0x0C

_QUAT_SCALE  = 1.0 / 16384.0
_GYRO_SCALE  = math.pi / (16.0 * 180.0)   # deg/s → rad/s
_ACCEL_SCALE = 0.01                         # cm/s² → m/s²


class CaraImuNode(Node):
    def __init__(self):
        super().__init__('cara_imu_node')
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('frame_id', 'imu_link')

        self._frame_id = self.get_parameter('frame_id').value
        rate = self.get_parameter('publish_rate_hz').value

        self._pub = self.create_publisher(Imu, '/imu/data', 10)
        self._bus = None

        if HAVE_HARDWARE:
            try:
                self._bus = SMBus(BUS)
                chip = self._bus.read_byte_data(BNO055_ADDR, _CHIP_ID_REG)
                if chip != 0xA0:
                    self.get_logger().error(f"BNO055 chip ID mismatch: 0x{chip:02X}")
                    self._bus.close()
                    self._bus = None
                else:
                    self._init_sensor()
                    self.get_logger().info("BNO055 initialized (NDOF mode).")
            except Exception as e:
                self.get_logger().error(f"IMU init failed: {e}")
                if self._bus:
                    self._bus.close()
                self._bus = None
        else:
            self.get_logger().warn("smbus2 unavailable — IMU in dummy mode.")

        self.create_timer(1.0 / rate, self._publish)

    def _write(self, reg, val):
        self._bus.write_byte_data(BNO055_ADDR, reg, val)
        time.sleep(0.02)

    def _init_sensor(self):
        self._write(_OPR_MODE_REG, _CONFIG_MODE)
        self._write(_PWR_MODE_REG, 0x00)
        self._write(_SYS_TRIGGER,  0x00)
        self._write(_OPR_MODE_REG, _NDOF_MODE)
        time.sleep(0.05)

    def _publish(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id

        if self._bus:
            try:
                # Quaternion (w, x, y, z)
                raw = self._bus.read_i2c_block_data(BNO055_ADDR, _QUA_W_LSB, 8)
                w, x, y, z = struct.unpack('<hhhh', bytes(raw))
                msg.orientation.w = w * _QUAT_SCALE
                msg.orientation.x = x * _QUAT_SCALE
                msg.orientation.y = y * _QUAT_SCALE
                msg.orientation.z = z * _QUAT_SCALE

                # Gyro (rad/s)
                raw = self._bus.read_i2c_block_data(BNO055_ADDR, _GYRO_X_LSB, 6)
                gx, gy, gz = struct.unpack('<hhh', bytes(raw))
                msg.angular_velocity.x = gx * _GYRO_SCALE
                msg.angular_velocity.y = gy * _GYRO_SCALE
                msg.angular_velocity.z = gz * _GYRO_SCALE

                # Linear acceleration (m/s²)
                raw = self._bus.read_i2c_block_data(BNO055_ADDR, _LIA_X_LSB, 6)
                ax, ay, az = struct.unpack('<hhh', bytes(raw))
                msg.linear_acceleration.x = ax * _ACCEL_SCALE
                msg.linear_acceleration.y = ay * _ACCEL_SCALE
                msg.linear_acceleration.z = az * _ACCEL_SCALE

            except OSError as e:
                self.get_logger().warn(f"IMU read error: {e}", throttle_duration_sec=2.0)
                return
        else:
            # Dummy: identity quaternion, zero everything
            msg.orientation.w = 1.0

        # -1 in [0] means covariance unknown (ROS convention)
        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0

        self._pub.publish(msg)

    def destroy_node(self):
        if self._bus:
            try:
                self._bus.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CaraImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
