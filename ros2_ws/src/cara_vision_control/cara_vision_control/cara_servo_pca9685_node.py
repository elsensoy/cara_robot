#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

HAVE_HARDWARE = False
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo as adafruit_servo
    HAVE_HARDWARE = True
except Exception as e:
    print(f"[servo_pca9685] Hardware libs not available, running in dummy mode: {e}")


class ServoPCA9685(Node):
    def __init__(self):
        super().__init__('servo_driver')

        # Parameters
        self.declare_parameter('i2c_addr', 0x40)
        self.declare_parameter('channels', [0, 1])
        self.declare_parameter('startup_deg', [90.0, 90.0])
        self.declare_parameter('neutral_deg', [90.0, 90.0])
        self.declare_parameter('min_deg', [45.0, 45.0])
        self.declare_parameter('max_deg', [175.0, 175.0])
        self.declare_parameter('max_speed_dps', [60.0, 60.0]) # Slower for stability
        self.declare_parameter('control_tau', 0.6)            # Smoother

        # --- NEW PARAMETERS FOR DIRECTION CORRECTION ---
        # Set to -1.0 to invert the axis!
        self.declare_parameter('pan_scale', -1.0)  
        self.declare_parameter('tilt_scale', 1.0)

        i2c_addr      = self.get_parameter('i2c_addr').value
        channels      = self.get_parameter('channels').value
        startup_deg   = self.get_parameter('startup_deg').value
        neutral_deg   = self.get_parameter('neutral_deg').value
        min_deg       = self.get_parameter('min_deg').value
        max_deg       = self.get_parameter('max_deg').value
        max_speed_dps = self.get_parameter('max_speed_dps').value
        self.tau      = self.get_parameter('control_tau').value
        
        self.pan_scale = self.get_parameter('pan_scale').value
        self.tilt_scale = self.get_parameter('tilt_scale').value

        self.pan_max_speed_dps, self.tilt_max_speed_dps = max_speed_dps
        self.last_pan_deg = startup_deg[0]
        self.last_tilt_deg = startup_deg[1]
        self.last_cmd_time = time.time()

        if len(channels) != 2:
            raise RuntimeError("servo_pca9685: 'channels' param must have length 2")

        self.pan_ch, self.tilt_ch = channels
        self.pan_neutral_deg, self.tilt_neutral_deg = neutral_deg
        self.pan_min_deg, self.tilt_min_deg = min_deg
        self.pan_max_deg, self.tilt_max_deg = max_deg

        self.get_logger().info(
            f"ServoPCA9685 starting. PanScale={self.pan_scale}, TiltScale={self.tilt_scale}"
        )

        self.pan_servo = None
        self.tilt_servo = None

        if HAVE_HARDWARE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                pca = PCA9685(i2c, address=i2c_addr)
                pca.frequency = 50

                self.pca = pca
                self.pan_servo = adafruit_servo.Servo(
                    pca.channels[self.pan_ch], min_pulse=500, max_pulse=2500
                )
                self.tilt_servo = adafruit_servo.Servo(
                    pca.channels[self.tilt_ch], min_pulse=500, max_pulse=2500
                )

                self.pan_servo.angle = startup_deg[0]
                self.tilt_servo.angle = startup_deg[1]
                time.sleep(0.3)
                self.get_logger().info("Hardware initialized.")
            except Exception as e:
                self.get_logger().error(f"Failed to init servos: {e}")

        self.sub = self.create_subscription(
            Float64MultiArray,
            '/head_cmd',
            self.head_cmd_callback,
            10
        )

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def _apply_ds_control(self, target_deg, current_deg, tau, max_speed_dps, dt):
        safe_tau = max(1e-3, tau)
        # 1. Calculate Error
        error = target_deg - current_deg
        
        # 2. Calculate ideal velocity (Proportional)
        # If error is positive (target > current), we want positive velocity
        raw_step = (error / safe_tau) * dt
        
        # 3. Clamp velocity
        max_step = max_speed_dps * dt
        clamped_step = self._clamp(raw_step, -max_step, max_step)
        
        return current_deg + clamped_step

    def head_cmd_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            return

        yaw_cmd = msg.data[0]    # radians
        pitch_cmd = msg.data[1]  # radians

        #  FIX: APPLY DIRECTION SCALING HERE  
        # If pan_scale is -1.0, positive yaw becomes negative degrees
        # target = Neutral + (Angle * Scale)
        
        target_pan_deg = self.pan_neutral_deg + (math.degrees(yaw_cmd) * self.pan_scale)
        target_tilt_deg = self.tilt_neutral_deg + (math.degrees(pitch_cmd) * self.tilt_scale)

        # Clamp targets
        target_pan_deg = self._clamp(target_pan_deg, self.pan_min_deg, self.pan_max_deg)
        target_tilt_deg = self._clamp(target_tilt_deg, self.tilt_min_deg, self.tilt_max_deg)

        now = time.time()
        dt = max(1e-3, now - self.last_cmd_time)
        self.last_cmd_time = now

        # Apply Smooth Control
        next_pan_deg = self._apply_ds_control(
            target_pan_deg, 
            self.last_pan_deg, 
            self.tau, 
            self.pan_max_speed_dps, 
            dt
        )
        
        next_tilt_deg = self._apply_ds_control(
            target_tilt_deg, 
            self.last_tilt_deg, 
            self.tau, 
            self.tilt_max_speed_dps, 
            dt
        )

        self.last_pan_deg = next_pan_deg
        self.last_tilt_deg = next_tilt_deg

        if self.pan_servo and self.tilt_servo:
            try:
                self.pan_servo.angle = next_pan_deg
                self.tilt_servo.angle = next_tilt_deg
            except Exception as e:
                self.get_logger().error(f"Servo error: {e}")

    def destroy_node(self):
        try:
            if hasattr(self, "pca") and self.pca:
                self.pca.deinit()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ServoPCA9685()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
