#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HeadExpressionNode(Node):
    """
    ROS 2 node that drives Cara's head/eyes/ears expression hardware.
    Supports two backends:
      - Jetson GPIO (RPi.GPIO on Jetson)
      - Arduino serial (pyserial)
    """

    def __init__(self):
        super().__init__("cara_head_expression_node")
 
        # Parameters
       
        self.declare_parameter("use_jetson_gpio", False)
        self.declare_parameter("arduino_port", "/dev/ttyAMA0")
        self.declare_parameter("arduino_baud", 9600)

        # GPIO pins (BOARD numbering)
        self.declare_parameter("jetson_motor_pin", 33)  # PWM
        self.declare_parameter("jetson_eyes_pin", 12)   # Digital

        # Motion tuning
        self.declare_parameter("happy_pulse_s", 0.50)
        self.declare_parameter("sad_blink_s", 0.30)
        self.declare_parameter("pwm_hz", 100)

        self.use_jetson_gpio = bool(self.get_parameter("use_jetson_gpio").value)
 
        # Backend init
      
        self._pwm_motor = None
        self._serial = None
        self._GPIO = None

        if self.use_jetson_gpio:
            self._init_gpio()
        else:
            self._init_serial()
 
        # ROS interfaces
       
        self.sub_affect = self.create_subscription(
            String, "/cara/affect_state", self.on_affect, 10
        )
        # Optional: also listen to vision emotion if you want plug-and-play
        self.sub_vision = self.create_subscription(
            String, "/cara/emotion", self.on_vision_emotion, 10
        )

        
        # Non-blocking action state machine
         
        self._active_action = None
        self._action_end_time = 0.0
        self._timer = self.create_timer(0.02, self._tick)  # 50 Hz tick

        self.get_logger().info(
            f"HeadExpressionNode ready | backend={'GPIO' if self.use_jetson_gpio else 'SERIAL'}"
        )
 
    # Parsing helpers
     
    def _normalize_label(self, raw: str) -> str:
        s = (raw or "").strip().lower()
        if not s:
            return "neutral"
        # Accept "happy (0.85)" format from vision node
        if " " in s and "(" in s:
            s = s.split("(", 1)[0].strip()
        return s
 
    # ROS callbacks
     
    def on_affect(self, msg: String):
        label = self._normalize_label(msg.data)
        self._schedule_expression(label, source="affect_state")

    def on_vision_emotion(self, msg: String):
        # If you ONLY want affect_state to control hardware, comment this out.
        label = self._normalize_label(msg.data)
        # Optional: ignore vision unless no affect messages are present.
        # For now, we just allow it.
        self._schedule_expression(label, source="vision")
  
    # Action scheduling (non-blocking)
     
    def _schedule_expression(self, label: str, source: str):
        # You can expand this mapping later without touching the rest.
        if label == "happy":
            self._start_action("happy_pulse", float(self.get_parameter("happy_pulse_s").value))
        elif label == "sad":
            self._start_action("sad_blink", float(self.get_parameter("sad_blink_s").value))
        else:
            self._start_action("neutral", 0.0)

        self.get_logger().debug(f"Scheduled '{label}' from {source}")

    def _start_action(self, action_name: str, duration_s: float):
        self._active_action = action_name
        self._action_end_time = time.monotonic() + max(0.0, duration_s)
        # Apply “start” immediately (don’t wait for timer tick)
        self._apply_action_start(action_name)

    def _tick(self):
        # Called at 50 Hz
        if self._active_action is None:
            return

        if time.monotonic() >= self._action_end_time:
            # End action
            self._apply_action_end(self._active_action)
            self._active_action = None
 
    # Hardware backend actions
    
    def _apply_action_start(self, action: str):
        if self.use_jetson_gpio:
            self._gpio_action_start(action)
        else:
            self._serial_action_start(action)

    def _apply_action_end(self, action: str):
        if self.use_jetson_gpio:
            self._gpio_action_end(action)
        else:
            self._serial_action_end(action)

    # --- GPIO backend ---
    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)

            motor_pin = int(self.get_parameter("jetson_motor_pin").value)
            eyes_pin  = int(self.get_parameter("jetson_eyes_pin").value)
            pwm_hz    = int(self.get_parameter("pwm_hz").value)

            GPIO.setup(eyes_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(motor_pin, GPIO.OUT, initial=GPIO.LOW)

            self._pwm_motor = GPIO.PWM(motor_pin, pwm_hz)
            self._pwm_motor.start(0)

            self.get_logger().info(
                f"GPIO initialized | motor_pin={motor_pin} eyes_pin={eyes_pin} pwm_hz={pwm_hz}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to init GPIO backend: {e}")
            raise

    def _gpio_action_start(self, action: str):
        motor_pin = int(self.get_parameter("jetson_motor_pin").value)
        eyes_pin  = int(self.get_parameter("jetson_eyes_pin").value)

        try:
            if action == "happy_pulse":
                # Start motor pulse
                self._pwm_motor.ChangeDutyCycle(100)
            elif action == "sad_blink":
                self._GPIO.output(eyes_pin, self._GPIO.HIGH)
            else:
                # Neutral: ensure off
                self._pwm_motor.ChangeDutyCycle(0)
                self._GPIO.output(eyes_pin, self._GPIO.LOW)
        except Exception as e:
            self.get_logger().warn(f"GPIO start action failed: {e}")

    def _gpio_action_end(self, action: str):
        eyes_pin  = int(self.get_parameter("jetson_eyes_pin").value)
        try:
            # End pulses / reset to neutral
            self._pwm_motor.ChangeDutyCycle(0)
            self._GPIO.output(eyes_pin, self._GPIO.LOW)
        except Exception as e:
            self.get_logger().warn(f"GPIO end action failed: {e}")

    #   Serial backend 
    def _init_serial(self):
        try:
            import serial
            port = str(self.get_parameter("arduino_port").value)
            baud = int(self.get_parameter("arduino_baud").value)

            self._serial = serial.Serial(port, baud, timeout=1)
            time.sleep(2.0)  # allow Arduino reset
            self.get_logger().info(f"Serial initialized | port={port} baud={baud}")
        except Exception as e:
            self.get_logger().error(f"Failed to init serial backend: {e}")
            raise

    def _serial_action_start(self, action: str):
        if self._serial is None or not self._serial.is_open:
            self.get_logger().warn("Serial not open; cannot send.")
            return
        try:
            if action == "happy_pulse":
                self._serial.write(b'H')
            elif action == "sad_blink":
                self._serial.write(b'B')
            else:
                self._serial.write(b'N')
        except Exception as e:
            self.get_logger().warn(f"Serial write failed: {e}")

    def _serial_action_end(self, action: str):
        #  serial don't need an "end" command.
        #  send 'N' Arduino sketch might still expect it.
        return
 
    # Cleanup
    def destroy_node(self):
        # Make shutdown robust
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
        except Exception:
            pass

        try:
            if self._pwm_motor:
                self._pwm_motor.stop()
            if self._GPIO:
                self._GPIO.cleanup()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HeadExpressionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

