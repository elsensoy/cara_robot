# head_controller.py
import time
import threading
from adafruit_servokit import ServoKit
import board
import busio
import serial
#debugging
#i2cdetect -y 1
class HeadController:
    def __init__(self, enable_serial=True):
        print("[Init] Starting HeadController")
        self.i2c = busio.I2C(board.SCL_1, board.SDA_1)
        self.kit = ServoKit(channels=16, i2c=self.i2c)
        self.heading_servo = 0
        self.pitch_servo = 1

        self._target_heading = 90
        self._target_pitch = 90
        self.servo_range = {
            self.heading_servo: (10, 170),
            self.pitch_servo: (35, 90)
        }

        # Serial for Arduino blinking
        self.serial_enabled = enable_serial
        self.ser = None
        if enable_serial:
            try:
                self.ser = serial.Serial("/dev/ttyAMA0", 9600, timeout=1)
                time.sleep(2)
            except serial.SerialException as e:
                print(f"[ERROR] Serial not available: {e}")

        self._run_thread = True
        self._thread = threading.Thread(target=self._servo_loop)
        self._thread.daemon = False
        self._thread.start()

    def _servo_loop(self):
        while self._run_thread:
            self._move_servos()
            time.sleep(0.05)

    def shutdown(self):
        print("[Shutdown] Stopping HeadController")
        self._run_thread = False
        if self.ser:
            self.ser.close()

    def _move_servos(self):
        self.kit.servo[self.heading_servo].angle = self._target_heading
        self.kit.servo[self.pitch_servo].angle = self._target_pitch

    def express_emotion(self, emotion):
        print(f"[Emotion] Expressing: {emotion}")

        if emotion == "happy":
            self.look_left_right()
            self._send_serial('H')
        elif emotion == "sad":
            self.nod_down_up()
            self._send_serial('B')
        else:
            self.center_head()
            self._send_serial('N')

    def _send_serial(self, code):
        if self.ser:
            try:
                self.ser.write(code.encode())
            except serial.SerialException as e:
                print(f"[Serial] Write failed: {e}")

    def look_left_right(self):
        self._target_heading = 60
        time.sleep(0.4)
        self._target_heading = 120
        time.sleep(0.4)
        self._target_heading = 90

    def nod_down_up(self):
        self._target_pitch = 70
        time.sleep(0.4)
        self._target_pitch = 90

    def center_head(self):
        self._target_heading = 90
        self._target_pitch = 90
