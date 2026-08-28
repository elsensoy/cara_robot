import Jetson.GPIO as GPIO
import time

motor_pin = 33  # GPIO pin number (check Jetson pinout)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(motor_pin, GPIO.OUT)

try:
    while True:
        GPIO.output(motor_pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(motor_pin, GPIO.LOW)
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
