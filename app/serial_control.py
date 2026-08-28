# Set this one variable to 'True' to use Jetson's pins.
# Set it to 'False' to send commands to the Arduino.
USE_JETSON_GPIO = False

import atexit
import time

if USE_JETSON_GPIO:
    import RPi.GPIO as GPIO
else:
    import serial

# Global placeholders
pwm_motor = None
arduino_serial = None

# -----------------------------------------------------------------
# --- SECTION 1: JETSON NANO (GPIO) LOGIC ---
# -----------------------------------------------------------------

# Pin 33 (PWM) for Motor
# Pin 12 (Digital) for Eyes
JETSON_MOTOR_PIN = 33
JETSON_EYES_PIN = 12

def _jetson_initialize():
    """
    Initializes the Jetson GPIO pins.
    """
    global pwm_motor
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(JETSON_EYES_PIN, GPIO.OUT, initial=GPIO.LOW)

        GPIO.setup(JETSON_MOTOR_PIN, GPIO.OUT, initial=GPIO.LOW)
        pwm_motor = GPIO.PWM(JETSON_MOTOR_PIN, 100)  # 100 Hz
        pwm_motor.start(0)

        print("--- SUCCESS: Using JETSON GPIO ---")
        print(f"  Motor (PWM) on Pin {JETSON_MOTOR_PIN}")
        print(f"  Eyes (Digital) on Pin {JETSON_EYES_PIN}")

    except Exception as e:
        print("--- ERROR: Could not initialize Jetson GPIO! ---")
        print(f"Error: {e}")

def _jetson_send_emotion(emotion_string):
    """
    Backward-compatible old interface.
    Maps simple emotion labels to behavior commands.
    """
    emotion_string = emotion_string.strip().lower()

    if emotion_string == "happy":
        _jetson_send_behavior("HEAD:110,BLINK:0")
    elif emotion_string == "sad":
        _jetson_send_behavior("HEAD:75,BLINK:1")
    else:
        _jetson_send_behavior("HEAD:90,BLINK:0")

def _jetson_send_behavior(command_string):
    """
    Executes a behavior command on Jetson GPIO.
    Example: HEAD:72,BLINK:1
    """
    global pwm_motor

    try:
        parts = {}
        for item in command_string.split(','):
            key, value = item.split(':')
            parts[key.strip().upper()] = value.strip()

        head = int(parts.get("HEAD", 90))
        blink = int(parts.get("BLINK", 0))

        # Simple approximation for now
        if head > 100:
            print(f"[GPIO]: HEAD up -> PWM burst for angle {head}")
            pwm_motor.ChangeDutyCycle(100)
            time.sleep(0.2)
            pwm_motor.ChangeDutyCycle(0)
        elif head < 80:
            print(f"[GPIO]: HEAD down -> PWM burst for angle {head}")
            pwm_motor.ChangeDutyCycle(60)
            time.sleep(0.2)
            pwm_motor.ChangeDutyCycle(0)
        else:
            pwm_motor.ChangeDutyCycle(0)

        if blink == 1:
            print("[GPIO]: Blink triggered")
            GPIO.output(JETSON_EYES_PIN, GPIO.HIGH)
            time.sleep(0.15)
            GPIO.output(JETSON_EYES_PIN, GPIO.LOW)

    except Exception as e:
        print(f"ERROR: Failed to execute Jetson behavior command: {e}")

def _jetson_cleanup():
    """
    Safely cleans up the Jetson GPIO pins on exit.
    """
    print("Shutting down and cleaning up Jetson GPIO pins...")
    try:
        if pwm_motor:
            pwm_motor.stop()
        GPIO.cleanup()
        print("Jetson GPIO cleanup complete.")
    except Exception as e:
        print(f"Jetson cleanup warning: {e}")


# -----------------------------------------------------------------
# --- SECTION 2: ARDUINO NANO (SERIAL) LOGIC ---
# -----------------------------------------------------------------

# Change this after checking:
# ls /dev/ttyUSB* /dev/ttyACM* /dev/ttyAMA* 2>/dev/null
ARDUINO_PORT = '/dev/ttyCH341USB0'
ARDUINO_BAUD = 9600

def _arduino_initialize():
    """
    Initializes the serial connection to the Arduino.
    """
    global arduino_serial
    try:
        arduino_serial = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)  # wait for Arduino reset
        print(f"--- SUCCESS: Using ARDUINO on {ARDUINO_PORT} ---")
    except serial.SerialException as e:
        print("--- ERROR: Could not connect to Arduino! ---")
        print(f"Error: {e}")
        print("1. Is the Arduino plugged in?")
        print(f"2. Is the port correct? (Currently {ARDUINO_PORT})")
        print("3. If using CH340, check whether the driver/device is present.")
    except Exception as e:
        print(f"An unknown error occurred during Arduino init: {e}")

def _arduino_send_emotion(emotion_string):
    """
    Backward-compatible old interface.
    Maps simple emotion labels to behavior commands.
    """
    emotion_string = emotion_string.strip().lower()

    if emotion_string == "happy":
        _arduino_send_behavior("HEAD:110,BLINK:0")
    elif emotion_string == "sad":
        _arduino_send_behavior("HEAD:75,BLINK:1")
    else:
        _arduino_send_behavior("HEAD:90,BLINK:0")

def _arduino_send_behavior(command_string):
    """
    Sends a full behavior command line to the Arduino.
    Example: HEAD:72,BLINK:1
    """
    global arduino_serial

    if not arduino_serial or not arduino_serial.is_open:
        print("Cannot send command: Arduino is not connected.")
        return

    try:
        line = command_string.strip() + '\n'
        print(f"[Serial]: Sending -> {line.strip()}")
        arduino_serial.write(line.encode('utf-8'))
    except Exception as e:
        print(f"ERROR: Failed to write to Arduino serial port: {e}")

def _arduino_cleanup():
    """
    Safely closes the serial port on exit.
    """
    global arduino_serial
    if arduino_serial and arduino_serial.is_open:
        print(f"Shutting down and closing Arduino port {ARDUINO_PORT}...")
        arduino_serial.close()
        print("Arduino port closed.")


# -----------------------------------------------------------------
# --- SECTION 3: PUBLIC WRAPPER FUNCTIONS ---
# -----------------------------------------------------------------

def initialize_arduino():
    """
    PUBLIC: Initializes the chosen controller.
    """
    if USE_JETSON_GPIO:
        _jetson_initialize()
    else:
        _arduino_initialize()

def send_emotion_to_arduino(emotion_string):
    """
    PUBLIC: Backward-compatible emotion interface.
    """
    if USE_JETSON_GPIO:
        _jetson_send_emotion(emotion_string)
    else:
        _arduino_send_emotion(emotion_string)

def send_behavior_command(command_string):
    """
    PUBLIC: Sends a direct behavior command to the chosen controller.
    Example: "HEAD:72,BLINK:1"
    """
    if USE_JETSON_GPIO:
        _jetson_send_behavior(command_string)
    else:
        _arduino_send_behavior(command_string)

def close_arduino():
    """
    PUBLIC: Cleans up the chosen controller.
    """
    if USE_JETSON_GPIO:
        _jetson_cleanup()
    else:
        _arduino_cleanup()

# Automatic cleanup on exit
atexit.register(close_arduino)
