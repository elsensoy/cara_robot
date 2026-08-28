import serial
import time
 
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
import os, serial
SERIAL_PORT = os.getenv("CARA_SERIAL_PORT", "/dev/ttyACM0")  # or /dev/ttyACM0
ser = serial.Serial(PORT, 115200, timeout=1)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Let Arduino reset
except serial.SerialException as e:
    print(f"Error: Could not open serial port {SERIAL_PORT}: {e}")
    ser = None

def send_emotion_to_arduino(emotion):
    if ser is None:
        print("Serial connection not available.")
        return

    try:
        if emotion == "happy":
            ser.write(b'H')  # Move head
        elif emotion == "sad":
            ser.write(b'B')  # Blink
        else:
            ser.write(b'N')  # Neutral/do nothing
    except serial.SerialException as e:
        print(f"Error writing to serial: {e}")

