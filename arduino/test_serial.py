import serial
import time

PORT = "/dev/ttyCH341USB0"  # or /dev/ttyUSB0 if you switch later
BAUD = 9600

print(f"Opening {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)  # let Arduino reset

cmd = "HEAD:75,BLINK:1\n"
print(f"Sending: {cmd.strip()}")
ser.write(cmd.encode("utf-8"))

time.sleep(1)

ser.close()
print("Done.")
