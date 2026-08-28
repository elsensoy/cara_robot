import serial
import time

PORT = "/dev/ttyAMA0"   # or /dev/ttyUSB0 if that is the real port
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

for cmd in [
    "HEAD:75,BLINK:0\n",
    "HEAD:105,BLINK:0\n",
    "HEAD:90,BLINK:1\n",
]:
    print("Sending:", cmd.strip())
    ser.write(cmd.encode("utf-8"))
    time.sleep(2)

ser.close()
print("Done.")
