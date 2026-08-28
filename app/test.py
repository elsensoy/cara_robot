import serial, time
ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
time.sleep(2)
ser.write(b'HEAD:75,BLINK:1\n')
