import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# 1) I2C + PCA9685 init
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c, address=0x40)   # change if i2cdetect shows different
pca.frequency = 50                 # standard servo freq

# 2) Create servo objects on channels 0 and 1 (guess; can swap if reversed)
pan_servo = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2500)
tilt_servo = servo.Servo(pca.channels[1], min_pulse=500, max_pulse=2500)

# 3) Center them
pan_servo.angle = 90
tilt_servo.angle = 90
time.sleep(1)

# 4) Sweep test
def sweep(s):
    for angle in range(45, 135, 5):
        s.angle = angle
        time.sleep(0.05)
    for angle in range(135, 45, -5):
        s.angle = angle
        time.sleep(0.05)

print("Sweeping pan...")
sweep(pan_servo)
print("Sweeping tilt...")
sweep(tilt_servo)

# Go back to center
pan_servo.angle = 90
tilt_servo.angle = 90
pca.deinit()
