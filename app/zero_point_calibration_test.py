#!/usr/bin/env python3
import time
from adafruit_servokit import ServoKit

# Initialize the two boards
kit0 = ServoKit(channels=16, address=0x40)
kit1 = ServoKit(channels=16, address=0x41)

def calibrate_cara():
    print("Starting 20-Servo Calibration for Cara...")
    print("Moving all joints to 90-degree center point. Please wait.")
    
    # Board 1 (0x40) - Hips, Waist, Neck
    for i in range(16):
        print(f"Centering Board 0x40, Pin {i}")
        kit0.servo[i].angle = 90
        time.sleep(0.1) # Small delay to prevent current spikes

    # Board 2 (0x41) - Shoulders, Ears
    for i in range(4):
        print(f"Centering Board 0x41, Pin {i}")
        kit1.servo[i].angle = 90
        time.sleep(0.1)

    print("\n[SUCCESS] All 20 servos are now centered at 90 degrees.")
 

if __name__ == "__main__":
    calibrate_cara()
