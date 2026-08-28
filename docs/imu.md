4. Homeostasis: The "Thermal Watchdog"

To make Cara "self-aware" of her temperature as planned, WE can add a simple Python thread on the Jetson that monitors the Orin Nano's onboard sensors. If the temperature exceeds a threshold (e.g., 70°C), WE can send a "Soft Override" signal to the deploy_loop to reduce the servo speed or trigger a "Sit Down" animation to let the Jetson cool.

WTo make **Cara** walk, she needs to "feel" gravity and her own momentum. This requires an **Inertial Measurement Unit (IMU)**.

**BNO055**. Unlike basic sensors (like the MPU6050) that give  raw, "noisy" data, the BNO055 has an internal processor that does **Sensor Fusion**. It outputs "clean" Euler angles (Roll, Pitch, Yaw) directly, which is vital for Reinforcement Learning (RL) stability.

---

### 1. Wiring the BNO055 to the Jetson Orin Nano

Since we are already using the  bus for PCA9685 servo drivers, what we can do is to  simply "daisy-chain" the IMU onto the same pins. The Jetson can talk to multiple devices on the same wires as long as they have different addresses (PCA9685 is `0x40`, BNO055 is usually `0x28`).

| BNO055 Pin | Jetson Orin Nano Pin | Note |
| --- | --- | --- |
| **Vin** | Pin 1 or 17 (3.3V) | Power |
| **GND** | Pin 6, 9, or 14 (GND) | **Must share a common ground** |
| **SDA** | Pin 3 (SDA) | Data line |
| **SCL** | Pin 5 (SCL) | Clock line |

---

### 2. The IMU Integration Code

We'll use the `adafruit-circuitpython-bno055` library. This script will read Cara's orientation and package it into the "Observation Vector" that RL policy needs to decide her next move.

```python
import time
import board
import adafruit_bno055
import numpy as np

# 1. Initialize I2C and Sensor
i2c = board.I2C() 
sensor = adafruit_bno055.BNO055_I2C(i2c)

def get_cara_observations(joint_positions):
    """
    Combines joint states and IMU data into a single vector for the RL Policy.
    Standard RL models expect a flat array (usually ~48 values).
    """
    # Orientation (Roll, Pitch, Yaw)
    euler = sensor.euler # (heading, roll, pitch)
    
    # Angular Velocity (Rad/s) - how fast is she tipping?
    gyro = sensor.gyro # (x, y, z)
    
    # Flatten everything into a single observation array
    # Format: [20 Joint Positions] + [3 Orientation] + [3 Angular Vel]
    obs = np.concatenate([
        joint_positions, 
        np.array(euler), 
        np.array(gyro)
    ])
    
    return obs.astype(np.float32)

# Test Loop
while True:
    # Example: assume all 20 joints are at 0 degrees for now
    fake_joints = np.zeros(20) 
    current_obs = get_cara_observations(fake_joints)
    
    print(f"Cara's Pitch: {sensor.euler[2]} | Is she stable? {'Yes' if abs(sensor.euler[2]) < 5 else 'Falling!'}")
    time.sleep(0.02) # 50Hz Loop

```

---

### 3. Sim-to-Real: The "Gravity" Secret

In **Isaac Lab**, virtual Cara has a "perfect" IMU. In the real world, the BNO055 might have a slight delay or vibration from the 3D-printed body.

 When training in Isaac Lab, we will use **Domain Randomization**. The goal will be  totell the simulator to add "noise" to the IMU data. This forces the RL policy to become robust; if it can learn to walk with "shaky" simulated data, it will walk perfectly on the real hardware.
* **Calibration:** The BNO055 calibrates itself. When we turn Cara on, move her in a "Figure-8" pattern in the air for a few seconds. This helps the magnetometer find North and stabilizes the internal "homeostasis" of the sensor.

---

### 4. Integrating with Arduino Head

Since **Arduino Nano** controls the head (blinking/ears), we can send the IMU's "Tilt" data from the Jetson to the Arduino over Serial. If Cara tips forward too fast, the Arduino can trigger a "Scared/Wide-eyed" expression automatically!
 

[BNO055 IMU - NVIDIA Jetson](https://www.youtube.com/watch?v=Xw9ApZrJaAs)

