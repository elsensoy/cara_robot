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
