import onnxruntime as ort
import numpy as np
from adafruit_servokit import ServoKit
import time
#training
#python isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Cara-v0 --headless
#export
#python isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Cara-v0 --checkpoint /path/#to/your/model.pt --export
# 1. Initialize the 20-servo "Nervous System" (2 boards)
kit0 = ServoKit(channels=16, address=0x40)
kit1 = ServoKit(channels=16, address=0x41)

# 2. Load the "Brain" (Policy) exported from the Lab computer
policy = ort.InferenceSession("cara_policy.onnx")

def get_observations():
    # In a real build, we'd pull this from your IMU and Encoder feedback
    # For now, we'll use a placeholder vector of 48 inputs (typical for RL)
    return np.random.randn(1, 48).astype(np.float32)

def deploy_loop():
    while True:
        # STEP A: Sense (Observations)
        obs = get_observations()
        
        # STEP B: Think (Inference)
        # The policy outputs the 20 target angles
        action = policy.run(None, {"obs": obs})[0][0]
        
        # STEP C: Act (Drive Servos)
        # Map actions (usually -1 to 1) to servo angles (0 to 180)
        for i in range(16):
            angle = np.clip((action[i] + 1) * 90, 0, 180)
            kit0.servo[i].angle = angle
            
        for i in range(4): # The remaining 4 joints on board 2
            angle = np.clip((action[16+i] + 1) * 90, 0, 180)
            kit1.servo[i].angle = angle
            
        time.sleep(0.02) # Run at 50Hz for smooth motion

if __name__ == "__main__":
    deploy_loop()
