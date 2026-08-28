The RTX 5060 (Blackwell architecture, released in early 2025) 

    During Training: The policy exists as a .pt (PyTorch) checkpoint file.

    Where to find it: After running the training script, look in Isaac Lab directory:

        logs/rsl_rl/cara_task/2025-12-28_18-00/model_10000.pt

    For Deployment: YWe will convert this .pt file into an ONNX or TorchScript (.jit) file. These formats are highly optimized for the Jetson Orin's hardware.

### 2. The Training Commands

On lab's 5060 computer, we will use the Isaac Lab wrapper to start the "evolutionary" learning process.

#### Step A: Start the Training Run this command to start thousands of "virtual Caras" learning to walk at once:

```
python isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Cara-v0 --headless
```

> Note: --headless makes it faster by not rendering the graphics while it "thinks."

#### Step B: Export the Brain Once the reward graph plateaus (she’s walking!), run the play.py script to generate the deployment file:
Bash
```
python isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Cara-v0 --checkpoint logs/rsl_rl/cara_task/model_final.pt --export
```

This will create an exported/policy.onnx file in that same folder. This is the file you copy onto Cara's Jetson.
---

### 3. Running it on the Orin Nano Super

To bridge the "Sim-to-Real" gap, you will use a small Python node on the Jetson that:

    Reads Observations: Gets the current angle of the 20 servos + the IMU (gyroscope) data.

    Inference: Feeds that data into the .onnx policy.

    Actions: Sends the output target angles to your PCA9685 boards.

### 4. Important: The "Observation Space"

For Cara to "feel" she is falling, the policy needs specific inputs. In Isaac Lab config, ensure to include:

    base_lin_vel: How fast she is moving.

    projected_gravity: Which way is "down" (from the IMU).

    joint_pos / joint_vel: The state of all 20 servos.

    thermal_state: (For your homeostasis goal) The calculated heat of the motors.

### Homeostasis "Secret Sauce"
Train on "stamina." not just walking. If we want Cara to be aware of her temperature, pass a "Virtual Thermometer" value into the policy during training. If the value gets too high, the policy learns to "limp" or sit down to "cool off."

