#### 4. Integration Plan: From Will Cogley to Isaac Sim

Got the head model and the walking animation ready:

    **URDF to USD:** Convert your CAD/Skeleton into a URDF file (standard for ROS 2), then use the URDF Importer in Isaac Sim to turn it into a USD.

    **Rigid Body Setup:** In Isaac Sim, you’ll assign "Mass" and "Friction" to Cara's limbs. (Since she's a teddy bear, you can actually simulate the "drag" of the fur!)

    **The "Bridge":** Use the topic_based_control in ROS 2. This allows your RL policy in Isaac Sim to talk to the same ROS 2 nodes that will eventually talk to your real PCA9685 boards.
    
    
#### The URDF Hierarchy for Cara:

        base_link: The center of the torso (Root).

        waist_link: (3 DoF: Pitch, Roll, Yaw) connects the torso to the hips.

        neck_link: (3 DoF: Pitch, Roll, Yaw) connects torso to head.

            ear_l_link / ear_r_link: (1 DoF each) for expressions.

        shoulder_l/r_link: (3 DoF each) for high-mobility arms.

        hip_l/r_link: (3 DoF each) for the walking backbone.
        
        
### 2. Implementation: The Sim-to-Real Pipeline

To teach Cara to walk like Olaf, will follow a "Two-Layer" control strategy:

    The Articulated Backbone (RL): Use NVIDIA Isaac Lab to train a policy for the Hips, Waist, and Neck. These joints handle balance. The reward function will penalize falling and high energy consumption.

    The Show Functions (Classical): The Ears, Shoulders, and Blinking don't heavily affect Cara's balance. You can control these with standard ROS 2 JointTrajectoryControllers or your Arduino scripts without needing complex RL training.\\
    
    
### 3. Homeostasis: Thermal Awareness

To implement the homeostasis, add a Temperature Reward to your Isaac Lab training script.

    Disney's Research Trick: They added a "Thermal Model" to the simulation. If a virtual servo stayed at max torque too long, it "overheated" in the sim, and the robot lost points. Eventually, the RL learned a walk cycle that naturally alternates legs to let the motors cool down.
