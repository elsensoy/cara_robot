
### 1. The URDF Architecture (Xacro)

Because 20 joints are a lot to type, we use **Xacro** (XML Macros) to keep the code clean. Below is the structural skeleton. Saved as `cara.urdf.xacro`.

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="cara">

  <link name="base_link"/>

  <link name="torso">
    <inertial>
      <mass value="2.5"/> <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
  </joint>

  <xacro:macro name="waist_joint" params="suffix axis">
    <joint name="waist_${suffix}" type="revolute">
      <parent link="torso"/>
      <child link="waist_link_${suffix}"/>
      <axis xyz="${axis}"/>
      <limit lower="-0.7" upper="0.7" effort="10" velocity="1.0"/>
    </joint>
  </xacro:macro>

  </robot>

```

---

### 2. Linking Sim to Real (The "Nervous System" Map)

Once we have this URDF, we need to map the **Joint Names** to **PCA9685 Channels**. This is done in a ROS 2 hardware configuration file.

| URDF Joint Name | Board | PCA9685 Pin | Hardware Purpose |
| --- | --- | --- | --- |
| `waist_pitch` | 0x40 | 0 | Balance / Leaning forward |
| `l_hip_yaw` | 0x40 | 3 | Leg rotation for turning |
| `r_ear_joint` | 0x41 | 6 | Expression (non-locomotion) |

By naming them clearly in the URDF, Reinforcement Learning policy in Isaac Sim can simply output a "Target Angle" for `waist_pitch`, and  ROS 2 node will automatically send that signal to **Board 0x40, Pin 0**.

---
### 3. Implementing Homeostasis in Simulation

To teach Cara to "care" about her temperature (like the Olaf robot), we add a **Thermal Penalty** to the Isaac Lab reward function.

* **Virtual Temperature:** In Python script for Isaac Lab, create a variable  that increases whenever a joint’s `effort` (torque) is high.
* **The Reward:** `reward -= weight * sum(max(0, T_joint - threshold))`.
* **Result:** The RL agent will learn to "rest" certain joints or use different muscles to avoid the "pain" of overheating.

### 4. Safety: Protecting the Orin Nano Super

Since we are using an Arduino Nano for the head and a Jetson Orin Nano Super for the body, we must prevent **Ground Loops**.

1. **Optical Isolation:** If possible, use an optoisolator between the Jetson's  pins and the PCA9685 logic.
2. **Star Grounding:** Connect all Ground wires (Battery, Jetson, PCA9685, Arduino) to a single, heavy-duty "Ground Block." This prevents "noise" from the servos from confusing the Jetson’s data lines.

---
[Disney's Olaf Robot Research](https://www.youtube.com/watch?v=-L8OFMTteOo)

