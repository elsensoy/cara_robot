
## Embodied Motion & Control

## 1. Cara as an Embodied Control System

Cara is a **20-DoF articulated robotic character**, and LLM, whose motion is governed by a unified control stack spanning simulation, real-time inference, and physical actuation.

Rather than treating emotion, memory, and motion as separate subsystems, Cara is designed around a **continuous control loop**:

```
Perception -> Internal State -> Motion Policy -> Body -> Feedback
```
https://github.com/user-attachments/assets/e023d652-621f-4e27-8c53-0dc29c00cf4d

*media/videos/cara_walk_mesh.webm*

Emotion and memory influence *how* motion is generated, by **shaping the constraints and targets of the controller**.

 
## 2. Control Stack Overview

**Runtime Platform**

* Jetson Orin Nano Super (GPU inference + ROS 2)
* Arduino Nano (low-level expression servos)
* ROS 2 Humble + Docker (deployment consistency)

**Core Control Loop**

* **IMU (BNO055):** gravity, orientation, stability
* **Policy:** learned locomotion + posture regulation
* **Actuation:** 20 PWM servos via dual PCA9685
* **Feedback:** inertial + joint state closure

Emotion detection (ViT-based) feeds into this loop **only as a modifier**, affecting posture bias, tempo, and expressive joints.

---

## 3. Emotion as a Control Signal 

Rather than driving discrete actions (“blink”, “tilt”), emotional state is represented as a **low-dimensional continuous vector**:

```
emotion_state ∈ ℝⁿ  ->   modifies policy targets
```

Examples:

* Sad -> reduced stride amplitude, forward torso bias
* Curious ->  increased head-leading motion
* Excited ->  higher gait tempo (without torque spikes)

This keeps balance guarantees intact while allowing expressive variation.

> **Important:** No emotional state overrides stability or safety constraints.

(Details of emotion learning and ViT adaptation are documented in `emotion.md`.)

---
3D CAD model of a 20-DoF humanoid teddy bear robot skeleton. Design follows research-grade joint standards: 3-axis waist (pitch/yaw/roll), two 3-DoF shoulders, two 3-DoF hips, a 3-axis neck, and 1-DoF expressive ears. No tail. The rounded mid-body must include a precise mounting cradle for a Jetson Orin Nano Super and a LiPo battery. Focus is on functional mechanical geometry: recessed pockets for standard servos, internal wire-routing channels, and reinforced bolt-hole patterns. Skeleton should have ergonomic, rounded edges to fit inside a plush fabric shell. The mesh is manifold, 3D-printable, and optimized for high-torque motion and stability. Proportions: wide torso, sturdy leg-base, and expressive head mount.


#### Mechanical Strategy for Cara

Since we are using the Jetson Orin Nano Super, we have significant compute power, but it also means the mid-body needs specific thermal and structural features:

**Weight Distribution:** To keep Cara stable while walking or moving, the Orin Nano and the battery should be placed as low as possible in the "belly" of the bear. This lowers the Center of Gravity (CoG).

**Thermal Venting:** Even though she’s a teddy bear, the Orin Nano Super generates heat. the cad modeling of torso might need manual "slats" or a small 5V fan mount added to the back of the skeleton to prevent the plush fur from trapping too much heat.

**Servo Selection:** For the 20-joint setup, using high-torque metal gear servos for the waist and hips recommended, as they will support the entire weight of the Orin and the upper body.

### A Note on the Ears

By assigning 2 servos to the ears, we can program "moods" (e.g., ears pinned back for "sad/scared," twitching for "listening/curious"). This is a huge part of what will make Cara feel sentient rather than just a machine.

##### Safety First: The Orin Nano Super Wiring

Since we are using a Jetson Orin Nano Super, our primary goal is protecting the carrier board from "back-EMF" (voltage spikes) from those 20 servos.

#### The "Isolation" Diagram:
 
```
[ 12V LiPo Battery ] 
      |
      +-----> [ High-Current 5V/6V BEC ] ------> [ PCA9685 Power Terminal ]
      |                                                  |
      +-----> [ Jetson Power Regulator ] ------> [ Jetson Orin Nano Super ]
                                                         |
                                                 [ I2C SDA/SCL Pins ]
                                                         |
                                                 [ PCA9685 Logic Pins ]
```
**CRITICAL: Connect the Ground (GND) of the BEC to the GND of the Jetson.**

# Locomotion, Balance, and IMU Integration

## 4. Walking as a Learned Stability Problem

Walking is trained in **Isaac Lab** as a reinforcement learning problem where balance, energy efficiency, and recoverability dominate raw speed.

**Observations**

* Joint positions (20)
* Joint velocities (20)
* Base linear velocity
* IMU orientation (simulated)

**Actions**

* Joint position targets (mapped 1:1 to servos)
```
Parameter,Value (Servo),Effect on Cara
Stiffness (kp​),400 - 800,"Higher = more ""rigid"" and aggressive walking. Lower = ""softer"" waddle."
Damping (kd​),10 - 40,Prevents the 3D-printed limbs from vibrating after a fast move.
Effort,2.5 - 5.0 Nm,Limits how much force the sim-robot can use (don't exceed real servo torque!).
Friction,0.05,"Simulates the internal drag of the metal gears and the ""fur"" friction."
```

**Design Choice**

> The simulation runs at the **same control frequency (50 Hz)** as the physical robot, allowing direct policy transfer.

---

## 5. IMU Integration (Gravity Awareness)

Cara’s balance depends on a **BNO055 IMU**, chosen for onboard sensor fusion and low-latency orientation estimates.

IMU data serves two roles:

1. **Stability feedback** (preventing falls)
2. **Expressive modulation** (postural tilt, attentional cues)

To support sim-to-real transfer:

* IMU noise is **domain-randomized in simulation**
* Real-world readings are clipped and smoothed before inference

---

# URDF, Xacro, and Servo Topology

## 6. The URDF as “Digital Anatomy”

Cara’s physical structure is formalized using **Xacro-based URDF**, which acts as a single source of truth for:

* Simulation dynamics (Isaac Sim)
* ROS 2 joint interfaces
* Hardware servo mapping

### Minimal Xacro Skeleton

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="cara">

  <link name="base_link"/>

  <link name="torso">
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
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

Using Xacro keeps the 20-DoF structure **maintainable and auditable**, especially as joint limits or masses change.

---

## 7. Servo Mapping & ros2_control

Cara uses **ros2_control** to abstract servos as standard joint interfaces.

**Why this matters**

* Policies operate on joint names, not hardware pins
* Simulation and hardware share identical interfaces
* Isaac-trained policies deploy without rewrites

### Example Mapping

| Joint Name    | Board | Channel | Function       |
| ------------- | ----- | ------- | -------------- |
| `waist_pitch` | 0x40  | 0       | Balance / lean |
| `l_hip_yaw`   | 0x40  | 3       | Turning        |
| `r_ear_joint` | 0x41  | 6       | Expression     |

Under the hood, a PCA9685 ros2_control hardware plugin translates joint targets -> PWM signals.

---

# Wiring, Power, and Safety 

## 8. Power & Grounding

* **Servos:** external 5–6 V high-current BEC
* **Jetson:** independent supply
* **Arduino:** logic-only load

 **All grounds are common**
Failure to do this causes unstable PWM and IMU noise.

## 9. Safety Constraints

* Physical E-Stop on servo power rail
* Free-fall detection -> tuck posture
* Thermal watchdog on Jetson + simulated thermal penalty during training

These constraints exist **in both simulation and hardware**, preventing policy shock during deployment.
 
 
## 🐻 Cara Locomotion & Embodied Emotion

### Walking, Balance, and Sim-to-Real Training in Isaac Sim

Cara’s walking behavior is not treated as a purely mechanical gait problem. Instead, locomotion is modeled as an **extension of her emotional and physiological state**, trained in simulation and deployed in real time on embedded hardware.

At a high level, Cara follows a **Sense -> Think -> Act** pipeline inspired by Disney Research’s articulated character control (e.g., *Olaf*), implemented using **ROS 2**, **NVIDIA Isaac Lab**, and **reinforcement learning** .

---

## 1. System Overview: Continuity From Emotion to Motion

Cara’s embodiment pipeline is deliberately continuous:

**Emotion -> Policy State -> Joint Motion -> Physical Feedback**

Emotion detection (LLM + sentiment classification) modulates internal targets such as posture, stiffness, and movement tempo. These signals influence the locomotion policy *indirectly* by shaping rewards and constraints rather than issuing discrete animation commands.

This avoids “puppeteering” and instead produces **emergent expressive motion**.

---

## 2. ROS 2 Architecture: Sense -> Think -> Act

Cara’s real-world walking controller runs on a **Jetson Orin Nano**, using ROS 2 to mirror the same structure used in simulation.

### Core Nodes

| Node            | Role                                             |
| --------------- | ------------------------------------------------ |
| `imu_node`      | Reads fused orientation data from the BNO055 IMU |
| `policy_node`   | Runs the trained RL policy (ONNX)                |
| `actuator_node` | Drives 20 servos via dual PCA9685 boards         |

All nodes communicate using standard ROS 2 messages (`sensor_msgs/Imu`, `trajectory_msgs/JointTrajectory`), enabling seamless simulation-to-real transfer.

---

## 3. Training Cara to Walk in Isaac Sim

Walking behavior is trained entirely in **simulation** using **Isaac Lab** before being deployed to hardware.

### 3.1 Environment Definition

Cara is imported as a fully articulated body (USD converted from URDF) and trained in a managed RL environment:

```python
class CaraLocomotionCfg(ManagerBasedRLEnvCfg):
    ...
```

**Scene**

* Cara USD asset
* Flat terrain (with optional friction randomization)

**Observations**

* 20 joint positions
* 20 joint velocities
* Base linear velocity (balance feedback)

**Actions**

* Target joint positions (mapped 1:1 to servos)

---

### 3.2 Reward Design: Walking With Homeostasis

Cara does not learn to walk *as fast as possible*. She learns to walk **sustainably**.

| Reward Term       | Purpose                     |
| ----------------- | --------------------------- |
| Forward velocity  | Encourages locomotion       |
| Energy penalty    | Prevents servo strain       |
| Thermal proxy     | Encourages alternating gait |
| Stability penalty | Prevents falling            |

This mirrors biological locomotion: Cara learns to walk in a way that **lets her motors “rest”**, rather than locking joints under constant torque.

---

## 4. The IMU: Teaching Cara Gravity

Walking requires balance. Balance requires a sense of gravity.

Cara uses a **BNO055 IMU**, chosen specifically for its onboard sensor fusion. Rather than learning from raw accelerometer noise, the policy observes **clean orientation estimates** (roll, pitch, yaw).

#### 2. Evaluating the "Character" Response

Since Cara is a teddy bear, her recovery should look "cute" but functional.

- The Waddle: If she takes small, fast steps to recover, kp​ (Stiffness) is likely well-tuned.

- The "Homeostasis" Feedback: During a push, the torques will spike. This is the perfect time to test if the Body Node triggers the "Scared" or "Sad" expression on the head. In the sim, we should see the thermal_penalty increase momentarily.
https://www.youtube.com/watch?v=TMHkFDhVt7g
## 5. Sim-to-Real: One Brain, Two Worlds

A core design principle of Cara is that **the same policy runs in simulation and on hardware**.

### Deployment Steps

1. **Train** locomotion policy in Isaac Lab
2. **Export** trained model to ONNX
3. **Load** policy on Jetson Orin Nano
4. **Run inference** at 50 Hz
5. **Map outputs** directly to servo angles

```text
Isaac Sim   ->  ROS 2 Topics    ->   ONNX Policy    ->  PCA9685    ->   Servos
```

No animation layers, no hand-coded gaits.

---

## 6. Hardware Actuation: 20-Servo Nervous System

Cara’s body uses **20 high-DOF servos**, driven by two PCA9685 boards:

| Component        | Power                 | Notes               |
| ---------------- | --------------------- | ------------------- |
| Jetson Orin Nano | 9–20V DC              | Logic + compute     |
| PCA9685 (×2)     | 5V logic              | I2C @ 0x40 / 0x41   |
| Servos (×20)     | High-current 5–6V BEC | Isolated power rail |

**Critical:** All components share a **common ground**. Without this, PWM commands become unstable and dangerous.

---

## 7. Safety & Self-Preservation
| Component               | Estimated Mass (g)         | Location    |
|------------------------ | ---------------------------| ----------- |
|Jetson Orin Nano Super   |	~150g (with heatsink)  | Low Torso   |
|2S LiPo Battery (5000mAh)|	~250g	               | Lowest Torso|
|20 Metal Gear Servos     |     ~1,100g (55g each)     | Distributed |
|3D Printed Skeleton/Shell|	~500g	               | Distributed |
|Total Target Mass  	  |     ~2,000g (2.0kg) 


### Built-In Safeguards

* **E-Stop:** Physical cutoff for servo power
* **Free-fall detection:** IMU triggers protective “tuck” posture
* **Thermal watchdog:** Jetson temperature throttles movement

These constraints are mirrored during RL training so that real-world safety does not surprise the policy.

---
### Health Monitor

The Health Monitor should act as the "Whistleblower." It doesn't move the servos itself; it publishes a "distress signal" that other nodes listen to.

**Health Node:** Notices the Orin Nano is at 75°C. It publishes String: "critical" to /cara/homeostasis_status.

**Head Node:** Sees the "critical" status and immediately triggers the "Sad/Tired" expression (drooping ears).

**Body Node (Motion):** Sees the "critical" status and switches the RL policy to a "Low Energy Mode" or a sitting position to let the Jetson cool down.
    
## 8. Emotional Locomotion

Locomotion is not isolated from Cara’s affective system.

* “Happy” -> higher stride energy, upright posture
* “Sad” -> reduced speed, forward pitch
* “Curious”-> head-led balance shifts

---
 
### Mechanical Bill of Materials (mBOM) - Cara 20-DoF Teddy Skeleton

## A. Printed Parts (CAD / 3D-Printed)

| Item                                         |    Qty | Notes                                                   |
| -------------------------------------------- | -----: | ------------------------------------------------------- |
| Torso shell (2-piece clamshell)              |  1 set | Includes Jetson cradle + battery bay + wiring channels  |
| Waist gimbal stack (pitch/roll/yaw housings) |  1 set | Reinforced ribs + heat-set insert pockets               |
| Neck gimbal stack (pitch/roll/yaw housings)  |  1 set | Separate from torso for service access                  |
| Head mount + face frame                      |  1 set | Servo pockets + wire pass-through                       |
| Hip modules (L/R, 3-DoF each)                | 2 sets | Highest-load joints; print in PETG/Nylon-CF if possible |
| Leg base / foot plates                       |      2 | Wide stance, anti-slip pads mount points                |
| Shoulder modules (L/R, 3-DoF each)           | 2 sets | Moderate load; PLA+ ok for prototyping                  |
| Arm linkages (upper/lower)                   | 2 sets | Rounded edges for plush integration                     |
| Ear linkages (L/R, 1-DoF each)               |      2 | Light load, high visibility                             |
| Internal cable guides / clips                |  10–20 | Snap-in routing + strain relief                         |

**Print guidance (practical):**

* Load-bearing (waist + hips): PETG / Nylon-CF, 40–60% infill, 4–6 walls
* Everything else: PLA+ acceptable for early iterations

---

## B. Fasteners & Inserts (Core Hardware)

| Item                | Spec                | Qty (start) | Notes                                   |
| ------------------- | ------------------- | ----------: | --------------------------------------- |
| Socket head screws  | M3 × 8mm            |         100 | General assembly                        |
| Socket head screws  | M3 × 12mm           |          60 | Thicker joints / gimbal stacks          |
| Socket head screws  | M3 × 16mm           |          30 | Through-bolting plates                  |
| Nuts                | M3 nyloc            |          50 | Where inserts aren’t used               |
| Washers             | M3 flat             |         150 | Prevents plastic creep/cracking         |
| Heat-set inserts    | M3 × (4–6mm)        |         100 | Strongly recommended for serviceability |
| Self-tapping screws | Servo mounting size |      1 pack | For servo tabs if don’t use inserts |
| Standoffs           | M3 (6–12mm)         |       10–20 | For Jetson cradle / internal plates     |


## C. Joint Mechanics

| Item             | Spec              |  Qty | Notes                                  |
| ---------------- | ----------------- | ---: | -------------------------------------- |
| Flanged bearings | 3–6mm ID (varies) | 8–16 | For hip/waist pivots if you add shafts |
| Shafts / pins    | 3mm or 4mm steel  | 6–12 | Use with bearings for smoother gimbals |
| Thrust washers   | 3–4mm ID          | 6–12 | Reduces axial friction in gimbals      |
| Threadlocker     | Medium (blue)     |    1 | For high-vibration joints              |

**Rule of thumb:** put bearings/shafts in **waist + hips** first

---

## D. Servo Coupling & Linkage Hardware

| Item                  | Spec            | Qty (for 20-DoF) | Notes                                     |
| --------------------- | --------------- | ---------------: | ----------------------------------------- |
| Servo horns           | Metal preferred |            20–30 | Extras because we’ll strip/iterate       |
| Horn screws           | Servo-specific  |            1 set | Usually included with servos; keep spares |
| Linkage rods          | M2–M3           |            10–20 | pushrods for shoulders/arms    |
| Ball links / rod ends | M2–M3           |            10–20 | Smoother motion, less binding             |
| Rubber grommets       | small           |            10–20 | Cable pass-through + vibration reduction  |

---

## E. Mounting: Jetson + Battery + Cooling (Mechanical-only items)

| Item                  | Spec           |     Qty | Notes                                |
| --------------------- | -------------- | ------: | ------------------------------------ |
| Jetson mounting plate | Custom/printed |       1 | Integrate standoff pattern in torso  |
| Thermal pad           | 1–2mm          |  1 pack | Helps conduct heat to internal plate |
| Micro fan (optional)  | 5V 30–40mm     |       1 | Add rear vent mount                  |
| Fan screws            | M2/M3          |  1 pack | Depends on fan model                 |
| Battery straps        | Velcro 20–25mm |       2 | Secure LiPo in belly cradle          |
| Foam padding          | EVA/neoprene   | 1 sheet | Battery isolation + rattle reduction |

---

## F. Wire Management & Strain Relief (Mechanical supports)

| Item                | Spec     |    Qty | Notes                                 |
| ------------------- | -------- | -----: | ------------------------------------- |
| Spiral wrap         | 6–10mm   | 1 roll | Protects bundles through joints       |
| Zip ties            | small    | 1 pack | General retention                     |
| Adhesive tie mounts | 10–20mm  | 1 pack | Internal routing anchors              |
| Heat shrink         | assorted |  1 kit | Strain relief on servo leads          |
| Cable braid sleeve  | 6–12mm   |   1–2m | Clean trunk bundle from torso to legs |

---

## G. Plush Interface (so the robot “fits” the bear shell)

| Item           | Spec        |    Qty | Notes                                      |
| -------------- | ----------- | -----: | ------------------------------------------ |
| Soft edge tape | fabric/foam | 1 roll | Prevents sharp edges tearing plush         |
| Hook-and-loop  | sew-on      | 1 pack | Service panels for torso access            |
| Lining fabric  | thin        | 0.5–1m | Optional internal liner to reduce snagging |
| Anti-slip pads | rubber      |    2–4 | Foot traction                              |

## H. Tools (Assembly Essentials)

| Tool                           | Notes                                          |
| ------------------------------ | ---------------------------------------------- |
| Heat-set insert soldering tip  | Makes inserts clean + repeatable               |
| Hex drivers (M3)               | Don’t use cheap allen keys if you can avoid it |
| Calipers                       | Joint fits + shaft/bearing alignment           |
| Small torque driver (optional) | Prevents cracking printed parts                |
| Sanding + deburring kit        | Smooth plush-contact edges                     |


## SERVO LIMITS 

|Joint Group|Lower (rad)    |Upper (rad)      |  Degree Equivalent  |    Reason                                         |
| ----------|---------------|-----------------|---------------------|---------------------------------------------------|
|Hips (Pitch)|-1.0          |      1.0        |         ±57∘        |Prevents the thigh from hitting the belly.         |
|Waist (Roll)|-0.5          |      0.5        |         ±28∘        |Keeps the Center of Mass stable for the Orin Nano. |
|Neck (Pitch)|-0.7          |      0.4        |     −40∘ to +23∘    |Prevents the heavy head from toppling Cara forward.|
|Shoulders   |-1.57         |      1.57       |         ±90∘        |Allows for full expressive waving gestures.        |
