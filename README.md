# 🐻 Project Cara

**An embodied, emotionally adaptive companion robot** — a 20-DoF articulated
teddy bear that learns to see, walk, and care.

> *Author: Elida Sensoy*
> *Platform: NVIDIA Jetson Orin Nano Super + Arduino Nano · ROS 2 Humble · Docker*
> *Stack: MuJoCo (physics + sim-to-real) · TensorRT · ViT emotion recognition · reinforcement learning · Gemini LLM*

Cara is intended as a warm companion, particularly for people facing chronic
illness, disability, or isolation. She is built around three intertwined ideas:

- **A body** — a 20-DoF humanoid teddy-bear skeleton, validated in simulation before hardware.
- **A face for the world** — a personalized Vision Transformer that learns *your* expressions, not a generic dataset's.
- **A self** — a homeostatic loop where motion, emotion, and memory are modulated by Cara's own internal "vitals."

https://github.com/user-attachments/assets/e023d652-621f-4e27-8c53-0dc29c00cf4d

---

## Design philosophy

Most systems treat perception, emotion, and motion as separate subsystems glued
together with rules. Cara assumes the opposite: **understanding is downstream of
agency.**

```
Perception → Internal state → Motion policy → Body → Feedback
```

Emotion and memory influence *how* motion is generated — they shape the
constraints and targets of the controller, and never override safety.

1. **Emotion is a control signal, not a label.** Detected affect modifies posture, gait tempo, and stiffness.
2. **One brain, two worlds.** The same policy runs in MuJoCo and on the Jetson — no hand-coded gaits, no animation layers.
3. **Self-maintenance is primary.** When Cara's internal vitals (thermal margin, actuator health, sensor confidence) degrade, behavior adapts — even at the cost of the current task.

---

## System architecture

A **sense → think → act** pipeline across ROS 2 nodes that mirror the structure
used in simulation (inspired by Disney Research's articulated-character control):

```
environment
  → physical Cara (servos, battery, IMU, Jetson)
  → telemetry aggregation (Arduino + INA219 + IMU, 50/10/1 Hz)
  → fault fusion + self-state estimate (health_node → wellness W(t))
  → policy arbiter (constrains the action space by wellness)
  → actuators → back to the environment
```

**Example loop closure (~1 s):** a hip servo stalls against an obstacle → the
Arduino fast layer flags a suspected stall, the INA219 sees a current spike, the
IMU sees no motion → `health_node` fuses the three into a confirmed fault and
drops wellness to 0.55 → the arbiter restricts to homeostatic actions and
commands REST → the leg moves to a safe pose, current falls, and over the next
second wellness recovers and the full action space is restored — with the stall
logged in episode memory.

### Core ROS 2 nodes

| Node | Package | Role |
|---|---|---|
| `imu_node` | `cara_motion_control` | Fused orientation from the BNO055 IMU |
| `face_yunet_node` | `cara_vision_control` | YuNet face detector — crops the face, publishes its center |
| `emotion_node` | `cara_vision_control` | ViT emotion inference + on-device fine-tune |
| `model_gaze_mapper` | `cara_gaze_control` | Face position → `/head_cmd` for neck pan/tilt tracking |
| `servo_pca9685_node` | `cara_vision_control` | Drives neck servos via PCA9685 |
| `behavior_node` | `cara_vision_control` | Emotion + mind-mode → blink-rate commands |
| `arduino_bridge_node` | `cara_vision_control` | Forwards blink commands to the Arduino |
| `policy_node` | `cara_motion_control` | Runs the trained RL locomotion policy (ONNX) |
| `actuator_node` | `cara_motion_control` | Drives 20 servos via dual PCA9685 boards |
| `health_node` | `cara_health` | Distress signals (thermal, battery, actuator load) |
| `cara_control_node` | `cara_control` | Servo-rail health → policy observation → controller (C++, observation-only) |

All nodes share a `ROS_DOMAIN_ID=7` environment. Three pipelines run in parallel
off the same vision input and own separate outputs:

| Hardware output | Owned by | Topic |
|---|---|---|
| Neck pan / tilt | `model_gaze_mapper` | `/head_cmd` |
| Eye blink / LEDs | `behavior_node` | `/cara/behavior_cmd` |
| Speech + language | `app/cara.py` (`CaraMind`) | `/cara/mind_mode`, direct TTS |

`CaraMind` classifies each exchange into a **mind-state mode** —
`ALPHA_SUPPORTIVE` (negative affect), `BETA_REASONING` (task/technical),
`THETA_CREATIVE` (generative), `NEUTRAL_OBSERVE` (ambiguous) — which routes
memory and sets the blink style.

```bash
# inside the container
source /workspace/install/setup.bash
ros2 launch cara_bringup cara_stack.launch.py use_servos:=true
cd /workspace/app && python3 cara.py     # separate terminal — language brain
```

---

## Embodied motion & control

Everything downstream — simulation, control, hardware mapping — is generated from
**one parameterised description**, [`cara_description/`](cara_description/README.md):

```
config/left_leg.yaml          SSOT: one leg + pelvis (fixed base)
  └─ cara_lower_body.yaml      + mirror l_→r_ + floating pelvis + poses
       └─ cara_full_body.yaml  + include cara_upper_body.yaml (torso, head/neck, electronics, arms, ears)

each model → urdf/<model>.urdf            (ROS 2 / ros2_control)
           → mjcf/<model>.xml             (MuJoCo, kinematic)
           → mjcf/<model>_dynamic.xml     (MuJoCo, gravity + PD + contact)
```

URDF and MJCF are *generated*, never hand-edited, so they cannot drift apart.
The model is built and checked in **strict stages**, so a morphology bug can
never hide inside a half-trained policy:

| Stage | Result |
|---|---|
| 1-leg kinematics → dynamics → MuJoCo dynamic validation | ✅ frame conventions, provisional mass/inertia (all `TODO`-marked), torques cross-checked against an analytic layer |
| 2 legs + floating pelvis · static standing · COM / support-polygon | ✅ **standing milestone met** — tilt ≤ 0.3°, COM margin 33–43 mm |
| Quasi-static weight shifting (task-space IK) | ✅ **milestone met** — double-support limit ~0.04 m COM |
| **U1–U6** upper body — torso, head/neck, Jetson/battery placement, passive arms, ears | ✅ each subsystem measured vs a frozen baseline; full body **4.43 kg**, weight-shift envelope tightens to ±0.020 m; morphology validation closed |
| **U7–U11** balance & stepping — unload a foot, single-support lift, COM-feedback balance, one forward step, a short walk | ✅ 5 s one-foot hold; **4 alternating steps forward (110 mm), periodic cycle** — quasi-static *stepping*, not a dynamic gait |
| **U12** continuous *kinematic* walk | ❌ formulation wall — topples at the double-support transfer (fast) / no forward drive (slow) |
| **U13** reduced-order model (LIPM + capture point) | ✅ **a dynamic walk IS within Cara's morphology** — lateral step time ≥ ~0.22 s; ω₀ cross-checks against MuJoCo |
| **U14–U15** DCM-tracking controller + **torque-controlled ankles** | 🔶 torque ankles working (default MJCF byte-identical, standing verified); the walk still doesn't complete — **gait initiation** from rest is the open piece |
| **U16** gait initiation fixed | 🔶 the from-rest warm-up rock now survives cleanly at any length and reaches the first real forward step (7/14 steps vs. U15's 1/14); the double-support → single-support **handoff** at first liftoff is the new, narrower blocker |
| **U17** — settle the handoff / ZMP-preview / MPC → RL policy | ⬜ next |

Full detail and the validation scripts:
[`cara_description/README.md`](cara_description/README.md) and
[`cara_description/docs/`](cara_description/docs/).

### Walking as a learned stability problem

Once the 20-DoF model is trusted, walking is trained as reinforcement learning in
MuJoCo (MJX for parallel rollouts), where balance, energy efficiency, and
recoverability dominate raw speed — Cara learns to walk *sustainably*, not as
fast as possible.

- **Observations:** 20 joint positions + 20 velocities, base linear velocity, IMU orientation (domain-randomized).
- **Actions:** target joint positions, mapped 1:1 to servo commands.
- **Reward:** forward velocity − energy penalty − stability penalty + a thermal proxy that encourages an alternating gait (lets the motors "rest").

```
MuJoCo (from cara_description) → RL policy → ONNX → ROS 2 → PCA9685 → servos
```

Control runs at **50 Hz** in sim and on hardware for direct transfer.

### Emotion as a motion modifier

Emotional state is a low-dimensional continuous vector that **shapes policy
targets**, not a set of discrete animations: *sad* → reduced stride, forward
torso bias; *curious* → head-leading motion, raised ears; *excited* → higher
tempo without torque spikes; *happy* → upright posture. **No emotional state
overrides stability or safety.**

### Joint topology (20 DoF)

| Region | DoF | Representative limits |
|---|---|---|
| Waist (pitch/yaw/roll) | 3 | roll ±28° — keeps CoG stable for the Orin Nano |
| Neck (pitch/yaw/roll) | 3 | pitch −40°…+23° — heavy head can't topple forward |
| Shoulders (2 × 3-DoF) | 6 | ±90° — expressive gestures |
| Hips (2 × 3-DoF) | 6 | pitch ±57° — highest-load joints |
| Ears (1-DoF each) | 2 | expressive, pinned |

Full per-joint axes, limits, and rationale:
[`cara_description/docs/frames_and_joints.md`](cara_description/docs/frames_and_joints.md).
Dynamic parameters (PD gains 30–45 N·m/rad, ±2–3 N·m effort ceiling, ground
friction, 50 Hz) are provisional and `TODO`-marked until real servos are chosen;
`dynamic_check.py` already flags that ±3 N·m servos saturate in a loaded crouch —
knee torque is the first real number to pin down.

---

## Personalized emotion recognition

Standard FER models (FER-2013 and similar) miss an individual's
micro-expressions. Cara uses a **pre-trained ViT-Tiny with a lightweight
trainable adapter head**, personalized on-device via human-in-the-loop feedback.

**Pipeline:** 640×480 @ 30 FPS → YuNet face detection → crop/resize to 224×224 →
`vit-tiny-patch16-224` CLS feature (192-dim) → MLP head → 7 emotion
probabilities.

**Why ViT:** self-attention gives global context immediately — analyzing a
smile, mouth patches *attend to* eye patches, so a real (Duchenne) smile is
recognized as curved mouth **plus** crinkled eyes.

**Parameter-efficient fine-tuning:** the backbone is frozen (no catastrophic
forgetting); only a few thousand adapter parameters update, so personalization
finishes in seconds on the Jetson. A typical run drops loss from ~1.26 to ~0.74
over 10 epochs.

```bash
ros2 run cara_vision_control emotion_node
ros2 topic pub --once /cara/feedback std_msgs/msg/String "{data: 'happy'}"   # label an expression
ros2 topic pub --once /cara/train    std_msgs/msg/Bool   "{data: true}"      # train after ~200–300 samples
```

**Robustness:** grouped train/val splitting (no day-to-day leakage), temperature
scaling (calibrated confidence), asymmetric augmentation (train hard, test
clean), label smoothing, and a rolling frame buffer for "save what you just saw"
corrections.

The detected emotion is injected into the LLM system prompt and simultaneously
drives expressive servos (head-tracking speed, head tilt, ear pose, speech
tone).

---

## Toward understanding: homeostatic agency

**Hypothesis:** understanding is a regulatory achievement, not a representational
one. A system that only predicts or generates never *needs* to understand;
understanding emerges when a system must act, persist, and regulate itself in a
world that can damage it. Cara adds the loop most systems skip — self-state
estimation between action and reward.

### Self-state vector

Vitals about *Cara*, not the user or the task:

| Variable | Source |
|---|---|
| Energy `E` | battery percentage |
| Thermal `T` | Jetson + servo temperatures |
| Sensor confidence `C` | mic SNR, camera blur, STT confidence, dropped frames |
| Actuator health `A` | servo error counts, stall events, current spikes |
| Cognitive load `L` | token budget, API latency, queue backlog |

A scalar wellness `W` weights these against setpoints. Each candidate action is
scored

```
Score(a) = α·ΔH(a) + β·U·ΔU(a) − γ·Cost(a) − ρ·Risk(a)
```

(`ΔH` = homeostatic deficit reduction, `U` = curiosity drive gated by wellness,
`ΔU` = expected information gain). **Hard constraint:** if `E < 0.2`, `T < 0.3`,
or `A < 0.5`, only homeostatic actions are allowed.

Curiosity is bounded prediction error — a simple world model predicts the next
observation, and its error drives exploration *only* when wellness is high
enough, which prevents the "curiosity spiral" of a system that explores while
it's failing.

**In practice:**
- Noisy room → STT confidence drops → Cara asks the user to move closer instead of guessing.
- Servo error counts climb during repeated gestures → Cara slows down and switches to a sitting policy.
- Jetson hits 75 °C → `health_node` publishes `critical` → ears droop (sad) while the body switches to low-energy mode.

Memory stores **agent episodes** `(S, o, a, outcome, PE, W)`, not just
conversations — which is what lets Cara learn "loud rooms break my hearing"
without anyone teaching her that.

### Actuator-health signal path (deployment prototype)

[`jetson/control/`](jetson/control/README.md) is the non-RL prototype of the
Jetson-side loop: it reads servo-rail power + IMU, estimates a **health scalar**
in `[0, 1]`, appends it to the policy observation vector, and runs a
hand-written controller whose motion authority scales with that health. It is
**observation-only** (publishes telemetry, sends no servo commands), so it runs
safely alongside the gaze/behavior pipelines; when an RL policy is trained, only
the controller stage is swapped. Brought up by `cara_stack.launch.py`
(`health_source:=sim` for a synthetic timeline with a scripted fault, or `hw`
for the real BNO055 + INA219). Full topic / parameter reference:
[`jetson/control/README.md`](jetson/control/README.md).

---

## Hardware

| Component | Role | Power |
|---|---|---|
| Jetson Orin Nano Super | GPU inference, ROS 2 | 9–20 V DC, ~150 g |
| Arduino Nano (CH340) | Servo & LED logic | logic-only |
| 2× PCA9685 | PWM expansion (I²C 0x40 / 0x41) | 5 V logic |
| 20× metal-gear servos | Joints (high-torque hips/waist) | isolated 5–6 V BEC, ~1.1 kg |
| BNO055 IMU | Fused orientation | logic-only |
| 2S LiPo 5000 mAh | Main power | ~250 g, low in the torso |
| USB camera | Vision (Arducam U20CAM-1080P) | USB |
| Mic + speakers | Voice I/O | USB / 3.5 mm |

**Target total mass ~2.0 kg**; battery and Jetson sit low to keep the CoG
stable.

**Power isolation (critical):** the LiPo splits to a 5/6 V BEC for the servo
rail and a separate regulator for the Jetson; PCA9685 logic is powered from the
Jetson side. **All grounds must be common** — failing to tie BEC GND to Jetson
GND causes unstable PWM, IMU noise, and risks back-EMF damage to the carrier
board.

![Cara blueprint](media/images/cara_blueprint_general_idea.png)

---

## Repo structure

| Path | What it is |
|---|---|
| `cara_description/` | **Robot description + simulation model.** Composed YAML → generated URDF + MJCF; validation + analysis scripts; frozen `baselines/` for regression. Full body (4.43 kg) stands, weight-shifts, and balances on one foot; dynamic-gait work in progress. |
| `ros2_ws/` | ROS 2 workspace — vision, gaze, emotion, behavior, health, and `policy_node`. |
| `jetson/` | Jetson-side deployment loop (C++): servo-rail power + IMU → health scalar → controller. |
| `app/` | Cara's language brain (`cara.py`, Gemini, memory). |
| `arduino/` | Nano firmware — eye LEDs / blink / head servos. |
| `training/` | Docker + config for policy-training runs. |
| `urdf/` | Pre-`cara_description` hand-written Xacro sketches — being superseded limb by limb. |
| `media/`, `configs/`, `cara_offsets.yaml` | Renders/videos, controller configs, hardware calibration offsets. |

---

## Quick start

```bash
# 1. build + start the runtime
docker compose build cara_runtime
xhost +local:root && docker compose up -d cara_runtime
docker compose exec cara_runtime bash

# 2. inside the container — vision stack + brain (separate terminals)
source /workspace/ros2_ws/install/setup.bash
ros2 launch cara_vision_control cara_runtime.launch.py
cd /workspace/app && python3 cara.py

# 3. simulation quick check (needs: pip install mujoco pyyaml)
cd cara_description
python3 scripts/validate_description.py config/cara_lower_body.yaml
python3 scripts/stand_check.py
python3 scripts/view_mujoco.py --dynamic --config config/cara_lower_body.yaml --regen --pose semi_squat
```

**Environment** — create `.env` at the project root:

```bash
ELEVENLABS_API_KEY=<key>
GEMINI_API_KEY=<key>
CARA_SERIAL_PORT=/dev/ttyUSB0
UID=1000
GID=1000
```

If `/dev/ttyUSB*` is missing: `sudo modprobe ch341 usbserial` and
`sudo usermod -a -G dialout $USER`. Camera calibration is one-time via
`camera_calibration cameracalibrator` with an 8×6 / 24 mm checkerboard; save the
YAML into `camera_info/`.

---

## Safety

Mirrored in simulation and on hardware: a physical **E-stop** on the servo rail ·
**free-fall detection** (IMU → protective tuck) · **thermal watchdog**
(`health_node` throttles movement, publishes `critical`) · **rate limits** on
movement and cloud API calls · **hard action gates** — only homeostatic actions
when wellness is critical.

---

## Further reading

- [`cara_description/README.md`](cara_description/README.md) — the parameterised description, the YAML → URDF/MJCF pipeline, the staged build
- [`cara_description/docs/frames_and_joints.md`](cara_description/docs/frames_and_joints.md) — coordinate conventions, per-joint math, the foot frame hierarchy
- [`cara_description/docs/dynamics_notes.md`](cara_description/docs/dynamics_notes.md) — provisional mass/COM/inertia, gravity-torque and Jacobian analysis
- [`cara_description/docs/standing_notes.md`](cara_description/docs/standing_notes.md) — mirroring the second leg, the floating-base rig, the standing milestone
- [`cara_description/docs/weight_shift_notes.md`](cara_description/docs/weight_shift_notes.md) — the task-space IK layer and the weight-shift milestone
- [`cara_description/docs/single_support_notes.md`](cara_description/docs/single_support_notes.md) — U7 → U16: unloading a foot → single-support balance → stepping → the DCM-tracking dynamic-walk work
- [`cara_description/docs/upper_body_notes.md`](cara_description/docs/upper_body_notes.md) — the composed config hierarchy and the staged upper-body mass/inertia analysis (U1–U6)
- [`jetson/control/README.md`](jetson/control/README.md) — the actuator-health controller: topics, parameters, launch arguments

---

*Cara is an ongoing research and engineering project.*
