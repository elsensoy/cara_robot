# Cara — Frames & Joints (pelvis + left leg)

Scope: the **kinematic** foundation for the pelvis/base link and one leg, plus
a thin provisional **dynamics** layer for analysis. This document is the
human-readable companion to [`config/left_leg.yaml`](../config/left_leg.yaml),
which is the single source of truth. If a number here disagrees with the YAML,
the YAML wins. Dynamics detail and the analysis scripts are documented
separately in [`dynamics_notes.md`](dynamics_notes.md).

Layer status:

| Layer | Lives where | Status now |
|-------|-------------|------------|
| Kinematics — joint positions, axes, ranges, link lengths | `provisional_geometry`, `links`, `joints` | **defined here** (values provisional) |
| Dynamics — masses, CoM, inertia, actuator torque/velocity limits | `dynamics:` block | **structure defined**, every value provisional / `TODO` / `TBD` — see `dynamics_notes.md` |
| Manufacturing — servo brackets, screw holes, shell thickness, wiring | future `cara_description/meshes/` + CAD | not started |

---

## 1. Coordinate-frame conventions

Right-handed world/base frame, origin at the **pelvis**:

| Axis | Direction | Positive rotation about it (right-hand rule) |
|------|-----------|----------------------------------------------|
| **+X** | forward (Cara's facing direction) | roll: right side of body drops |
| **+Y** | left | pitch: nose/torso tips backward |
| **+Z** | up (opposite gravity) | yaw: nose turns left |

Units are SI and only SI: **m, kg, rad, N·m, s**. No degrees anywhere in the
description (servo calibration offsets in degrees are a separate hardware
concern — repo-root `cara_offsets.yaml`).

### Zero pose

All joint angles `= 0` ⟹ leg fully extended, straight down, foot sole
parallel to the ground, toes pointing `+X`.

At the zero pose **every link frame is axis-aligned with the base frame**.
Consequences used throughout:

- every joint axis is numerically identical in its parent and child frames,
- every joint-origin `rpy` is `(0, 0, 0)`,
- joint motion is pure rotation about one unit axis (all joints revolute).

The kinematic zero is **not** the servo "true neutral" — that offset is
applied downstream by the hardware layer.

### Sign convention (important)

Axes are kept **clean and consistent** rather than flipped to make "positive"
feel intuitive:

- all three pitch joints rotate about **+Y**,
- both roll joints rotate about **+X**,
- yaw rotates about **+Z**.

Because **+Y points left**, a positive right-hand rotation about +Y moves the
distal segment **rearward**. Rather than hide this by negating axes, the
physical meaning of "positive" is documented per joint (below, and in the
YAML `positive_rotation` field). When deriving transforms or Jacobians, use
the clean axes; when commanding or interpreting an angle, consult the
per-joint meaning.

---

## 2. Kinematic tree

```
pelvis                       (base_link / root)
└── l_hip_yaw     (Rz)   ──► l_hip_yaw_link      [virtual coupling]
    └── l_hip_roll   (Rx)   ──► l_hip_roll_link  [virtual coupling]
        └── l_hip_pitch  (Ry)   ──► l_thigh      [segment, length L_thigh]
            └── l_knee_pitch (Ry)   ──► l_shin   [segment, length L_shin]
                └── l_ankle_pitch (Ry) ──► l_ankle_link  [virtual coupling]
                    └── l_ankle_roll  (Rx) ──► l_foot     [segment]
                        • l_foot_sole_center   (fixed frame of interest)
```

- **6 revolute DoF**: `hip_yaw, hip_roll, hip_pitch, knee_pitch,
  ankle_pitch, ankle_roll`. Two legs → 12; the remaining 8 of Cara's 20 DoF
  are waist/neck/ears/shoulders (see `docs/urdf_notes.md`), not modelled here.
- **7 links**. `l_hip_yaw_link`, `l_hip_roll_link`, `l_ankle_link` are
  **virtual coupling links**: the three hip axes are modelled as intersecting
  at a single point (spherical-hip approximation), likewise the two ankle
  axes. Their joint origins are `[0, 0, 0]` with a `TODO` to introduce real
  inter-axis offsets once the servo/bracket stack exists. They are **massless
  abstractions** — no mass, COM, inertia or collision geometry, ever — and the
  generated URDF emits them as bare `<link/>` frames. The four **physical**
  links (`pelvis, l_thigh, l_shin, l_foot`) carry provisional dynamics.
- The tree is a single chain (no branching) for one leg.

---

## 3. Joint-by-joint definition

Notation: for joint *j*, `r_j` is the translation from the parent link frame
to the child link frame at the zero pose, and `a_j` is the unit rotation
axis. Provisional lengths:

```
L_thigh = 0.120 m   L_shin = 0.120 m   h_ankle = 0.035 m
w_hip_half = 0.050 m   (pelvis centre → hip axis, +Y)
```

All limits below are provisional (`TODO: refine from a range-of-motion study`).

### 3.1 `l_hip_yaw`

| | |
|---|---|
| parent → child | `pelvis` → `l_hip_yaw_link` |
| origin `r` | `[x_hip, w_hip_half, z_hip] = [0, 0.050, 0]` m |
| axis `a` | `[0, 0, 1]` |
| limits | `[-0.79, +0.79]` rad (≈ ±45°) |
| +angle means | toe yaws toward +Y (toe-out / outward heading) |
| purpose | rotate the whole leg about vertical — turning, directional foot placement |

$$
r_{\text{hip\_yaw}} = \begin{bmatrix} 0 \\ w_{hip} \\ 0 \end{bmatrix}
= \begin{bmatrix} 0 \\ 0.050 \\ 0 \end{bmatrix},
\qquad
a_{\text{hip\_yaw}} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

### 3.2 `l_hip_roll`

| | |
|---|---|
| parent → child | `l_hip_yaw_link` → `l_hip_roll_link` |
| origin `r` | `[0, 0, 0]` m — *coincident-axis approximation, `TODO` real offset* |
| axis `a` | `[1, 0, 0]` |
| limits | `[-0.52, +0.61]` rad (≈ −30°…+35°) |
| +angle means | thigh abducts — knee moves toward +Y, away from midline |
| purpose | hip abduction/adduction — lateral balance, stance width, side-step |

$$
r_{\text{hip\_roll}} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix},
\qquad
a_{\text{hip\_roll}} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
$$

### 3.3 `l_hip_pitch`

| | |
|---|---|
| parent → child | `l_hip_roll_link` → `l_thigh` |
| origin `r` | `[0, 0, 0]` m — *coincident-axis approximation, `TODO` real offset* |
| axis `a` | `[0, 1, 0]` |
| limits | `[-1.75, +1.05]` rad (≈ −100°…+60°) |
| +angle means | thigh swings **rearward** (hip extension); negative = forward flexion |
| purpose | fore/aft thigh swing — primary driver of stride length |

$$
r_{\text{hip\_pitch}} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix},
\qquad
a_{\text{hip\_pitch}} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

### 3.4 `l_knee_pitch`

| | |
|---|---|
| parent → child | `l_thigh` → `l_shin` |
| origin `r` | `[0, 0, -L_thigh] = [0, 0, -0.120]` m |
| axis `a` | `[0, 1, 0]` |
| limits | `[0.0, +2.36]` rad (0°…≈135°) — flexes one way only |
| +angle means | shin swings rearward relative to thigh (knee **flexion**) |
| purpose | knee flexion — swing-leg ground clearance, shock absorption, crouch depth |

$$
r_{\text{knee}} = \begin{bmatrix} 0 \\ 0 \\ -L_{thigh} \end{bmatrix}
= \begin{bmatrix} 0 \\ 0 \\ -0.120 \end{bmatrix},
\qquad
a_{\text{knee}} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

### 3.5 `l_ankle_pitch`

| | |
|---|---|
| parent → child | `l_shin` → `l_ankle_link` |
| origin `r` | `[0, 0, -L_shin] = [0, 0, -0.120]` m |
| axis `a` | `[0, 1, 0]` |
| limits | `[-0.87, +0.61]` rad (≈ −50°…+35°) |
| +angle means | toe drops (plantarflexion); negative = dorsiflexion |
| purpose | ankle plantar/dorsiflexion — fore/aft CoP control, push-off, toe clearance |

$$
r_{\text{ankle\_pitch}} = \begin{bmatrix} 0 \\ 0 \\ -L_{shin} \end{bmatrix}
= \begin{bmatrix} 0 \\ 0 \\ -0.120 \end{bmatrix},
\qquad
a_{\text{ankle\_pitch}} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

### 3.6 `l_ankle_roll`

| | |
|---|---|
| parent → child | `l_ankle_link` → `l_foot` |
| origin `r` | `[0, 0, 0]` m — *coincident with ankle-pitch axis, `TODO` real offset* |
| axis `a` | `[1, 0, 0]` |
| limits | `[-0.44, +0.44]` rad (≈ ±25°) |
| +angle means | sole tilts outer-edge-up (inversion) |
| purpose | sole tilt — lateral CoP control, uneven-ground adaptation |

$$
r_{\text{ankle\_roll}} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix},
\qquad
a_{\text{ankle\_roll}} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
$$

### Explicit foot frame hierarchy

Three stacked frames, proximal → distal:

| # | Frame | Defined as | Offset from the frame above |
|---|-------|-----------|------------------------------|
| 1 | **ankle joint frame** | origin of `l_ankle_link` (= the `l_ankle_roll` joint frame; ankle-pitch and ankle-roll axes intersect here) | — |
| 2 | **foot body frame** | `l_foot` link frame | `l_ankle_roll.origin = [0, 0, 0]` — coincident-axis approx, `TODO` real offset. So (1) and (2) coincide at the zero pose. |
| 3 | **sole / contact frame** | `l_foot_sole_center` (`frames_of_interest`) — centre of the ground-contact patch | `[foot_x_off, 0, -h_ankle] = [+0.015, 0, -0.035]` m |

**The +0.015 m X offset of the zero-pose sole comes entirely from one
parameter: `provisional_geometry.foot_x_off`.** Nothing else in the model
produces an X offset at the zero pose. It is **intentional in sign**
(ankle-roll axis sits *behind* the foot centre, over the rear third of the
foot — correct for a forward-pointing foot) and **provisional in magnitude**
(`0.015 m` is a guess; `TODO` set from CAD). Do not "fix" it to zero — that
would centre the ankle on a symmetric foot, which is wrong. See
[`dynamics_notes.md` §3](dynamics_notes.md) for the full discussion and its
one visible consequence (a ~0.007 N·m gravity torque at the zero pose).

`l_foot_sole_center` is the foot reference point for the FK sanity check, the
Jacobian, the torque scripts and, later, contact geometry.

---

## 4. Forward kinematics

Each joint contributes a translation then a rotation:

$$
T^{parent}_{child}(q_j) =
\underbrace{\begin{bmatrix} I_3 & r_j \\ 0 & 1 \end{bmatrix}}_{\text{fixed origin}}
\cdot
\underbrace{\begin{bmatrix} R(a_j, q_j) & 0 \\ 0 & 1 \end{bmatrix}}_{\text{joint rotation}}
$$

where `R(a, q)` is the Rodrigues rotation of `q` rad about unit axis `a`.
The base→foot transform is the ordered product down the chain:

$$
T^{\text{pelvis}}_{\text{l\_foot}}(q) =
T^{\text{pelvis}}_{\text{l\_hip\_yaw\_link}}(q_1)\;
T^{\cdot}_{\text{l\_hip\_roll\_link}}(q_2)\;
T^{\cdot}_{\text{l\_thigh}}(q_3)\;
T^{\cdot}_{\text{l\_shin}}(q_4)\;
T^{\cdot}_{\text{l\_ankle\_link}}(q_5)\;
T^{\cdot}_{\text{l\_foot}}(q_6)
$$

### Zero-pose foot position (closed form)

With all `q = 0` and the coincident hip/ankle approximations:

$$
p_{\text{sole}}(0) =
\begin{bmatrix} x_{hip} + \text{foot\_x\_off} \\ w_{hip} \\
z_{hip} - (L_{thigh} + L_{shin} + h_{ankle}) \end{bmatrix}
= \begin{bmatrix} 0.015 \\ 0.050 \\ -0.275 \end{bmatrix} \text{ m}
$$

`scripts/fk_sanity_check.py` asserts exactly this, plus the directional
behaviour of each joint sweep and the reach bound
`‖hip → ankle‖ ≤ L_thigh + L_shin` (equality iff knee = 0).

---

## 5. Provisional values & how to sweep them

Unmeasured **kinematic** numbers live in one block: `provisional_geometry`.
Unmeasured **dynamic** numbers live in `dynamics.links.*` (mass / com /
inertia). Nothing in `links:` or `joints:` contains a raw dimension — joint
origins and COMs reference the symbols by name (`[0, 0, -L_thigh]`,
`[0, 0, -L_thigh/2]`).

`scripts/morphology_sweep.py` overrides one dotted path in memory and
recomputes workspace / COM / torque:

```bash
python3 scripts/morphology_sweep.py --param provisional_geometry.L_thigh --values 0.10,0.12,0.14
python3 scripts/morphology_sweep.py --param dynamics.links.l_thigh.mass  --values 0.10,0.15,0.20
```

Or copy the YAML, change only the block you want, and re-run every check
against the copy (each script takes an optional config path):

```bash
python3 scripts/generate_urdf.py       config/variant_A.yaml -o urdf/variant_A.urdf
python3 scripts/generate_mjcf.py       config/variant_A.yaml -o mjcf/variant_A.xml
python3 scripts/validate_description.py config/variant_A.yaml
python3 scripts/fk_sanity_check.py      config/variant_A.yaml
python3 scripts/center_of_mass.py       config/variant_A.yaml
python3 scripts/gravity_torques.py      config/variant_A.yaml
```

---

## 6. URDF and MJCF from one spec

`generate_urdf.py` → `urdf/cara_left_leg.urdf` and `generate_mjcf.py` →
`mjcf/cara_left_leg.xml` both read this YAML; neither output is hand-edited
(each has a `--check` flag that fails on drift). The frame convention, joint
origins/axes/limits and the coincident hip/ankle abstraction are preserved
identically in both.

MJCF specifics: the virtual coupling links are represented as **stacked
`<joint>` on the physical body downstream** (MuJoCo needs positive mass on any
DOF-carrying body, so they are not bodies and get no fake inertia). The model
is kinematics-only for now — gravity off, geoms non-colliding, no actuators.
`scripts/validate_mjcf.py` compiles it in MuJoCo and confirms every body pose
and the sole site match `forward_kinematics` here to < 1e-16 m across all
reference poses. Details in the README "MJCF / MuJoCo" section.

---

## 7. Open TODOs before this is "real"

- [ ] Replace all `provisional_geometry` values with CAD/measured numbers.
- [ ] Introduce real inter-axis offsets for hip yaw/roll/pitch and ankle
      pitch/roll (removes the coincident-axis approximation). Both generators
      already handle non-zero inter-axis offsets.
- [ ] Replace `method`-tagged approximate inertia with CAD inertia tensors +
      true CoM (`dynamics.links.*`; see `dynamics_notes.md`).
- [ ] Choose servos → real `effort` / `velocity` limits (`dynamics.actuators`
      is all TBD).
- [ ] Range-of-motion study → real joint limits.
- [ ] Mirror to the right leg; attach both to the waist/torso chain.
- [ ] Turn on gravity + add actuators in the MJCF for dynamics/control work.
