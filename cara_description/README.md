# cara_description

Parameterised description of Cara — currently the pelvis/base link and **one
(left) leg only**. Validated **kinematics** plus a thin **provisional dynamics
layer** for analysis. Foundation for later MuJoCo simulation and
locomotion-policy work; it is **not** a full robot, not CAD, not a policy.

## What this package is (and isn't)

| | |
|---|---|
| ✅ | 6-DoF left-leg kinematic tree: joint origins, axes, limits, purpose |
| ✅ | one parameterised config file as the single source of truth |
| ✅ | generated, inspectable URDF |
| ✅ | validation + forward-kinematics sanity scripts |
| ✅ | COM, gravity-torque, foot-Jacobian and morphology-sweep analysis scripts |
| 🔶 | dynamics — mass / COM / inertia / actuator limits are **provisional, method-tagged placeholders** |
| ❌ | right leg, arms, waist, neck, head |
| ❌ | CAD geometry, servo brackets, wiring, shells |
| ❌ | MuJoCo/MJCF, any RL / control policy |

Design constraint being followed: **kinematics, dynamics, and manufacturing
are kept separate.** The YAML is layered accordingly. No servo is selected;
every dynamic number is labelled `TODO` / `TBD`.

## Layout

```
cara_description/
├── config/
│   └── left_leg.yaml            # SINGLE SOURCE OF TRUTH (edit here)
├── urdf/
│   └── cara_left_leg.urdf       # GENERATED from the YAML — do not hand-edit
├── scripts/
│   ├── leg_model.py             # shared loader + pure-Python kinematics & dynamics
│   ├── generate_urdf.py         # YAML -> URDF
│   ├── validate_description.py  # structural checks (kinematics + dynamics)
│   ├── fk_sanity_check.py       # forward-kinematics behaviour checks
│   ├── center_of_mass.py        # whole-model COM for any joint configuration
│   ├── gravity_torques.py       # gravitational joint torques for reference poses
│   ├── jacobian.py              # foot-position Jacobian + finite-difference validation
│   └── morphology_sweep.py      # effect of a parameter sweep on workspace / COM / torque
├── docs/
│   ├── frames_and_joints.md     # frame conventions + per-joint math + foot frame hierarchy
│   └── dynamics_notes.md        # provisional dynamics layer + the analysis scripts
└── README.md
```

Only dependency beyond the Python standard library is **PyYAML**
(`pip install pyyaml`). **No numpy** — all linear algebra is plain Python.

## Kinematic tree

```
pelvis                        (root / base_link; physical)
└─ l_hip_yaw    Rz  →  l_hip_yaw_link      virtual coupling link (massless)
   └─ l_hip_roll  Rx  →  l_hip_roll_link   virtual coupling link (massless)
      └─ l_hip_pitch Ry  →  l_thigh        segment  (L_thigh; physical)
         └─ l_knee_pitch Ry  →  l_shin     segment  (L_shin; physical)
            └─ l_ankle_pitch Ry → l_ankle_link   virtual coupling link (massless)
               └─ l_ankle_roll Rx → l_foot       segment (physical)
                  • l_foot_sole_center   ground-contact frame
```

The three hip axes (and the two ankle axes) are modelled as **intersecting at
a point**; the virtual coupling links have zero-length joint origins and
**never** carry mass, inertia or collision geometry. Real servo-stack offsets
are a `TODO` deferred until mechanical packaging is designed.

| Joint | Axis | Limits (rad) | + angle means | Purpose |
|-------|------|--------------|---------------|---------|
| `l_hip_yaw`     | `+Z` | −0.79 … 0.79 | toe turns left | leg heading, turning |
| `l_hip_roll`    | `+X` | −0.52 … 0.61 | thigh abducts (+Y) | lateral balance, stance width |
| `l_hip_pitch`   | `+Y` | −1.75 … 1.05 | thigh swings **back** (extension) | stride length |
| `l_knee_pitch`  | `+Y` | 0.0 … 2.36 | shin folds back (flexion) | ground clearance, crouch |
| `l_ankle_pitch` | `+Y` | −0.87 … 0.61 | toe drops (plantarflexion) | push-off, CoP fore/aft |
| `l_ankle_roll`  | `+X` | −0.44 … 0.44 | sole tilts outer-edge-up | CoP lateral, uneven ground |

**Frame convention:** right-handed, `+X` forward, `+Y` left, `+Z` up, origin
at the pelvis. Zero pose = leg straight down, sole flat, toes forward. All
pitch joints share the `+Y` axis for consistency; because `+Y` is left, a
positive rotation about `+Y` is rearward motion — the physical meaning of
"positive" is documented per joint rather than hidden by flipping axes.
Full details: [`docs/frames_and_joints.md`](docs/frames_and_joints.md).

All limits, lengths, masses, and inertias are **provisional placeholders**.
Kinematic geometry is isolated in `provisional_geometry:`; mass / COM /
inertia in `dynamics.links.*`. Neither requires touching the tree structure.

## Usage

```bash
cd cara_description

# validate everything (axes, tree, limits, dynamics structure, virtual links…)
python3 scripts/validate_description.py

# (re)generate the URDF; --check exits non-zero if it has drifted
python3 scripts/generate_urdf.py
python3 scripts/generate_urdf.py --check

# forward-kinematics behaviour checks + a foot-position reference table
python3 scripts/fk_sanity_check.py

# --- analysis (all use the provisional dynamics layer) ---
python3 scripts/center_of_mass.py                       # COM per reference pose
python3 scripts/center_of_mass.py --pose deep_crouch    # per-link breakdown
python3 scripts/gravity_torques.py                      # hold torques, all poses
python3 scripts/gravity_torques.py --carry-fraction 0.5 # + body-weight at the foot
python3 scripts/jacobian.py                             # foot Jacobian + validation
python3 scripts/morphology_sweep.py                     # L_thigh 0.10/0.12/0.14
python3 scripts/morphology_sweep.py --param dynamics.links.l_thigh.mass --values 0.10,0.15,0.20
```

Every script takes an optional config path as its first positional argument,
so morphology variants can be checked side by side.

## The question this layer answers

> *Given Cara's current kinematics, how do geometry and mass distribution
> affect COM, foot motion, and required joint torque?*

- **COM:** `center_of_mass.py` — and `--extra KG@frame` to test battery
  placement or a carried load.
- **Foot motion:** `jacobian.py` — `ẋ_foot = J(q) q̇`, validated against
  finite-difference FK; manipulability per pose.
- **Torque:** `gravity_torques.py` — base-fixed hold torques (`τ = ∂U/∂q`),
  optionally with body weight as a foot load (`τ = Jᵀ F`, cross-checked).
- **Sweeps:** `morphology_sweep.py` — one parameter → workspace, COM, torque
  deltas in a table.

See [`docs/dynamics_notes.md`](docs/dynamics_notes.md) for the math, the
approximations, and the findings.

## Editing the model

- Change parameters **only** in `config/left_leg.yaml`.
- Re-run `generate_urdf.py`, then `validate_description.py`, then the checks.
- Never hand-edit `urdf/cara_left_leg.urdf` — `generate_urdf.py --check`
  returns non-zero if it has drifted (handy for CI / a pre-commit hook).

## Roadmap (not in this task)

1. CAD/measured values replace every `TODO` in `provisional_geometry` and
   `dynamics.links.*`.
2. Real inter-axis offsets replace the coincident-hip/ankle approximation.
3. Right leg by mirroring; attach both legs to a waist/torso chain → full
   20-DoF Cara.
4. A `generate_mjcf.py` beside `generate_urdf.py` so URDF and MuJoCo are both
   generated from the one spec and cannot drift apart. **Virtual coupling
   links must not become MJCF bodies** — represent the coincident hip/ankle
   joints as multiple `<joint>` on one `<body>` (see `dynamics_notes.md` §1).
5. Inverted / stance-leg dynamics model for true body-weight torque.
6. Only then: locomotion-policy training.
