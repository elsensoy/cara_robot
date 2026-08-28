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
| ✅ | generated, inspectable URDF **and** MJCF from that one spec |
| ✅ | MuJoCo model verified to reproduce the pure-Python FK to machine precision |
| ✅ | validation + forward-kinematics sanity scripts |
| ✅ | COM, gravity-torque, foot-Jacobian and morphology-sweep analysis scripts |
| 🔶 | dynamics — mass / COM / inertia / actuator limits are **provisional, method-tagged placeholders** |
| ❌ | right leg, arms, waist, neck, head |
| ❌ | CAD geometry, servo brackets, wiring, shells |
| ❌ | actuators, RL / control policy (MJCF has gravity off, no motors) |

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
├── mjcf/
│   └── cara_left_leg.xml        # GENERATED from the YAML — do not hand-edit
├── scripts/
│   ├── leg_model.py             # shared loader + pure-Python kinematics & dynamics
│   ├── generate_urdf.py         # YAML -> URDF
│   ├── generate_mjcf.py         # YAML -> MJCF (same source of truth)
│   ├── validate_description.py  # structural checks (kinematics + dynamics)
│   ├── fk_sanity_check.py       # forward-kinematics behaviour checks
│   ├── validate_mjcf.py         # MuJoCo body/site positions vs leg_model FK
│   ├── view_mujoco.py           # load the generated MJCF and open mujoco.viewer
│   ├── center_of_mass.py        # whole-model COM for any joint configuration
│   ├── gravity_torques.py       # gravitational joint torques for reference poses
│   ├── jacobian.py              # foot-position Jacobian + finite-difference validation
│   └── morphology_sweep.py      # effect of a parameter sweep on workspace / COM / torque
├── docs/
│   ├── frames_and_joints.md     # frame conventions + per-joint math + foot frame hierarchy
│   └── dynamics_notes.md        # provisional dynamics layer + the analysis scripts
└── README.md
```

One spec, two robot descriptions — never edit URDF or MJCF by hand:

```
                    ┌──> urdf/cara_left_leg.urdf
    left_leg.yaml ──┤
                    └──> mjcf/cara_left_leg.xml
```

**Dependencies:** **PyYAML** (`pip install pyyaml`) for everything; **mujoco**
(`pip install mujoco`) only for `validate_mjcf.py` and `view_mujoco.py`.
**No numpy** — all linear algebra is plain Python.

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

# (re)generate URDF and MJCF; --check exits non-zero if either has drifted
python3 scripts/generate_urdf.py   &&  python3 scripts/generate_urdf.py --check
python3 scripts/generate_mjcf.py   &&  python3 scripts/generate_mjcf.py --check

# forward-kinematics behaviour checks + a foot-position reference table
python3 scripts/fk_sanity_check.py

# confirm YAML -> MJCF -> MuJoCo reproduces the pure-Python FK  (needs mujoco)
python3 scripts/validate_mjcf.py

# inspect the model in the MuJoCo viewer, loaded programmatically  (needs mujoco)
python3 scripts/view_mujoco.py --regen
python3 scripts/view_mujoco.py --pose deep_crouch

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

## MJCF / MuJoCo

`generate_mjcf.py` writes `mjcf/cara_left_leg.xml` from the same YAML. Design
choices (documented in the file header and the script docstring):

- **Coincident abstraction, no fake mass.** MuJoCo requires positive mass on
  any body with a DOF, so the three virtual coupling links are *not* bodies.
  The coincident joints are stacked on the physical body downstream —
  `l_hip_yaw/roll/pitch` as three `<joint>` on `l_thigh`, both ankle joints on
  `l_foot`. This is mathematically identical to the URDF chain; the generator
  emits per-joint `<joint pos>` anchors so it also stays correct if real
  inter-axis offsets are added later.
- **Kinematics-only.** `option gravity="0 0 0"`, all link geoms non-colliding
  (`contype=conaffinity=0`), no actuators. The viewer shows the pose you set;
  the leg does not sag. A visual ground plane sits at the zero-pose sole
  height. Turn gravity on and add actuators when dynamics work begins.
- **Geometry from the YAML only.** Primitive box/cylinder geoms with sizes
  straight from `dynamics.links.*` — no new numbers.
- **Load programmatically.** `view_mujoco.py` builds the model with
  `mujoco.MjModel.from_xml_path(...)` so the whole pipeline is reproducible —
  don't use the viewer's *File > Open* as the normal workflow.

`validate_mjcf.py` compiles the model and checks every physical body's world
position + orientation, and the `l_foot_sole_center` site, against
`leg_model.forward_kinematics` for all reference poses. Current agreement:
**< 1e-16 m** — the MJCF reproduces the already-validated kinematics exactly.

## Editing the model

- Change parameters **only** in `config/left_leg.yaml`.
- Re-run `generate_urdf.py` **and** `generate_mjcf.py`, then
  `validate_description.py`, then the checks (`validate_mjcf.py` included).
- Never hand-edit `urdf/cara_left_leg.urdf` or `mjcf/cara_left_leg.xml` — the
  `--check` flag on each generator returns non-zero if it has drifted (handy
  for CI / a pre-commit hook).

## Roadmap (not in this task)

1. CAD/measured values replace every `TODO` in `provisional_geometry` and
   `dynamics.links.*`.
2. Real inter-axis offsets replace the coincident-hip/ankle approximation.
   (`generate_mjcf.py` already emits per-joint `<joint pos>` anchors, so this
   works without a rewrite.)
3. Right leg by mirroring; attach both legs to a waist/torso chain → full
   20-DoF Cara.
4. Inverted / stance-leg dynamics model for true body-weight torque.
5. Turn gravity on in the MJCF and add actuators — the start of dynamics /
   control work.
6. Only then: locomotion-policy training.
