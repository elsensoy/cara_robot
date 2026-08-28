# cara_description

Parameterised description of Cara — currently the pelvis/base link and **one
(left) leg only**. Validated **kinematics**, a **provisional dynamics layer**
(mass / COM / inertia / actuator + PD gains), and a **dynamic MuJoCo model**
verified to move plausibly under gravity + PD control. Foundation for later
whole-robot simulation and locomotion-policy work; it is **not** a full robot,
not CAD, not a policy, and not yet a floating-base / balancing model.

## What this package is (and isn't)

| | |
|---|---|
| ✅ | 6-DoF left-leg kinematic tree: joint origins, axes, limits, purpose |
| ✅ | one parameterised config file as the single source of truth |
| ✅ | generated, inspectable URDF **and** MJCF (kinematic + dynamic) from that one spec |
| ✅ | MuJoCo model verified to reproduce the pure-Python FK to machine precision |
| ✅ | dynamic MJCF: gravity + PD servos + foot–ground contact, validated over scripted poses |
| ✅ | validation + forward-kinematics + dynamic-plausibility scripts |
| ✅ | COM, gravity-torque, foot-Jacobian and morphology-sweep analysis scripts |
| 🔶 | dynamics — mass / COM / inertia / actuator limits / PD gains are **provisional placeholders** |
| ❌ | right leg, arms, waist, neck, head |
| ❌ | CAD geometry, servo brackets, wiring, shells |
| ❌ | floating-base / stance test, RL / locomotion policy |

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
│   ├── cara_left_leg.xml          # GENERATED — kinematic (gravity off, no motors)
│   └── cara_left_leg_dynamic.xml  # GENERATED — gravity + PD servos + foot contact
├── scripts/
│   ├── leg_model.py             # shared loader + pure-Python kinematics & dynamics
│   ├── generate_urdf.py         # YAML -> URDF
│   ├── generate_mjcf.py         # YAML -> MJCF  (--dynamic for the physics model)
│   ├── validate_description.py  # structural checks (kinematics + dynamics)
│   ├── fk_sanity_check.py       # forward-kinematics behaviour checks
│   ├── validate_mjcf.py         # MuJoCo body/site positions vs leg_model FK
│   ├── dynamic_check.py         # gravity + PD plausibility test over the reference poses
│   ├── view_mujoco.py           # load a generated MJCF and open mujoco.viewer
│   ├── center_of_mass.py        # whole-model COM for any joint configuration
│   ├── gravity_torques.py       # gravitational joint torques for reference poses
│   ├── jacobian.py              # foot-position Jacobian + finite-difference validation
│   └── morphology_sweep.py      # effect of a parameter sweep on workspace / COM / torque
├── docs/
│   ├── frames_and_joints.md     # frame conventions + per-joint math + foot frame hierarchy
│   └── dynamics_notes.md        # provisional dynamics layer + the analysis scripts
└── README.md
```

One spec, every robot description — never edit URDF or MJCF by hand:

```
                    ┌──> urdf/cara_left_leg.urdf
    left_leg.yaml ──┼──> mjcf/cara_left_leg.xml           (kinematic)
                    └──> mjcf/cara_left_leg_dynamic.xml   (gravity + PD + contact)
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

# (re)generate URDF and both MJCFs; --check exits non-zero on drift
python3 scripts/generate_urdf.py             && python3 scripts/generate_urdf.py --check
python3 scripts/generate_mjcf.py             && python3 scripts/generate_mjcf.py --check
python3 scripts/generate_mjcf.py --dynamic   && python3 scripts/generate_mjcf.py --dynamic --check

# forward-kinematics behaviour checks + a foot-position reference table
python3 scripts/fk_sanity_check.py

# confirm YAML -> MJCF -> MuJoCo reproduces the pure-Python FK  (needs mujoco)
python3 scripts/validate_mjcf.py
python3 scripts/validate_mjcf.py --dynamic

# dynamic plausibility: gravity + PD control over every reference pose  (needs mujoco)
python3 scripts/dynamic_check.py
python3 scripts/dynamic_check.py --verbose

# inspect in the MuJoCo viewer, loaded programmatically  (needs mujoco)
python3 scripts/view_mujoco.py --regen
python3 scripts/view_mujoco.py --dynamic --pose knee_lift

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
- **Two modes from one generator.**
  - *kinematic* (default → `cara_left_leg.xml`): `gravity="0 0 0"`, geoms
    non-colliding, no actuators — the viewer shows the pose you set, nothing
    sags. This is the pose-inspection / FK-reference model.
  - *dynamic* (`--dynamic` → `cara_left_leg_dynamic.xml`): gravity on
    (`analysis.gravity`), a PD `<position>` actuator per joint
    (`dynamics.actuators.control` gains, `forcerange = ±effort`), the foot box
    colliding with a ground plane (`analysis.ground`), and a `<keyframe>` per
    reference pose. Pelvis welded to the world — a fixed-base single-leg rig.
- **Geometry from the YAML only.** Primitive box/cylinder geoms with sizes
  straight from `dynamics.links.*` — no new numbers.
- **Load programmatically.** `view_mujoco.py` builds the model with
  `mujoco.MjModel.from_xml_path(...)` so the whole pipeline is reproducible —
  don't use the viewer's *File > Open* as the normal workflow.

`validate_mjcf.py` (`--dynamic` optional) compiles the model and checks every
physical body's world position + orientation, and the `l_foot_sole_center`
site, against `leg_model.forward_kinematics` for all reference poses —
agreement **< 1e-16 m**.

`dynamic_check.py` runs the dynamic model: it commands the PD servos to each
reference pose, lets it settle, and reports pose-tracking error, jitter, peak
actuator torque + saturation, `|τ − τ_gravity|` against the analytic layer,
foot-contact gap / normal force, and MuJoCo-vs-FK error. Current result: all
poses settle with **zero residual velocity**, and airborne hold torques match
`gravity_torques.py` to 4+ decimals. See `docs/dynamics_notes.md` §8.

## Editing the model

- Change parameters **only** in `config/left_leg.yaml`.
- Re-run `generate_urdf.py`, `generate_mjcf.py` **and**
  `generate_mjcf.py --dynamic`, then `validate_description.py`, then the checks.
- Never hand-edit the generated URDF / MJCF files — each generator's `--check`
  flag returns non-zero on drift (handy for CI / a pre-commit hook).

## Roadmap (not in this task)

1. CAD/measured values replace every `TODO` in `provisional_geometry` and
   `dynamics.links.*` (and real servo `effort` / PD gains).
2. Real inter-axis offsets replace the coincident-hip/ankle approximation.
   (`generate_mjcf.py` already emits per-joint `<joint pos>` anchors.)
3. Floating / vertical-slide pelvis → single-leg **stance** test with the foot
   carrying real body weight (this phase's rig is fixed-base).
4. Right leg by mirroring; attach both legs to a waist/torso chain → full
   20-DoF Cara.
5. Only then: locomotion-policy training.
