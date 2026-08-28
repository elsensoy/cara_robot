# cara_description

Parameterised description of Cara's **lower body** — pelvis + both legs
(12 DoF). Validated kinematics, a provisional dynamics layer
(mass / COM / inertia / actuator + PD gains), and MuJoCo models verified to
(a) reproduce the pure-Python FK exactly and (b) **stand statically under
joint PD control**. Foundation for balance and locomotion work; it is **not**
a full robot, not CAD, not a policy, and has no upper body yet.

## What this package is (and isn't)

| | |
|---|---|
| ✅ | 12-DoF pelvis + both legs; right leg = mirror of left (not hand-written) |
| ✅ | two parameterised configs: `left_leg.yaml` (SSOT for one leg) and `cara_lower_body.yaml` (`extends` + mirror + floating base) |
| ✅ | generated, inspectable URDF **and** MJCF (kinematic + dynamic) for each |
| ✅ | MuJoCo verified to reproduce the pure-Python FK to machine precision |
| ✅ | dynamic MJCF: gravity + PD servos + foot–ground contact |
| ✅ | **static standing** — holds 3 poses 10 s each, COM inside the support polygon, low torque |
| ✅ | validation + FK + single-leg-plausibility + standing scripts, and COM / gravity-torque / Jacobian / sweep analysis |
| 🔶 | dynamics — mass / COM / inertia / actuator limits / PD gains are **provisional placeholders** |
| ❌ | head, ears, arms, waist |
| ❌ | CAD geometry, servo brackets, wiring, shells |
| ❌ | dynamic balance, weight shifting, RL / locomotion policy |

Design constraint being followed: **kinematics, dynamics, and manufacturing
are kept separate.** The YAML is layered accordingly. No servo is selected;
every dynamic number is labelled `TODO` / `TBD`. The head/battery/Jetson
masses are added *after* the lower body is stable, one at a time.

## Layout

```
cara_description/
├── config/
│   ├── left_leg.yaml            # SSOT for ONE leg + pelvis (fixed base)
│   └── cara_lower_body.yaml     # extends left_leg + mirror l_→r_ + floating pelvis
├── urdf/                        # GENERATED — cara_left_leg.urdf, cara_lower_body.urdf
├── mjcf/                        # GENERATED — <model>.xml (kinematic) + <model>_dynamic.xml
├── scripts/
│   ├── leg_model.py             # shared loader (extends/mirror) + pure-Python kinematics & dynamics
│   ├── generate_urdf.py         # YAML -> URDF
│   ├── generate_mjcf.py         # YAML -> MJCF  (--dynamic for the physics model)
│   ├── validate_description.py  # structural checks (kinematics + dynamics + base/mirror)
│   ├── fk_sanity_check.py       # single-leg forward-kinematics behaviour checks
│   ├── validate_mjcf.py         # MuJoCo body/site positions vs leg_model FK  (--dynamic)
│   ├── dynamic_check.py         # single leg: gravity + PD plausibility over scripted poses
│   ├── stand_check.py           # lower body: hold standing poses 10 s + COM/support-polygon
│   ├── view_mujoco.py           # load a generated MJCF and open mujoco.viewer
│   ├── center_of_mass.py        # whole-model COM for any joint configuration
│   ├── gravity_torques.py       # gravitational joint torques for reference poses
│   ├── jacobian.py              # foot-position Jacobian + finite-difference validation
│   └── morphology_sweep.py      # effect of a parameter sweep on workspace / COM / torque
├── docs/
│   ├── frames_and_joints.md     # frame conventions + per-joint math + foot frame hierarchy
│   ├── dynamics_notes.md        # provisional dynamics layer + single-leg analysis
│   └── standing_notes.md        # mirroring the 2nd leg + the standing milestone
└── README.md
```

One SSOT per model, every robot description generated — never edit URDF or MJCF by hand:

```
config/left_leg.yaml ──────┐        ┌──> urdf/<model>.urdf
                           ├─ each ─┼──> mjcf/<model>.xml           (kinematic)
config/cara_lower_body.yaml ┘        └──> mjcf/<model>_dynamic.xml   (gravity + PD + contact)
```

`cara_lower_body.yaml` is ~30 lines: `extends: left_leg.yaml`, `mirror: {source: l_, target: r_}`,
a floating `base`, a wider pelvis, and bilateral standing poses. The right leg
is reflected through the sagittal plane at load time (`leg_model._apply_mirror`).

**Dependencies:** **PyYAML** (`pip install pyyaml`) for everything; **mujoco**
(`pip install mujoco`, brings numpy) only for `validate_mjcf.py`,
`dynamic_check.py`, `stand_check.py`, `view_mujoco.py`. The pure-Python
kinematics/dynamics core needs no numpy.

## Kinematic tree

```
pelvis  (root; physical)
├─ l_hip_yaw  Rz → l_hip_yaw_link  ─ l_hip_roll Rx → l_hip_roll_link ─ l_hip_pitch Ry → l_thigh
│    └─ l_knee_pitch Ry → l_shin ─ l_ankle_pitch Ry → l_ankle_link ─ l_ankle_roll Rx → l_foot
│         • l_foot_sole_center   (ground-contact frame)
└─ r_hip_yaw  … (mirror of the left leg through the sagittal plane)
     … r_foot ─ • r_foot_sole_center
```

`l_hip_yaw_link`, `l_hip_roll_link`, `l_ankle_link` (and their `r_` twins) are
**virtual coupling links**: the 3 hip axes / 2 ankle axes are modelled as
intersecting at a point, so these links have zero-length joint origins and
**never** carry mass, inertia or collision geometry. Real servo-stack offsets
are a `TODO`.

Per-leg joints (the `r_` leg is the mirror — same limits, `+angle` means the
same physical motion on both sides):

| Joint (`l_`/`r_`) | Axis (`l_`) | Limits (rad) | + angle means | Purpose |
|-------|------|--------------|---------------|---------|
| `hip_yaw`     | `+Z` | −0.79 … 0.79 | toe turns outward (toe-out) | leg heading, turning |
| `hip_roll`    | `+X` | −0.52 … 0.61 | thigh abducts (away from midline) | lateral balance, stance width |
| `hip_pitch`   | `+Y` | −1.75 … 1.05 | thigh swings **back** (extension) | stride length |
| `knee_pitch`  | `+Y` | 0.0 … 2.36 | shin folds back (flexion) | ground clearance, crouch |
| `ankle_pitch` | `+Y` | −0.87 … 0.61 | toe drops (plantarflexion) | push-off, CoP fore/aft |
| `ankle_roll`  | `+X` | −0.44 … 0.44 | sole tilts outer-edge-up (inversion) | CoP lateral, uneven ground |

**Frame convention:** right-handed, `+X` forward, `+Y` left, `+Z` up, origin
at the pelvis. Zero pose = legs straight down, soles flat, toes forward. All
pitch joints share the `+Y` axis for consistency; the physical meaning of
"positive" is documented per joint rather than hidden by flipping axes.
Full details: [`docs/frames_and_joints.md`](docs/frames_and_joints.md).

All limits, lengths, masses, and inertias are **provisional placeholders**.
Kinematic geometry is isolated in `provisional_geometry:`; mass / COM /
inertia in `dynamics.links.*`. Neither requires touching the tree structure.

## Usage

```bash
cd cara_description
LB=config/cara_lower_body.yaml

# --- lower body: the standing milestone --------------------------------
python3 scripts/validate_description.py $LB                       # structural checks
python3 scripts/generate_urdf.py $LB && python3 scripts/generate_urdf.py $LB --check
python3 scripts/generate_mjcf.py $LB           && python3 scripts/generate_mjcf.py $LB --check
python3 scripts/generate_mjcf.py --dynamic $LB && python3 scripts/generate_mjcf.py --dynamic $LB --check
python3 scripts/validate_mjcf.py --dynamic $LB                   # MuJoCo poses vs FK  (needs mujoco)
python3 scripts/stand_check.py                                   # HOLD 3 poses 10 s each  (needs mujoco)
python3 scripts/stand_check.py --verbose --hold 15
python3 scripts/view_mujoco.py --dynamic --config $LB --regen --pose semi_squat

# --- single leg: kinematics + dynamics foundation ---------------------
python3 scripts/validate_description.py
python3 scripts/generate_urdf.py && python3 scripts/generate_mjcf.py && python3 scripts/generate_mjcf.py --dynamic
python3 scripts/fk_sanity_check.py            # FK behaviour + foot-position table
python3 scripts/validate_mjcf.py              # MuJoCo vs pure-Python FK  (needs mujoco)
python3 scripts/dynamic_check.py              # gravity + PD over scripted poses  (needs mujoco)

# --- analysis (fixed-base, provisional dynamics layer) ---------------
python3 scripts/center_of_mass.py [$LB]                 # COM per reference pose
python3 scripts/gravity_torques.py --carry-fraction 0.5 # hold torques + foot load
python3 scripts/jacobian.py                             # foot Jacobian + validation
python3 scripts/morphology_sweep.py --param dynamics.links.l_thigh.mass --values 0.10,0.15,0.20
```

Every script takes an optional config path as its first positional argument.
Generators name their output from the model (`meta.name`), so a non-default
config never clobbers another model's files.

## Lower body & standing

`cara_lower_body.yaml` mirrors the left leg to a right leg through the
sagittal plane — position `(x,y,z)→(x,−y,z)`, axis `(aₓ,a_y,a_z)→(−aₓ,a_y,−a_z)`,
joint limits unchanged. The axis rule keeps `+angle` meaning the same physical
thing on both legs, so a bilateral pose is written once (`"*_hip_roll": 0.15`).

The dynamic MJCF puts the pelvis on a **floating base** (`<freejoint>`) — 6
unactuated DoF. The 12 leg joints have PD `<position>` servos; standing is only
stable if the commanded posture keeps the COM over the support polygon.

`stand_check.py` holds each of `stand_nominal`, `semi_squat`, `stand_wide` for
10 s and checks: upright & near rest height, no drift / residual velocity, COM
inside the convex hull of the foot contacts (with margin), no servo
saturation, feet planted, and MuJoCo-vs-FK agreement. **Current result —
milestone met:** tilt ≤ 0.3°, 0 mm drift, COM margin 33–43 mm, peak torque
≤ 0.4 N·m (limit ±3 N·m), FK error < 1e-15 m. Details in
[`docs/standing_notes.md`](docs/standing_notes.md).

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

`generate_mjcf.py <config>` writes `mjcf/<model>.xml` (and, with `--dynamic`,
`mjcf/<model>_dynamic.xml`) from the same YAML. Design choices (documented in
the file header and the script docstring):

- **Coincident abstraction, no fake mass.** MuJoCo requires positive mass on
  any body with a DOF, so the virtual coupling links are *not* bodies. The
  coincident joints are stacked on the physical body downstream —
  `hip_yaw/roll/pitch` as three `<joint>` on `thigh`, both ankle joints on
  `foot` (per leg). Mathematically identical to the URDF chain; the generator
  emits per-joint `<joint pos>` anchors so it stays correct if real inter-axis
  offsets are added later.
- **Two modes from one generator.**
  - *kinematic* (default → `<model>.xml`): `gravity="0 0 0"`, geoms
    non-colliding, no actuators — the FK-reference / pose-inspection model.
  - *dynamic* (`--dynamic` → `<model>_dynamic.xml`): gravity on, a PD
    `<position>` actuator per joint (`dynamics.actuators.control` gains,
    `forcerange = ±effort`), feet colliding with a ground plane
    (`analysis.ground`), a `<keyframe>` per reference pose. The base is
    **welded** (fixed) unless the YAML sets `base: {type: floating}` — the
    lower body uses a floating (`<freejoint>`) pelvis so it can actually stand.
- **Geometry from the YAML only.** Primitive box/cylinder geoms with sizes
  straight from `dynamics.links.*` — no new numbers.
- **Load programmatically.** `view_mujoco.py` builds the model with
  `mujoco.MjModel.from_xml_path(...)` so the whole pipeline is reproducible —
  don't use the viewer's *File > Open* as the normal workflow.

- **`validate_mjcf.py [--dynamic] [config]`** — compiles the model and checks
  every physical body's position + orientation and every sole site against
  `leg_model.forward_kinematics` for all reference poses (pelvis-relative, so
  it works for a floating base). Agreement **< 1e-15 m**.
- **`dynamic_check.py`** (single leg) — commands the PD servos to each scripted
  pose, settles, reports tracking error, jitter, peak torque + saturation,
  `|τ − τ_gravity|` vs the analytic layer, contact gap / force, FK error.
  Airborne hold torques match `gravity_torques.py` to 4+ decimals.
- **`stand_check.py`** (lower body) — the standing milestone: holds each
  standing pose 10 s and checks upright / no-drift / COM-in-support-polygon /
  no-saturation / feet-planted / FK. See `docs/standing_notes.md`.

## Editing the model

- Change parameters **only** in the YAML (`left_leg.yaml`, or
  `cara_lower_body.yaml` for pelvis-mass / standing poses).
- Re-run the generators for each config you touched, then
  `validate_description.py`, then the checks.
- Never hand-edit the generated URDF / MJCF files — each generator's `--check`
  flag returns non-zero on drift (handy for CI / a pre-commit hook).

## Roadmap

Done: 1-leg kinematics → 1-leg dynamics → 1-leg PD tests → 2 legs + pelvis →
**static standing** → COM / support-polygon checks.

Next:

1. **Weight shifting** — command the COM laterally toward one foot and back,
   hold at each side; then single-support (lift one foot).
2. **Balance** — an ankle/hip strategy on top of joint PD; disturbance recovery.
3. **Upper-body masses, deliberately** — add head → ears → waist → arms one at
   a time; re-run `stand_check.py` and record the change in COM height, tilt
   and hold torque. Then a `cara_full.yaml` (`extends` the lower body).
4. CAD/measured values replace every `TODO` (geometry, mass, inertia, servo
   `effort`, PD gains); real inter-axis offsets replace the coincident
   approximation (both generators already emit `<joint pos>` anchors).
5. Only then: locomotion, then a learned policy.
