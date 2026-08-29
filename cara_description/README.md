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
| ✅ | composed configs: `left_leg` → `cara_lower_body` (`extends` + mirror + float) → `cara_full_body` (`+ include` upper body) |
| ✅ | generated, inspectable URDF **and** MJCF (kinematic + dynamic) for each model |
| ✅ | MuJoCo verified to reproduce the pure-Python FK to machine precision |
| ✅ | **static standing** + **quasi-static weight shifting** — both milestones met (lower body & full body) |
| ✅ | **Phases U1–U6** — welded torso + head/neck + Jetson/battery placement study + symmetric passive arms + ears/ear-servos (`I ~ m r²` study), then a full-body regression with a per-subsystem summary table; every metric compared to a frozen lower-body baseline |
| ✅ | **full body (4.43 kg): standing MET, weight-shift MET at ±0.020 m** — morphology-validation phase (U1–U6) closed |
| ✅ | **U7 — controlled single-foot unloading MET** — the other foot reaches 0 N with the COM inside the *stance* foot polygon (+8.5 mm), swing foot only 3.9 mm off the ground |
| ✅ | validation + FK + plausibility + standing + weight-shift + foot-unload scripts, and COM / whole-body-inertia / gravity-torque / Jacobian / sweep analysis |
| 🔶 | dynamics — mass / COM / inertia / actuator limits / PD gains are **provisional placeholders** |
| ❌ | waist joints, articulated neck / shoulders / ears (all present structurally, locked at 0 for now) |
| ❌ | CAD geometry, servo brackets, wiring, shells |
| ❌ | actually lifting a foot (U8), single-support balance (U9), stepping, RL / locomotion |

Design constraint being followed: **kinematics, dynamics, and manufacturing
are kept separate.** The YAML is layered accordingly. No servo is selected;
every dynamic number is labelled `TODO` / `TBD`. Upper-body subsystems (torso →
head → electronics → arms → ears) are added *one at a time*, each measured
against the frozen lower-body baseline before the next.

## Layout

```
cara_description/
├── config/
│   ├── left_leg.yaml            # SSOT for ONE leg + pelvis (fixed base)
│   ├── cara_lower_body.yaml     # extends left_leg + mirror l_→r_ + floating pelvis + poses
│   ├── cara_upper_body.yaml     # FRAGMENT (not standalone): torso (U1) + head/neck (U2) + electronics (U3) + arms (U4) + ears (U5)
│   └── cara_full_body.yaml      # extends cara_lower_body + include cara_upper_body
├── baselines/                   # frozen lower-body results — the regression comparison target
├── urdf/                        # GENERATED — <model>.urdf
├── mjcf/                        # GENERATED — <model>.xml (kinematic) + <model>_dynamic.xml
├── scripts/
│   ├── leg_model.py             # shared loader (extends/include/mirror) + pure-Python kinematics & dynamics
│   ├── generate_urdf.py         # YAML -> URDF
│   ├── generate_mjcf.py         # YAML -> MJCF  (--dynamic for the physics model)
│   ├── validate_description.py  # structural checks (kinematics + dynamics + base/mirror)
│   ├── fk_sanity_check.py       # single-leg forward-kinematics behaviour checks
│   ├── validate_mjcf.py         # MuJoCo body/site positions vs leg_model FK  (--dynamic)
│   ├── dynamic_check.py         # single leg: gravity + PD plausibility over scripted poses
│   ├── stand_check.py           # lower/full body: hold standing poses 10 s + COM/polygon  (--baseline)
│   ├── weight_shift.py          # lower/full body: quasi-static lateral COM shift via task-space IK  (--baseline)
│   ├── unload_foot.py           # U7: shift + unweight one foot to ~0 N with the COM over the stance foot  (--view)
│   ├── view_mujoco.py           # load a generated MJCF and open mujoco.viewer
│   ├── center_of_mass.py        # whole-model COM for any joint configuration
│   ├── gravity_torques.py       # gravitational joint torques for reference poses
│   ├── jacobian.py              # foot-position Jacobian + finite-difference validation
│   ├── morphology_sweep.py      # pure-Python: param sweep → workspace / COM / analytic torque
│   ├── subsystem_sweep.py       # MuJoCo: param sweep → standing COM / tilt / torque + weight-shift limit
│   ├── placement_study.py       # U3: compare electronics.layouts (Jetson/battery mount points)
│   ├── ear_inertia_study.py     # U5: I ~ m r² -- ear mass/offset vs head inertia about the neck axis
│   └── subsystem_summary.py     # U6: build the full body one subsystem at a time -> summary table
├── docs/
│   ├── frames_and_joints.md     # frame conventions + per-joint math + foot frame hierarchy
│   ├── dynamics_notes.md        # provisional dynamics layer + single-leg analysis
│   ├── standing_notes.md        # mirroring the 2nd leg + the standing milestone
│   ├── weight_shift_notes.md    # task-space IK + the weight-shift milestone
│   ├── single_support_notes.md  # U7→U9: unloading a foot → lifting → single-support balance
│   ├── upper_body_notes.md      # config hierarchy + staged upper-body mass/inertia analysis (U1–U6)
│   └── subsystem_summary.md     # GENERATED (subsystem_summary.py) — the U6 per-subsystem table
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
python3 scripts/weight_shift.py                                  # quasi-static lateral COM shift + sweep  (needs mujoco)
python3 scripts/weight_shift.py --amplitude 0.04 --csv shift.csv --verbose
python3 scripts/weight_shift.py --view                           # watch the shift loop (green dot = target COM, orange = measured)
python3 scripts/weight_shift.py --view --amplitude 0.05          # watch it topple at the limit
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

### Weight shifting

`weight_shift.py` moves a smooth lateral **COM target** centre → foot → centre →
other foot → centre, holding at each. A transparent **frontal-plane IK**
(`leg_model.leg_ik`, restricted to `hip_roll` + `ankle_roll` vs. foot-y +
foot-roll) turns the desired COM into joint targets — no hard-coded hip-roll
trajectory, no gain tuning, feet never lift. It logs desired/measured COM, COM
margin vs the full and each individual foot polygon, pelvis roll/pitch,
left/right vertical contact force, foot slip, `q`/`qdot`/torque and saturation,
then sweeps the target magnitude to find the limit and prints failure cases.
**Milestone met:** at ±0.03 m the load shifts 14 N / 6 N between feet with both
planted, pelvis level (0.35°), < 15 % torque; the quasi-static double-support
limit is ~0.04 m (the opposite foot fully unloads beyond that and she topples).
Details in [`docs/weight_shift_notes.md`](docs/weight_shift_notes.md).

## Upper body (staged mass analysis)

`cara_full_body.yaml` = `cara_lower_body.yaml` **`+ include cara_upper_body.yaml`**.
The upper body is added as a *design-analysis tool* — each subsystem is added
one at a time and its effect on COM / inertia / standing / weight-shifting /
torque is measured against a **frozen lower-body baseline**:

```bash
python3 scripts/stand_check.py  config/cara_full_body.yaml --baseline baselines/lower_body_standing.json
python3 scripts/weight_shift.py config/cara_full_body.yaml --baseline baselines/lower_body_weightshift.json
```

**Phase U1 — rigid torso lump** (welded, 1.20 kg): whole-body mass 2.06 → 3.26 kg,
COM rises ~67 mm; standing peak torque ~doubled (< 30 % of limit), weight-shift
limit **0.04 → 0.03 m**.

**Phase U2 — head + neck lump** (head 0.35 kg on a `solid_sphere`; neck yaw/roll/
pitch joints present but `locked: true`): whole-body mass → 3.61 kg, COM now
**+22 mm *above* the pelvis** (−72 → +22). `subsystem_sweep.py` runs a
head-mass sweep in MuJoCo:

| head mass | COM height | worst knee torque | shift limit |
|---|---|---|---|
| 0.20 kg | 0.284 m | 0.89 N·m | 0.030 m |
| 0.35 kg | 0.296 m | 0.97 N·m | 0.030 m |
| 0.60 kg | 0.312 m | **1.08 N·m** | 0.030 m |

COM height and knee-servo demand rise linearly with head mass; head *height*
(`upper_body.neck.length`) matters ~3× less.

**Phase U3 — Jetson + battery** (0.15 + 0.25 kg lumped masses on a *switchable*
mount point). `placement_study.py` compares the named `electronics.layouts`:

| layout (jetson / battery) | COM vs pelvis | knee τ | shift limit |
|---|---|---|---|
| `both_pelvis_low` | +17.3 mm | 1.13 N·m | 0.030 m |
| `both_torso_mid` | +31.2 mm | 1.15 N·m | 0.030 m |
| `both_high` | +36.7 mm | 1.16 N·m | **0.020 m** |

Read-off: keep the 0.40 kg **low in the pelvis** — COM ~20 mm lower, shift
envelope preserved; "everything high" costs a third of the envelope. It
*reports*, it does not choose.

**Phase U4 — passive arm masses** (one 0.18 kg lump per side, welded at the
shoulder in a neutral hang; `l_arm` authored, `r_arm` mirror-generated —
identical by construction, whole-body COM y = 0). No shoulder joint, no swing.
`subsystem_sweep.py` sweeps `upper_body.arm.mass` (both arms together):

| arm mass /side | COM vs pelvis | Ixx (roll) | Izz (yaw) | worst knee τ | shift limit |
|---|---|---|---|---|---|
| 0.00 kg | +17.3 mm | 0.0781 | 0.0108 | 1.13 N·m | 0.030 m |
| 0.18 kg | +20.8 mm | 0.0827 | 0.0138 | 1.28 N·m | 0.020 m |
| 0.30 kg | +22.8 mm | 0.0858 | 0.0158 | 1.38 N·m | 0.020 m |

The arms hang near pelvis height, so +0.6 kg lifts the COM only **+5.5 mm** —
their effect is on the **inertia tensor**: roll +0.0076, yaw +0.0050 kg·m²
(roughly linear in mass). This must be understood before the shoulder is
articulated — arm swing will then modulate exactly that roll/yaw inertia.
`leg_model.whole_body_inertia(spec, q, about="com")` computes the parallel-axis
tensor; `center_of_mass.py` prints it.

**Phase U5 — ears + head asymmetry study** (0.02 kg plush ear + 0.01 kg
ear-twitch servo per side, welded to the head; `l_ear_joint` is the structural
1-DoF ear, `locked` at 0). `ear_inertia_study.py` measures the **head-subsystem
inertia about the neck axis** (`leg_model.whole_body_inertia(..., about="head",
links=<subset>)`) as ear mass and lateral offset vary:

| | Ixx (roll) | Izz (yaw) about the neck axis |
|---|---|---|
| head only | 0.001379 | 0.000539 kg·m² |
| head + ears (nominal) | 0.001668 (+21 %) | 0.000689 (**+28 %**) |

The ears are 17 % of the head mass but add **28 % of the yaw inertia** the
neck-yaw servo must accelerate, and that grows with the **square** of the
lateral ear offset (measured ΔIzz matches the `2·m·(y²−y₀²)` point-mass
prediction exactly). Whole-body standing tilt and the weight-shift envelope
**do not move** — ears are a head/neck-servo concern, not a balance concern.

Full upper body **with arms + ears**: **2.06 → 4.43 kg**, COM +24 mm above the
pelvis, standing solid (all 3 poses PASS), weight-shift envelope **~0.020 m**
(lower body 0.040 → U2 0.030 → U3 0.025 → U4 0.020 → U5 0.020 — each subsystem
tightens the *dynamic* margin; standing is unaffected).

**Phase U6 — full-body regression + per-subsystem summary.** No new hardware.
`subsystem_summary.py` builds the full body up **one subsystem at a time**
(by pruning the composed spec) and measures the same MuJoCo metrics at each
stage; the pruned "lower body" stage reproduces `cara_lower_body.yaml` to
0.0 g. The saved table ([`docs/subsystem_summary.md`](docs/subsystem_summary.md))
makes each addition explicit — COM height is dominated by the **torso** (+67 mm)
and **head** (+28 mm); whole-body **yaw inertia** by the **arms** (+0.0030,
widest masses); the weight-shift envelope is cut only by the **torso**
(0.040 → 0.030 m) and **arms** (0.030 → 0.020 m). Full-body milestones vs the
frozen baseline: **standing MET** (tilt ≤ 1.6°, peak τ 1.23 N·m of ±3);
**weight shift MET at ±0.020 m** (8/8 checks, roll 0.36°, slip 1.5 mm). This
closes the morphology-validation phase (U1–U6).
Details in [`docs/upper_body_notes.md`](docs/upper_body_notes.md).

`type: fixed` / `locked: true` joints (the torso weld, the neck joints, the two
electronics mounts, the two shoulders, the two ears + two ear-servo mounts)
carry 0 DOF and no servo. The single-leg and lower-body regression outputs stay
**byte-identical** through every U-phase.

**Phase U7 — controlled single-foot unloading** (the first balance/control
phase). `unload_foot.py` runs two transparent quasi-static phases on the full
body: (1) the `weight_shift` frontal-plane IK shifts the COM toward the stance
foot, then (2) the swing leg is shortened in the sagittal plane a fraction of a
mm at a time until its `Fz` crosses 5 % of body weight, then frozen. **Milestone
MET** at a COM target of 0.030 m, both feet: the unloaded foot reaches **0 N**
with the swing foot only **3.9 mm** off the ground (not a deliberate lift), the
whole-body COM **inside the stance foot's own polygon with +8.5 mm margin**,
pelvis tilt 3.8°, no saturation. The valid window is narrow (~0.030 m only) —
pre-single-support sits right at the edge of the ±0.020 m double-support
envelope, which is what "about to enter single support" means. `baselines/
full_body_unload.json` freezes the result. Details in
[`docs/single_support_notes.md`](docs/single_support_notes.md).

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
- **`weight_shift.py`** (lower body) — the weight-shift milestone: task-space
  IK drives a lateral COM trajectory; logs force transfer / margins / slip /
  torque; sweeps the amplitude to the double-support limit. See
  `docs/weight_shift_notes.md`.

## Editing the model

- Change parameters **only** in the YAML (`left_leg.yaml`, or
  `cara_lower_body.yaml` for pelvis-mass / standing poses).
- Re-run the generators for each config you touched, then
  `validate_description.py`, then the checks.
- Never hand-edit the generated URDF / MJCF files — each generator's `--check`
  flag returns non-zero on drift (handy for CI / a pre-commit hook).

## Roadmap

Done: 1-leg kin/dyn/PD → 2 legs + pelvis → **static standing** →
COM/support-polygon → **quasi-static weight shifting** → **U1–U6 upper body
(morphology validation complete)**.

Morphology / design validation (each measured vs the frozen baseline):

- **U1 torso lump** ✅
- **U2 head + neck lump** ✅ (neck joints structural but `locked` at 0)
- **U3 Jetson + battery placement study** ✅ (`placement_study.py` over `electronics.layouts`)
- **U4 passive arm masses** ✅ (symmetric, welded shoulders, no articulation; effect is on the inertia tensor)
- **U5 ears + head asymmetry study** ✅ (`ear_inertia_study.py`: `I ~ m r²` about the neck axis; ear joints `locked`)
- **U6 full-body regression + per-subsystem summary** ✅ (`subsystem_summary.py`; standing MET, weight-shift MET at ±0.020 m)

Balance / control (the boundary — new controllers start here):

- **U7 unload one foot toward `Fz → 0`** ✅ (`unload_foot.py`; the other foot hits
  0 N with the whole-body COM inside the stance foot polygon, +8.5 mm margin,
  swing foot 3.9 mm off the ground — a valid pre-single-support state, both sides)
- **U8** lift the unloaded foot 5–10 mm, hold, return
- **U9+** single-support balance → stepping → locomotion → learned policy

CAD/measured values replace every `TODO` before single-support locomotion.
