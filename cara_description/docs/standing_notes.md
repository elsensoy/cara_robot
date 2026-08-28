# Cara — Lower body & static standing

Companion to [`frames_and_joints.md`](frames_and_joints.md) and
[`dynamics_notes.md`](dynamics_notes.md). This covers the step from *one leg*
to *pelvis + both legs standing under PD control*.

Roadmap position:

```
1-leg kinematics ✅ → 1-leg dynamics ✅ → 1-leg PD tests ✅ →
2 legs + pelvis ✅ → STATIC STANDING ✅ (this doc) →
COM / support-polygon checks ✅ → weight shifting → balance → locomotion → RL
```

Deliberately **not** added yet: head, ears, arms, waist, Jetson, battery. The
lower-body dynamics are made stable first; those masses go in afterward, one at
a time, measuring the effect on balance and torque demand.

---

## 1. How the second leg is made — `config/cara_lower_body.yaml`

The right leg is **not** hand-written. `cara_lower_body.yaml` is tiny:

```yaml
extends: left_leg.yaml           # left leg + pelvis stay the single source of truth
mirror: {source: "l_", target: "r_"}
base:  {type: floating, rest_pose: stand_nominal}
dynamics: {links: {pelvis: {mass: 1.10, inertia: {box: [0.12, 0.16, 0.07]}}}}
analysis: {reference_poses: {stand_nominal: ..., semi_squat: ..., stand_wide: ...}}
```

`leg_model.load_spec` then:

1. **`extends`** — deep-merges this file on top of `left_leg.yaml`.
2. **`mirror`** — reflects every `l_*` link / joint / dynamics entry / frame
   through the sagittal (x-z) plane to make the `r_*` side:

   | quantity | mirror rule |
   |---|---|
   | position `(x, y, z)` | `(x, −y, z)` |
   | rotation axis `(aₓ, a_y, a_z)` | `(−aₓ, a_y, −a_z)`  (axial-vector reflection) |
   | joint limits | **unchanged** |
   | `+Y`/`left` in doc strings | swapped to `−Y`/`right` |

   The axis rule is what makes `+hip_roll = abduction`, `+hip_yaw = toe-out`,
   `+ankle_roll = inversion` mean the *same physical thing* on both legs — so a
   bilateral pose can be written once (`"*_hip_roll": 0.15`) and limits copy
   across untouched.

Result: 13 links, 12 revolute DoF, pelvis as the single root.

`"*_<suffix>"` keys in `reference_poses` expand to every joint ending in
`<suffix>` (both legs).

---

## 2. The standing rig — `generate_mjcf.py --dynamic config/cara_lower_body.yaml`

`mjcf/cara_lower_body_dynamic.xml`:

- **Floating pelvis** (`<freejoint>`) — 6 unactuated DoF. Standing is only
  stable if the commanded posture keeps the COM over the support polygon;
  there is no cheat force holding the pelvis up.
- **12 PD `<position>` servos**, one per leg joint (`dynamics.actuators.control`
  gains, `forcerange = ±effort`).
- **Both feet** get a box collision geom vs a ground plane at `z = 0`
  (`analysis.ground.friction`). All other geoms stay non-colliding.
- **`<keyframe>` per pose** — `qpos` places the pelvis at a height where the
  feet just reach the floor for that pose, `ctrl` sets the PD target.

Total lower-body mass with the current placeholders: **2.06 kg** (pelvis 1.10,
each leg 0.48).

---

## 3. Static standing — `stand_check.py`

**Milestone: hold three target poses for 10 s each without instability or
unrealistic control effort.**

For each of `stand_nominal`, `semi_squat`, `stand_wide` the script resets to the
keyframe, holds for 10 s, and checks: pelvis stayed upright & near rest height,
no drift / residual velocity, COM stays inside the support polygon (convex hull
of the foot contact points) with margin, no servo saturation, feet planted with
no penetration, and MuJoCo body poses match `forward_kinematics`.

### Result — MILESTONE MET

| pose | tilt | sink | drift | COM margin | peak torque | RMS torque |
|------|------|------|-------|-----------|-------------|-----------|
| `stand_nominal` | 0.16° | 3.5 mm | 0.0 mm | +33 mm | 0.078 N·m (knee) | 0.078 |
| `semi_squat` | 0.30° | 4.1 mm | 0.0 mm | +43 mm | 0.402 N·m (knee) | 0.402 |
| `stand_wide` | 0.16° | 3.8 mm | 0.0 mm | +33 mm | 0.202 N·m (hip_roll) | 0.200 |

- **Upright & still.** Tilt ≤ 0.3°, pelvis horizontal drift 0.0 mm, residual
  joint velocity 0 — the poses are genuinely static, not slowly toppling.
- **COM well inside the feet.** 33–43 mm of margin to the nearest polygon edge;
  `y` component of COM is exactly 0 (symmetry holds).
- **Control effort is tiny.** Peak 0.4 N·m in the deepest squat, ~0.08 N·m
  standing — far below the ±3 N·m provisional limit. No servo saturates.
  (Contrast the fixed-base single-leg `dynamic_check.py`, where a *loaded*
  crouch saturated: here the double-support stance shares the load and the
  knee moment arm is short.)
- **Kinematics reproduced exactly.** MuJoCo vs `forward_kinematics`
  agreement < 1e-15 m in every pose (pelvis-relative, orientation-corrected).

### The three poses

All keep `hip_pitch + knee_pitch + ankle_pitch = 0` (soles flat) and are
left-right symmetric (COM on the mid-sagittal line):

- `stand_nominal` — slight knee bend, feet under the hips.
- `semi_squat` — `hip −0.55 / knee 1.10 / ankle −0.55`; COM ~13 mm forward,
  ~63 mm below the pelvis.
- `stand_wide` — hips abducted 0.15 rad, ankles roll −0.15 to keep the soles
  flat; wider base.

---

## 4. What this does and doesn't prove

Proves: the mirrored 12-DoF lower body is kinematically consistent, and pure
joint-space PD holds statically-stable postures with plausible torque and no
numerical trouble, for 10 s, in three different postures.

Does **not** prove: dynamic balance, disturbance rejection, or that the COM
stays in the polygon during *motion*. Those are the next steps
(weight-shifting → balance).

---

## 5. Open TODOs

- [ ] Replace provisional masses / inertia / PD gains with CAD / measured / tuned values.
- [ ] Weight-shifting: shift COM laterally between the feet and back, hold at each side.
- [ ] Single-support: lift one foot, hold on the other (needs an ankle/hip balance strategy, not just joint PD).
- [ ] Add head → ears → waist → arms masses one at a time; re-run `stand_check` and record the change in COM height, tilt and hold torque.
- [ ] Disturbance test: push the pelvis, measure recovery.
