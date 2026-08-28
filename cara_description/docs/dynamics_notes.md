# Cara — Dynamics-analysis layer (pelvis + left leg)

Companion to [`frames_and_joints.md`](frames_and_joints.md). Same scope: the
pelvis/base link and one left leg. This document covers the **provisional
dynamics** layer that was added on top of the validated kinematics, and the
four analysis scripts that use it.

> **Status:** every mass, COM and inertia here is a **provisional
> placeholder** (`config/left_leg.yaml → dynamics`). No servo has been
> selected. The goal of this layer is to make *structure* and *method*
> explicit and to let you ask *"how do geometry and mass distribution affect
> COM, foot motion and required torque?"* — not to publish final numbers.

---

## 1. What lives where

| Layer | YAML block | Trust |
|-------|-----------|-------|
| kinematics | `provisional_geometry`, `links`, `joints` | structure trusted; lengths provisional |
| **dynamics** | `dynamics.links` (physical links only), `dynamics.actuators` | **all provisional / TBD** |
| analysis inputs | `analysis` (gravity, reference poses, rough total mass) | not robot properties — script inputs |

### Physical vs virtual links

Only the four **physical** links carry mass/COM/inertia:

```
pelvis, l_thigh, l_shin, l_foot
```

The three **virtual coupling links** (`l_hip_yaw_link`, `l_hip_roll_link`,
`l_ankle_link`) are massless mathematical abstractions — the points where the
3 hip axes / 2 ankle axes are modelled as intersecting. In the YAML they are
just `{is_physical: false}`. They **must never** acquire:

- mass, COM, or inertia,
- collision geometry,
- visual geometry.

`generate_urdf.py` emits them as bare `<link name="…"/>` frames.
`validate_description.py` fails if any of them gains `mass`/`com`/`inertia`.

**MJCF / MuJoCo note (for when a `generate_mjcf.py` is added):** MuJoCo
requires every `<body>` to have inertia. The correct representation there is
*not* to give the virtual links a fake epsilon inertia — it is to **not make
them bodies at all**: attach `l_hip_yaw`, `l_hip_roll`, `l_hip_pitch` as three
`<joint>` elements on the single `l_thigh` body (and both ankle joints on the
`l_foot` body). That removes the massless-body problem and is exactly what the
"coincident axes" approximation means physically.

---

## 2. Provisional inertia — how each tensor is approximated

Each physical link records an `inertia.method` so it is obvious what is a
guess. Tensors are diagonal, about the link COM, aligned with the link frame.

| Link | method | formula |
|------|--------|---------|
| `pelvis` | `solid_box` | box `[0.06, 0.10, 0.06]` m, uniform density |
| `l_thigh` | `uniform_rod_z` | `Ixx = Iyy = m L²/12`,  `Izz = m r²/2`  (`L = L_thigh`, `r = 0.020`) |
| `l_shin` | `uniform_rod_z` | same, `L = L_shin`, `r = 0.015` |
| `l_foot` | `solid_box` | box `[foot_len, foot_width, h_ankle]` |

COM positions are expressions over `provisional_geometry`, so they follow a
geometry sweep automatically:

```
pelvis   com = [0, 0, 0]
l_thigh  com = [0, 0, -L_thigh/2]      # assumed mid-segment
l_shin   com = [0, 0, -L_shin/2]
l_foot   com = [foot_x_off, 0, -h_ankle/2]   # centroid of the foot box
```

The slender-rod transverse term `I ≈ m L²/12` is the quantity you sanity-check
by hand before replacing it with a CAD inertia tensor.

---

## 3. The explicit foot frame hierarchy, and the +0.015 m sole offset

Three stacked frames, proximal → distal:

```
(1) ankle joint frame           origin of l_ankle_link
        │                       (= l_ankle_roll joint frame; the ankle-pitch
        │                        and ankle-roll axes intersect here)
        │  l_ankle_roll.origin = [0, 0, 0]        ← coincident-axis approx, TODO
        ▼
(2) foot body frame             l_foot link frame
        │                       (coincides with the ankle joint frame at q = 0)
        │  offset = [foot_x_off, 0, -h_ankle] = [+0.015, 0, -0.035] m
        ▼
(3) sole / contact frame        l_foot_sole_center   (frames_of_interest)
                                centre of the ground-contact patch
```

### Where the +0.015 m X offset comes from

It is **one parameter**: `provisional_geometry.foot_x_off = 0.015`. It appears
only in the sole frame's `xyz_expr: [foot_x_off, 0, -h_ankle]`. Nothing else in
the model produces an X offset at the zero pose (all joint origins are on the
Z axis or zero).

### Is it intentional?

**The sign/structure is intentional; the magnitude is a placeholder.**

- *Intentional:* `foot_x_off > 0` places the contact centre **forward of the
  ankle-roll axis**, i.e. the ankle sits over the **rear third of the foot**
  (heel:toe ≈ 1:2 with the current `foot_len`). This is the correct anatomy
  for a foot that points forward, and it is why the leg can push off.
- *Placeholder:* the value `0.015 m` is a guess. `# TODO` set it from CAD /
  measurement of the real foot.

It is **not** an error and should not be "corrected" to zero. A zero offset
would put the ankle at the centre of a symmetric foot, which is wrong.

### Its one visible consequence right now

The foot COM inherits the same `foot_x_off` X component, so at the zero pose
gravity on the foot mass produces a small but real torque about the three
pitch axes: `m_foot · g · foot_x_off ≈ 0.05 · 9.81 · 0.015 ≈ 0.0074 N·m`.
`gravity_torques.py` reports exactly this at the `zero` pose — a good
demonstration that the offset matters.

---

## 4. Centre of mass — `center_of_mass.py`

$$
r_{COM}(q) \;=\; \frac{\sum_i m_i\, r_i(q)}{\sum_i m_i}
$$

over the physical links, with `r_i(q)` the world COM of link *i* from forward
kinematics. Optional extra point masses (`--extra KG@frame`) are added to both
the numerator and denominator — use them to test *"battery near the pelvis vs
battery out at the foot"* or a carried load.

```bash
python3 scripts/center_of_mass.py                     # table over all reference poses
python3 scripts/center_of_mass.py --pose deep_crouch  # per-link breakdown
python3 scripts/center_of_mass.py --extra 0.4@l_foot_sole_center
```

With the current placeholders the leg+pelvis COM sits ~47 mm below the pelvis
origin at the zero pose and rises to ~31 mm in a deep crouch (folding the leg
brings mass back up toward the pelvis).

---

## 5. Gravitational joint torques — `gravity_torques.py`

**Base-fixed model: the pelvis is ground.** For a pose *q*:

$$
\tau_j(q) \;=\; \frac{\partial U}{\partial q_j}
\;=\; \sum_{i\ \text{distal to}\ j} m_i\, g \,
\big[\, \hat a_j \times (r_i - o_j) \,\big]_z
\qquad U = \sum_i m_i\, g\, z_i
$$

with `â_j`, `o_j` the world axis and origin of joint *j*.

**This is the torque to hold the *dangling* leg segments in pose *q*.** It
does **not** include supporting Cara's body weight. For the stance-leg case,
add body weight as a point load at the foot:

```bash
python3 scripts/gravity_torques.py --carry-fraction 0.5   # 50% of the (provisional) 3 kg
```

which internally uses `τ = Jᵀ F` with `F = (0, 0, m_carry·g)` and is
cross-checked against `∂U/∂q` (they agree exactly). Every run also
cross-checks the analytic torque against a central-difference of `U(q)` and
prints the classic single-link pendulum term `m g r sin θ` for the thigh.

Order of magnitude with the current placeholders:

| pose | peak hold torque (leg only) | with 1.5 kg at the foot |
|------|-----------------------------|--------------------------|
| zero | ~0.007 N·m (hip/knee/ankle pitch) | — |
| half_crouch | ~0.08 N·m at hip_pitch | ~0.6 N·m at knee_pitch |
| deep_crouch | ~0.14 N·m at hip_pitch | — |

The knee dominates once body weight is carried in a crouch — the expected
result, and the reason knee-servo torque is the number to pin down first.

---

## 6. Foot Jacobian — `jacobian.py`

$$
\dot x_{foot} = J(q)\,\dot q, \qquad
J(q) = \frac{\partial x_{foot}}{\partial q} \in \mathbb{R}^{3\times 6},\qquad
J_{:,j} = \hat a_j \times (p_{foot} - o_j)
$$

The script computes `J` two ways — geometric (above) and central-difference of
FK — and validates:

1. `|J_geometric − J_numeric|` (≈ 1e-11 with the current step),
2. `J·q̇` against a central-difference of FK along a random `q̇` (the
   `ẋ_foot ≈ J(q) q̇` check).

It also prints the manipulability `w = √det(J Jᵀ)`: lowest at the zero pose
(straight leg, near the reach singularity), higher in a crouch.

This same `J` is what later connects a ground reaction force `F` at the foot
to joint torques via `τ = Jᵀ F` (`leg_model.joint_torques_from_foot_force`).

---

## 7. Morphology sweeps — `morphology_sweep.py`

Overrides **one dotted path** in the spec (in memory only) and recomputes the
analysis quantities:

```bash
python3 scripts/morphology_sweep.py --param provisional_geometry.L_thigh --values 0.10,0.12,0.14
python3 scripts/morphology_sweep.py --param dynamics.links.l_thigh.mass  --values 0.10,0.15,0.20
python3 scripts/morphology_sweep.py --param dynamics.links.pelvis.mass   --values 0.6,1.1,1.6
```

Per value it reports total mass, COM depth (zero & crouch), max reach, foot
workspace X/Z span over a `hip_pitch × knee` grid, deepest sole Z, and peak
`|gravity hold torque|` per joint over all reference poses.

Findings with the current placeholders:

- **thigh length 10 → 14 cm:** COM drops ~11 mm deeper at the zero pose, max
  reach +4 cm, foot X workspace +7.5 cm, peak hip_pitch hold torque +0.07 N·m
  — and total mass is **unchanged**, because thigh mass is currently an
  independent parameter. Sweep `dynamics.links.l_thigh.mass` separately to see
  the mass effect (that is a deliberate separation, not an oversight).
- **pelvis mass (battery placement proxy):** raising pelvis mass moves the
  whole-model COM toward the pelvis and barely changes joint torques (the
  pelvis is the fixed base — nothing distal to any joint). Moving the battery
  *down the leg* (e.g. onto `l_thigh`) is what raises hold torque; use
  `--param dynamics.links.l_thigh.mass` and/or edit its `com` to explore that.

---

## 8. Open TODOs

- [ ] Replace every `dynamics.links.*` value with CAD / measured mass, COM and
      a full inertia tensor (drop the `method` approximations).
- [ ] Decide whether segment mass should scale with a swept length (currently
      independent) — add a density model if so.
- [ ] Real actuator `effort` / `velocity` once servos are chosen (`dynamics.
      actuators` is all TBD).
- [ ] Inverted / stance-leg model (pelvis free, foot on ground) for true
      body-weight torque — the `--carry-fraction` point load is only a proxy.
- [ ] `analysis.provisional_total_robot_mass` is a rough guess; refine once the
      other 14 DoF exist.
