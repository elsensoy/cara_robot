# Cara — Quasi-static bilateral weight shifting

Companion to [`standing_notes.md`](standing_notes.md). The step from *standing
still* to *deliberately moving the centre of mass between the two feet*, still
in double support (neither foot lifts), still no RL.

Roadmap position:

```
… → static standing ✅ → COM / support-polygon ✅ →
WEIGHT SHIFTING ✅ (this doc) → single-support → balance → locomotion → RL
```

Script: `scripts/weight_shift.py` (headless logs + sweep; `--view` opens the
MuJoCo viewer and loops the shift with a green dot at the desired COM and an
orange dot at the measured COM). Milestone question:

> **Can Cara deliberately transfer her weight from one leg to the other while
> remaining in controlled double support?**

---

## 1. The control stack (all transparent, no tuning)

```
com_y_desired(t)         smooth lateral COM target: centre → +foot → centre → −foot → centre
      │  1-D table inversion  (com_world_y is monotone in pelvis shift)
      ▼
pelvis lateral shift     the pelvis displacement that puts the COM where asked
      │  frontal-plane IK, per leg:
      │    free joints  = { <side>hip_roll , <side>ankle_roll }
      │    constraints  = foot y-position + foot roll  →  sole stays flat & planted
      │    (sagittal joints hip_pitch / knee / ankle_pitch stay at the base pose;
      │     hip_yaw stays 0)
      ▼
q_target (12 joints)  ──►  the EXISTING PD <position> servos  ──►  MuJoCo
```

- **No hard-coded hip-roll trajectory.** `leg_model.leg_ik` is a damped
  least-squares Newton loop over the already-validated `forward_kinematics` and
  a 6×6 spatial Jacobian. For the shift it is restricted to a 2×2 problem
  (hip_roll + ankle_roll vs. foot-y + foot-roll); residual < 1e-8 m.
- **The COM target is inverted, not approximated.** Shifting the pelvis by
  `Δy` moves the world COM by *less* than `Δy` — the swing-side leg abducts and
  its mass swings the other way. `weight_shift.py` builds a table of
  `pelvis_shift → predicted world COM y` from the IK + `center_of_mass`, then
  looks up the pelvis shift that achieves `com_y_desired`. (For `com_y = 0.03 m`
  the pelvis actually shifts ~0.038 m.)
- **No gains are tuned.** The `dynamics.actuators.control` PD gains from the
  standing model are used unchanged.
- **Quasi-static** = 3 s smoothstep ramps between targets, 3 s dwells.

Config: `analysis.weight_shift` in `config/cara_lower_body.yaml`
(`base_pose`, `amplitude`, `sweep`, ramp/hold/settle seconds, and the sweep
`accept` thresholds). All provisional.

---

## 2. What is logged (per simulation step, and to `--csv`)

| quantity | source |
|---|---|
| desired & measured COM `(x, y, z)` | trajectory input vs `data.subtree_com[0]` |
| COM margin vs the **full** support polygon | convex hull of all foot–floor contact points |
| COM margin vs **each foot's own** polygon | that foot's collision-box bottom-face hull |
| pelvis roll & pitch | free-joint quaternion |
| left / right **vertical contact force** | `mj_contactForce` projected to world z, summed per foot |
| foot **slip** | horizontal drift of each foot collision geom from its post-settle position |
| `q`, `qdot`, **actuator torque** (12 each) | `data.qpos/qvel/actuator_force` |
| **torque saturation** | count of joints at `±forcerange` |

---

## 3. Demonstration run (COM target ± 0.03 m)

Centred baseline: each foot carries **10.1 N** (half the 20.2 N lower-body
weight).

| window | COM y (des / meas) | Fn left | Fn right | m_left | m_right | pelvis roll | peak \|τ\| | sat |
|--------|--------------------|---------|----------|--------|---------|-------------|-----------|-----|
| centred | 0.000 / 0.000 | 10.1 N | 10.1 N | −27 mm | −27 mm | 0.0° | 0.08 N·m | 0 |
| **+A** (toward left) | 0.030 / **0.033** | **14.1 N** | **6.1 N** | **+6 mm** | −33 mm | −0.35° | 0.32 N·m | 0 |
| **−A** (toward right) | −0.030 / −0.032 | **6.1 N** | **14.1 N** | −33 mm | **+6 mm** | +0.35° | 0.32 N·m | 0 |

All eight validation checks pass:

- shifting toward a foot **raises that foot's normal force** (10.1 → 14.1 N)
  and **unloads the other** (10.1 → 6.1 N), both directions;
- **both soles stay planted** — minimum foot force 6.0 N (30 % of body weight),
  4 contact corners each, the whole run;
- **feet do not slip** — max drift 1.3 mm;
- **pelvis stays level** — max tilt 0.35°;
- **COM stays in the support polygon** — 33 mm margin;
- **no actuator saturation** — peak torque 11 % of the ±3 N·m limit.

At the +A hold the work is done almost entirely by `hip_roll` (−0.19 rad,
0.32 N·m) and `ankle_roll` (+0.18 rad, −0.19 N·m) per leg; the sagittal joints
sit at their nominal values carrying < 0.11 N·m.

---

## 4. Sweep — how far can she shift?

`weight_shift.py` sweeps the COM-target magnitude and fails a value if it
breaks **any** acceptance criterion at the hold (margin, opposite-foot load,
pelvis tilt, slip, torque, planted). Failure cases are printed, not hidden.

| A (m) | COM y @+A | ΔFn left | ΔFn right | opp. foot load | m_full | tilt | slip | verdict |
|-------|-----------|----------|-----------|----------------|--------|------|------|---------|
| 0.010 | 0.011 | +1.4 N | −1.4 N | 43 % | 33 mm | 0.2° | 0.2 mm | PASS |
| 0.020 | 0.022 | +2.7 N | −2.7 N | 36 % | 33 mm | 0.2° | 0.6 mm | PASS |
| 0.030 | 0.033 | +4.0 N | −4.0 N | 30 % | 33 mm | 0.3° | 1.3 mm | PASS |
| 0.040 | 0.044 | +5.1 N | −5.1 N | 24 % | 27 mm | 0.7° | 2.4 mm | PASS |
| 0.050 | 0.16 (!) | — | — | **0 %** | — | 4.0° | 172 mm | **FAIL** — opposite foot fully unloads, topples sideways |
| 0.060 | 0.34 (!) | — | — | 0 % | — | 180° | 499 mm | **FAIL** — falls |

**Quasi-static double-support limit ≈ 0.04 m** COM shift (~80 % of the hip
half-width). It is a *cliff*, not a gradual fade: once the target reaches the
loaded foot's centre (~0.045 m) the opposite foot's normal force hits zero,
double support is lost, and with pure joint PD there is nothing to catch the
tip — she falls. Staying in *controlled* double support means keeping the
opposite foot loaded (here ≳ 20 % of body weight).

---

## 5. Milestone verdict

**MET.** Cara can deliberately transfer weight left↔right in controlled double
support, demonstrated at ± 0.03 m COM target (load shifts 14 N / 6 N between
feet, both planted, pelvis level, < 15 % torque), with a quasi-static limit of
~0.04 m before the opposite foot unloads.

Everything is provisional (masses, PD gains, friction) and there is no foot
lifting and no RL.

---

## 6. Open TODOs

- [ ] Replace provisional masses / gains / friction with real values; re-run
      the sweep (the limit will move).
- [ ] The IK assumes the pelvis stays level; log shows ≤ 0.4° so it holds
      here, but a closed-loop pelvis-orientation term would extend the range.
- [ ] Single-support: from the +A shift, unload the light foot to zero and
      lift it — needs an actual balance strategy (ankle/hip), not joint PD.
- [ ] Add head / ears / waist / arms masses one at a time; the COM rises and
      the shift limit changes — measure it.
