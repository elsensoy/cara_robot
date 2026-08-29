# Cara — Toward single support (U7 → U9)

Companion to [`weight_shift_notes.md`](weight_shift_notes.md). This is the first
work **past the morphology boundary** — U1–U6 validated the whole-body mass model
(it stands and weight-shifts); U7 onward is **balance / control** on that model.

```
… static standing ✅ → weight shifting ✅ → morphology U1–U6 ✅  ──┼── boundary
    U7 unload one foot ✅ → U8 lift one foot ✅ → U9 single-support balance ✅ (this doc) → stepping → …
```

Still transparent — the same frontal-plane IK from `weight_shift.py`, plus a
small feedback controller on top of the position PD: a **minimal pelvis-roll
trim** for U7/U8, upgraded to a **COM-feedback balance controller** for U9. No
RL. The model under test is the **complete** Cara — `cara_full_body.yaml`,
4.43 kg.

> **The roll trim.** With both feet planted Cara balances passively, but the
> instant one foot unweights, the single-support roll moment exceeds what the
> standing-tuned position PD holds at one leg — she rolls ~10° and catches the
> free-foot edge. U7 and U8 therefore add the smallest thing that makes the hold
> possible: `trim = SIDE · (kp·roll + kd·roll̇)` added to the stance
> `ankle_roll` target, plus `−SIDE·kh·roll` to the stance `hip_roll`
> (`SIDE = ±1` because the mirror flips those axis signs). Gains live in
> `analysis.unload_foot` / `analysis.lift_foot`, all provisional. A full
> disturbance-rejecting balance controller is **U9** — this is just enough to
> hold still.

---

## Phase U7 — controlled single-foot unloading

Script: `scripts/unload_foot.py` (`--view` loops it). Milestone question:

> **Can Cara reach a physically valid *pre-single-support* configuration?** —
> transfer weight toward one foot until the other reaches ~0 N, *without a
> deliberate lift*, with the whole-body COM inside the **stance foot's own**
> polygon.

### Maneuver (`analysis.unload_foot`)

1. **COM shift** — the `weight_shift` frontal-plane IK (free = {hip_roll,
   ankle_roll} per leg) ramps the lateral COM target toward the stance foot.
2. **Swing-leg unweight** — the swing leg is shortened in the sagittal plane
   ({hip_pitch, knee_pitch, ankle_pitch}, foot held *level* — task = {foot z,
   foot pitch}), raising its foot target until `Fz` crosses 5 % of body weight,
   then **frozen**. The minimal roll trim runs throughout.

Valid pre-single-support = at the freeze point: `Fz` ≤ 5 % weight, reached with
the free foot risen < 5 mm (`accept.not_lifted_rise`), COM inside the stance
polygon with margin, tilt / slip / torque within limits, stance sole planted.

### Result

| COM target | free-foot Fz | rise at crossing | stance margin | tilt | slip | verdict |
|---|---|---|---|---|---|---|
| 0.024–0.030 m | ~2 N (4–5 %) | ~1 mm | +5…+13 mm | ~2–3° | ~16 mm | stance foot slips |
| **0.033 m** | **2.2 N (5.0 %)** | **1.0 mm** | **+13.3 mm** | **0.6°** | **2.7 mm** | **valid** |

**MILESTONE MET** at COM target 0.033 m, both feet: the free foot carries
**2.2 N** with its sole only **1.0 mm** off the ground (genuinely not lifted),
the whole-body COM is **+13.3 mm inside the stance foot polygon**, pelvis tilt
0.6°, stance slip 2.7 mm, torque 39 % of limit. `baselines/full_body_unload.json`
freezes it.

Below 0.033 m she reaches ~0 N but the COM is not yet centred over the stance
foot and the roll trim works harder, walking the stance foot ~16 mm. The valid
window is a **single point** — pre-single-support sits right at the edge.

### Note — an IK fix corrected this phase

The first `unload_foot.py` targeted the swing foot's *centred* position while
the body was shifted, so its sagittal IK ran at a ~30 mm residual (never
converged) and the "unweighting" was partly a numerical artifact. Fixed: the
swing target is now the foot's **actual position in the COM-shifted config**,
task = {z, pitch}, which converges to < 1e-9. With the correct IK the phase also
needs the roll trim (added here and in U8) — without it Cara topples.

---

## Phase U8 — first single-support milestone: lift, hold, return

Script: `scripts/lift_foot.py` (`--view`). Milestone question:

> **Can Cara stand on one foot for ~1.5 s** — free foot a few mm clear, COM
> inside the stance polygon, pelvis near level, stance foot not slipping, no
> actuator saturated — **and put the foot back down cleanly?**

### Maneuver (`analysis.lift_foot`)

A. shift the COM onto the stance foot (COM target 0.028 m) ·
B. + C. raise the free foot with **closed-loop world clearance** control (the
pelvis sags on the swing side as load transfers, so a pelvis-frame command
under-delivers — the loop drives the *measured* world clearance to
`lift_height`) and **hold** `hold_seconds` in single support ·
D. lower, ramp the COM back, settle → double support.
The minimal roll trim (kp 1.6 / 0.8, kd 0.10) runs through B–D.

### Result — `lift_foot.py config/cara_full_body.yaml`

| lift height | free-foot clearance | free-foot Fz | stance Fz | COM margin | tilt | slip (hold) | peak torque | return | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 5 mm | 4.8 mm | 0.0 N | 100 % wt | +10.7 mm | 4.0° | 1.9 mm | 84 % (swing hip_roll) | 4/4 | **PASS** |
| 7 mm | 6.8 mm | 0.0 N | 100 % wt | +10.5 mm | 4.0° | 3.8 mm | 87 % | 4/4 | **PASS** |
| 10 mm | 9.7 mm | 0.0 N | 100 % wt | +10.5 mm | 4.1° | 4.1 mm | 88 % | 4/4 | **PASS** |

**MILESTONE MET**, both feet (identical by symmetry), every lift height 5–10 mm:
Cara stands on one foot with the free foot fully unloaded and the COM **+10.5 mm
inside the stance polygon**, tilt 4°, then returns cleanly to flat double
support. The lower body alone does it even more comfortably (slip 0 mm, torque
36 %, tilt 2°). `baselines/full_body_lift.json` freezes it.

### What's tight

- **The swing `hip_roll` is the torque bottleneck** — ~2.6 N·m of the
  provisional ±3.0, holding the lifted leg out to the side. It does **not**
  saturate, but a heavier or longer-articulated leg would. The stance
  `ankle_roll` stays within its provisional ±2.0 N·m (`--ankle-effort 3.0`
  changes nothing — the ankle isn't the limit).
- **Pelvis tilt ~4°** of the 6° budget — the minimal trim holds her, but not
  level. **U9's COM controller brings this to 2.8°.**
- **Hold is brief (1.5 s) and undisturbed.** Push recovery and the longer hold
  are **U9, below.**
- COM target 0.028 m (U8) vs 0.033 m (U7): each phase's shift is tuned for its
  own dynamics — U7 freezes at ~1 mm rise, U8 lifts to 5–10 mm and holds.

---

## Phase U9 — single-support balance

Script: `scripts/single_support.py` (`--view` opens the viewer and applies a
gentle alternating pulse). U8 got Cara onto one foot for 1.5 s with a
pelvis-roll trim; U9 replaces that with a **COM-feedback balance controller** and
asks two things:

1. can she hold single support *indefinitely* (tested to `hold_seconds` = 5 s)?
2. how big a lateral push can she reject without the free foot touching down?

### The controller (`analysis.single_support.balance`)

A PD on the **whole-body COM-y drift** relative to the stance foot, trimming the
stance `ankle_roll` target, plus a P term on the stance `hip_roll` (ankle + hip
strategy):

```
drift  = (COM_y − stance_foot_y) − (its value at the start of the hold)
ankle_roll_target += SIDE · (kp·drift + kd·COM_y_velocity)     # SIDE = ±1 (mirror flips the axis)
hip_roll_target   += SIDE · kp_hip·drift
```

`kp/kd_ankle = 50/10`, `kp_hip = 15` — hand-set provisional values. The swing
foot stays on the U8 closed-loop clearance. Disturbances are scripted lateral
force pulses on the pelvis, swept in magnitude, both directions.

### Result — `single_support.py config/cara_full_body.yaml`

| | value (both sides, identical) |
|---|---|
| **5 s hold** | COM-y drift **2.3 mm**, pelvis tilt **2.8°**, free foot **7.8 mm** clear, torque 76 % |
| lateral push **toward the swing foot** | recovers **~1.0 N × 100 ms**, falls at 2.0 N |
| lateral push **toward the stance foot** | recovers **~3.0 N × 100 ms**, falls at 4.0 N |

**MILESTONE MET**, both stance sides. The COM feedback holds the COM **5× tighter
than U8's trim** (2.3 mm drift vs 17 mm) and is gentler on the joints (76 % vs
88 %).

### The disturbance envelope is small — and that's a design finding

A 1 N × 100 ms impulse ≈ 0.1 kg·m/s ≈ a **0.02 m/s** lateral COM-velocity kick.
Cara can't do much better with an ankle strategy because her **feet are tiny**:
the roll-axis half-width is 22.5 mm, so the most CoP moment the stance ankle can
make before the foot rolls onto its edge is ≈ `Fz · d` ≈ 43.5 × 0.0225 ≈
**1 N·m**. The envelope is asymmetric (bigger toward the stance foot) because
that direction pushes the COM *deeper* into the support polygon.

To reject a real disturbance Cara needs a **wider foot**, a **stronger hip /
angular-momentum strategy**, or a **protective step** — which is U10. Adding a
hip-*velocity* term here only made it oscillate (tested, reverted).

### Open TODOs

- [ ] Foot size (45 × 22.5 mm half-extents) is the single biggest limit on the
      balance envelope — a design-level input, not a control problem.
- [ ] Balance gains are hand-set provisional values, tuned for the **full body**
      (the milestone target); the lower-body model needs its own.
- [ ] Sagittal (COM-x) balance is not yet controlled — the disturbances tested
      are lateral only.
- [ ] Swing `hip_roll` torque headroom is thin (~2.6 of ±3.0 N·m) — a
      servo-sizing input.
- [ ] U10: a protective / deliberate step, once the balance envelope is
      exceeded.
