# Cara — Toward single support, the first step, and a short walk (U7 → U11)

Companion to [`weight_shift_notes.md`](weight_shift_notes.md). This is the first
work **past the morphology boundary** — U1–U6 validated the whole-body mass model
(it stands and weight-shifts); U7 onward is **balance / control** on that model.

```
… static standing ✅ → weight shifting ✅ → morphology U1–U6 ✅  ──┼── boundary
    U7 unload one foot ✅ → U8 lift one foot ✅ → U9 single-support balance ✅
      → U10 one forward step ✅ → U11 a short walk ✅ (this doc) → dynamic gait / RL → …
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

### The disturbance envelope is small — and *why*  (`scripts/balance_margin.py`)

`balance_margin.py` gets into the same held single-support state and measures the
mechanism instead of asserting it. For the full body, stance `r_`:

| measured in the held pose | value |
|---|---|
| stance Fz | 43.5 N (100 % of body weight on one foot) |
| COM height above the sole | 295 mm → inverted-pendulum ω = 5.8 rad/s (τ ≈ 173 ms) |
| foot sole half-width (roll axis) | **22.5 mm** |
| stance foot centre, off the body midline | −45.6 mm (toward stance) |
| COM / CoP, off the **foot centre** | **+16.0 mm toward the swing side** |

The COM only shifts ~30 mm off the midline (`com_target` 28 mm, and the lifted
leg hangs inboard and pulls it back), but the stance foot centre is ~46 mm out —
so the CoP sits **16 mm toward the inner edge of a 22.5 mm half-width foot**,
with only **6.5 mm of lateral room left toward the swing foot** (38.5 mm toward
stance).

- **Static budget.** Max restoring moment = `Fz · half-width` ≈ 43.5 × 0.0225 ≈
  **0.98 N·m**; **71 %** of it is already spent just holding the pose. Past the
  edge the foot rolls onto its rim and no gain helps.
- **Capture-point estimate.** `J_max ≈ margin · m · ω` → ~1.7 N × 100 ms toward
  swing (first-order, ~1.5–2× high because it ignores the ankle pulling back
  during the pulse), which the fine validation sweep confirms at **~1.0 N**.
- **Asymmetry.** Toward stance she has ~38 mm of sole but still falls at ~3 N —
  there the failure is the **recovery overshoot** swinging her back past the
  inner edge and dropping the lifted foot, a controller limit rather than the
  CoP wall. Either way the **6.5 mm swing-side gap is the binding constraint.**
- **Foot half-width sensitivity** (analytic, first-order): 22.5 → 30 mm roughly
  doubles the swing-side tolerance (~1.7 → ~3.6 N); 45 mm ≈ 4×.

So the envelope is set by **foot geometry and where the swing leg parks the COM**,
not by the gains. To reject a real disturbance Cara needs a **wider/longer
foot**, the **lifted foot tucked toward the midline** (re-centres the CoP), an
**arm / trunk angular-momentum strategy**, or a **protective step** — which is
U10. Adding a hip-*velocity* term here only made it oscillate (tested, reverted).

### Open TODOs

- [ ] Foot size (45 × 22.5 mm half-extents) is the single biggest limit on the
      balance envelope — `balance_margin.py` quantifies it: only 6.5 mm of
      lateral CoP room toward the swing foot. A design-level input, not a
      control problem.
- [ ] The lifted-foot posture pulls the COM ~16 mm toward the inner edge —
      tucking it inboard (or a larger `com_target` now that the foot is clear)
      would re-centre the CoP. Worth a trajectory experiment.
- [ ] Balance gains are hand-set provisional values, tuned for the **full body**
      (the milestone target); the lower-body model needs its own.
- [ ] Sagittal (COM-x) balance is not yet controlled — the disturbances tested
      are lateral only.
- [ ] Swing `hip_roll` torque headroom is thin (~2.6 of ±3.0 N·m) — a
      servo-sizing input.

---

## Phase U10 — one deliberate forward step

Script: `scripts/step_once.py` (`--view` loops it). Milestone question:

> **Can Cara take one full step** — shift onto one foot, lift the other, swing it
> forward to a new foothold, place it, and transfer the weight — **and end in a
> stable staggered stance with the pelvis advanced**, both legs leading?

This is the answer to the U9 finding (past ~1 N toward the swing side she *has*
to move a foot) and the first building block of a gait (U11).

### Maneuver (`analysis.step`) — six quasi-static phases

| | phase | how |
|---|---|---|
| A | shift the COM onto the stance foot | `weight_shift` frontal IK table, `com_target` 0.028 m |
| B | lift the swing foot to `lift_height` (10 mm) | closed-loop world clearance (as U8/U9) |
| C | **swing it forward** to the foothold | swing-leg IK table over the step progress `s ∈ [0,1]` (foot held level, {hip,knee,ankle}_pitch) |
| D | place it | lower the clearance to the ground |
| E | **transfer** | ramp both legs to the final staggered pose — lead foot forward `step_len`, **pelvis advanced `step_len/2`** — bringing the COM into the new, larger polygon |
| F | hold the new stance `hold_seconds` | check stable |

Only U9's **lateral (COM-y) roll trim** runs during B–D (same gains, 50/10/15).
Sagittal (COM-x) feedback is deliberately left out — the quasi-static trajectory
keeps COM-x safe, and a COM-x → `ankle_pitch` term fought the swing at every gain
tried (the foot is 90 mm long, far more fore/aft CoP room than the roll trim
needs). Ramps are **4 s** (3 s was too fast — the lift + trim went unstable).

### Result — `step_once.py config/cara_full_body.yaml`

| step | lead | placed | place err | COM margin (swing) | pelvis tilt (swing) | stance slip | peak τ | COM advance | final tilt | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 mm | l_ / r_ | ✅ | 5.4 mm | +5.8 mm | 2.4° | 4.2 mm | 76 % | **10.7 mm** | 0.4° | **PASS** |
| 30 mm | l_ / r_ | ✅ | 5.9 mm | +5.8 mm | 2.3° | 4.1 mm | 76 % | **15.4 mm** | 0.4° | **PASS** |
| 40 mm | l_ / r_ | ✅ | 6.6 mm | +5.8 mm | 2.2° | 4.1 mm | 76 % | **20.2 mm** | 0.3° | **PASS** |

**MILESTONE MET**, both legs leading, every step length 20–40 mm: the swing foot
lands within **7 mm** of the target foothold, the COM stays **inside the support
polygon** throughout (the +5.8 mm swing-side margin during the single-support
phase is the U9 CoP limit again), the pelvis stays under **2.5°**, the stance
foot slips **< 4.2 mm**, no actuator saturates (swing `hip_roll` at 76 % is the
worst, as in U8), and she settles into the new staggered stance **level (< 0.5°)
with ~35 mm of COM margin**. The COM advances **~half the step**, as designed.
`baselines/full_body_step.json` freezes it.

### What's tight / deferred

- **Forward only.** A sideways / widening step is past Cara's lateral balance
  envelope with these provisional feet (the U9 CoP limit) — `step_once.py` does
  not attempt one.
- **One step, then stop.** Chaining steps into a gait (bring the trailing foot
  through, alternate) is U11.
- **Lower body fails it** — the roll-trim gains are full-body-tuned (as in U9);
  the lower-body model's stance foot slides. Reported, not hidden.
- **Sagittal COM-x feedback** is still unbuilt — the step gets away with a
  quasi-static COM-x trajectory; a faster or disturbed step will need it.
- Step length is capped at 40 mm by the swing IK (the knee reaches its extension
  limit reaching further at constant foot height).

---

## Phase U11 — a short walk

Script: `scripts/gait.py` (`--view` loops it). Milestone question:

> **Can Cara take N consecutive quasi-static steps** (alternating legs),
> advancing steadily, and end standing — COM inside the support polygon every
> step, pelvis near level, feet not slipping, no actuator saturated?

**Nothing new in the controller.** Each step is U10's six phases plus U9's
lateral roll trim. The only new machinery in `gait.py`:

1. **Start each step from the staggered stance the last one left**, not always
   from `stand_nominal`. The lateral COM shift (phase A) only moves the roll
   joints, which are decoupled from the sagittal stagger — so the *same*
   `weight_shift` roll deltas are overlaid on whatever staggered sagittal pose
   she's in.
2. **Alternate the lead foot** (`l_, r_, l_, r_, …`), each new foothold one
   `stride` ahead of the current stance foot.
3. Phase E puts the pelvis at the **midpoint of the two feet**, which advances it.

Step 1 from rest lands directly in the canonical staggered stance, and every
step after that is the same cycle mirrored — so the gait is genuinely periodic.

### Result — `gait.py config/cara_full_body.yaml` (4 steps, `stride` 24 mm)

| step | lead | COM advance | foot placed within | COM margin (swing) | pelvis tilt | stance slip | peak τ | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | l_ | 30.6 mm | 5.4 mm | +5.6 mm | 2.2° | 4.3 mm | 76 % | **PASS** |
| 2 | r_ | 26.6 mm | 5.8 mm | +4.5 mm | 2.6° | 3.9 mm | 76 % | **PASS** |
| 3 | l_ | 26.4 mm | 5.8 mm | +4.6 mm | 2.3° | 3.8 mm | 76 % | **PASS** |
| 4 | r_ | 26.3 mm | 5.9 mm | +4.6 mm | 2.3° | 3.8 mm | 76 % | **PASS** |

**MILESTONE MET**: Cara walks **4 steps forward (110 mm total)**, alternating
legs, and holds the final stance for 3 s **level (0.9°) with 0.1 mm of COM
drift**. Steps 2–4 are within **0.3 mm** of each other — the gait has settled to
a **periodic cycle** (verified out to 6 steps). Every step keeps the same
+4.5 mm swing-side COM margin (the U9 CoP limit, once more), 76 % peak torque
(swing `hip_roll`), and < 4.3 mm stance slip.

### What's tight / deferred

- **`stride` ≲ 25 mm.** At 34 mm step 2 topples — a full periodic step's swing
  foot must travel ~2 × `stride` relative to the (stationary) pelvis, and that
  exceeds the swing leg's reach (U10's ~40 mm cap, plus the from-behind part).
- **Quasi-static, not walking gait.** Each step ramps over ~14 s of sim time and
  settles to a full stop between steps — this is *stepping*, not a dynamic walk.
  A real gait (continuous, no stop, using momentum) is the next phase.
- **Forward, straight, flat ground.** No turning, no slopes, no pushes.
- **Lower body fails it** — full-body-tuned roll gains, as in U9/U10. Reported.
- Same open items as U9/U10: foot size, sagittal balance feedback, servo
  headroom, lower-body gains.
