# Cara — Single support → stepping → the dynamic-walk model and controller (U7 → U16)

Companion to [`weight_shift_notes.md`](weight_shift_notes.md). This is the first
work **past the morphology boundary** — U1–U6 validated the whole-body mass model
(it stands and weight-shifts); U7 onward is **balance / control** on that model.

```
… static standing ✅ → weight shifting ✅ → morphology U1–U6 ✅  ──┼── boundary
    U7 unload one foot ✅ → U8 lift one foot ✅ → U9 single-support balance ✅
      → U10 one forward step ✅ → U11 a short quasi-static walk ✅
      → U12 continuous *kinematic* walk ❌ (formulation wall) → U13 reduced-order
        model ✅ (a dynamic walk IS feasible) → U14 DCM-tracking controller 🔶
        → U15 torque-controlled ankles ✅ + warm-start 🔶 → U16 gait initiation
        fixed ✅, double-support→single-support handoff is the open piece 🔶
        (this doc) → …
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

---

## Phase U12 — continuous walk: a **documented limit**, not a milestone

Script: `scripts/walk.py` (`--view`). Question:

> **Can Cara walk continuously** (no stop between steps), the COM advancing at a
> roughly steady speed, staying upright — then stop and stand?

### Approach (still no ZMP, no RL)

One step (lead `l_`, `r_` stance) is precomputed as a dense joint trajectory:
the **sagittal** joints track each foot's world path (foot forward `stride`,
pelvis forward `stride`); the four **roll** joints track a lateral COM sway from
`weight_shift`'s inverted table. The step is translation-periodic (`p=1` equals
the `l↔r` swap of `p=0`), so it's played back on a loop, mirroring on alternate
steps. A **gated** roll trim (U9's, faded in only over the single-support swing —
U9's `kp 50` destabilises a *moving* reference, and a `/dt` D-term on a moving
signal blows up, so the D-term is off) rides on top.

### Result — **MILESTONE NOT MET**

| cadence | what happens |
|---|---|
| **fast** (`t_step` ≲ 4 s) | completes the single-support swing, then **topples at the double-support transfer** to the next stance foot — ~0.8 steps in (`baselines/full_body_walk.json`) |
| **slow** (`t_step` ≈ 8 s) | **stays upright** (tilt < 6°, ends standing) but **barely advances** (~0 mm/step of a 20 mm stride) |

Two distinct walls, both structural:

1. **Lateral capture (fast).** The half-sine COM sway gives the COM real lateral
   velocity toward the stance foot. To hand off to the *other* foot the COM must
   cross the midline and be arrested — but the 22.5 mm foot can make only
   ≈ 1 N·m of ankle-roll CoP moment (the U9 `balance_margin` wall, ~6.5 mm
   single-support margin). The momentum carries her past the new foot and she
   topples. This is the same limit U9 measured; continuous walking is the first
   phase that actually spends it.
2. **Forward drive (slow).** Kinematic pose-playback induces **no net forward
   translation** without push-off or momentum — commanding the stance foot
   "backward in the pelvis frame" makes the *foot slide back*, not the pelvis
   advance. U11 got around this by ramping to a world-anchored staggered pose
   each step and stopping; a continuous gait can't.

### What a continuous walk needs

- a **dynamic gait controller** — a ZMP or capture-point pattern generator that
  plans the CoP trajectory and includes ankle **push-off**, not just kinematic
  playback; and/or
- a **wider foot** (the recurring U9/U10/U11 finding — `balance_margin.py`'s
  sweep: 22.5 → 45 mm roughly quadruples the lateral tolerance).

**Quasi-static stepping (U11) remains Cara's locomotion** until one of those
lands. `walk.py` stays in the tree as the generator + the honest failure
characterisation (it prints which wall it hit and ties it to the U9 numbers).

---

## Phase U13 — the reduced-order walking model (LIPM / DCM)

Script: `scripts/walk_model.py` (**pure Python** — no sim needed; `--check`
adds a MuJoCo cross-check). U12 replayed joint poses and hit walls; U13 stops
replaying and starts **predicting where the COM must go** for a dynamically
valid step, with the linear inverted pendulum + capture point:

```
x'' = ω₀² (x − p),   ω₀ = √(g / z_com),   ξ = x + ẋ/ω₀   (the DCM / capture point)
```

Closed form over a step (`p` constant): `x(T) = p + (x₀−p)cosh ω₀T + (ẋ₀/ω₀)sinh ω₀T`;
the DCM diverges as `ξ(T) = p + (ξ₀−p)e^{ω₀T}`. **The DCM is what the next
foot has to catch** — exactly the hand-off that failed in U12.

### Cara's LIPM parameters (from the model + `provisional_geometry`)

| | value |
|---|---|
| total mass | 4.43 kg |
| COM height `z_com` | 298 mm |
| **ω₀ = √(g/z)** | **5.74 rad/s** (τ = 174 ms) — MuJoCo cross-check: released-lean divergence fits **5.1 /s**, consistent |
| foot half-length `a_x` (fore/aft CoP range) | 45 mm |
| foot half-width `a_y` (lateral CoP range) | 22.5 mm |
| foot lateral offset `s` (½ stance width) | 50 mm |

### Lateral limit cycle — the side-to-side rock

With the CoP at the stance-foot centre, the symmetric periodic solution has the
COM **crossing the midline at every L↔R hand-off** with the DCM at
`ξ_y = s·tanh(ω₀T/2)`. For the next foot to catch it: `ξ_y ≥ s − a_y`, i.e.

> **`T ≥ T_min = (2/ω₀)·atanh(1 − a_y/s) ≈ 0.22 s`.**

| step time `T` | sway peak | DCM_y at hand-off | inner margin | feasible |
|---|---|---|---|---|
| 0.20 s | 7 mm | 26 mm | −1.6 mm | **no** (topple inward) |
| 0.25 s | 11 mm | 31 mm | +3.3 mm | yes |
| 0.30 s | 14 mm | 35 mm | +7.3 mm | yes |
| 0.40 s | 21 mm | 41 mm | +13 mm | yes |
| 1.00 s | 44 mm | 50 mm | +22 mm | yes (→ one-foot stance, i.e. U11) |

**A dynamic lateral rock is feasible for Cara for step times ≳ 0.22 s.** The
22.5 mm foot is *enough* — U12's slow runs (`t_step` 3.5–8 s ≫ T_min) did stay
upright; U12's fast run toppled because the hand-picked sway + static-hold trim
**did not follow the pendulum**, not because the morphology forbids it.

### Forward motion

Roomier — the fore/aft CoP range is `a_x` = 45 mm (2× the lateral). At 50 mm/s
the DCM lands 8–20 mm ahead of the new foothold for `T` = 0.3–0.7 s (CoP margin
25–37 mm); it only runs out past `T` ≈ 1.4 s (75 mm steps). U12's slow run had
no forward speed because kinematic playback carries no momentum — the planner
has to command the forward lean (DCM offset `v/ω₀` ≈ 9 mm) and place each foot
*ahead of* the DCM.

### Feasibility map — foot half-width × step time

```
     a_y \ T   0.12  0.15  0.18  0.22  0.28  0.35  0.50        T_min
    15.0 mm     .     .     .     .     .     Y     Y           302 ms
    22.5 mm     .     .     .     Y     Y     Y     Y   ← Cara   216 ms
    30.0 mm     .     Y     Y     Y     Y     Y     Y           148 ms
    40.0 mm     Y     Y     Y     Y     Y     Y     Y            71 ms
```

Widening the foot lowers `T_min` (more time before the DCM must reach the foot),
but Cara **already lands in the feasible region** at sensible step times.

### Finding

> **A dynamically-consistent walk is within Cara's current morphology.** The U12
> wall was the *kinematic formulation* (replay a fixed pose cycle + a
> static-hold trim), not the hardware. **U14** is a DCM-tracking walk: plan the
> CoP trajectory and footholds from this model (`walk_model.py` gives `T_min`,
> the sway, the foothold offsets), then drive the legs to track the planned DCM
> — no fixed pose cycle, U9's roll term repurposed as a DCM feedback law.
> `baselines/full_body_walk_model.json` freezes the numbers.

### Caveats

- Point-mass LIPM — no trunk/leg angular momentum, no knee-height variation, no
  finite-time double support. It's a *planning* model; U14's controller closes
  the gap to the full dynamics (the ω₀ cross-check already shows ~12 % model
  error from leg compliance).
- CoP fixed at the foot centre in the limit-cycle solve — the real ±`a_y` CoP
  range is margin against disturbance, not yet spent here.

---

## Phase U14 — the DCM-tracking controller  🔶 **built, blocked on ankle torque control**

Script: `scripts/dcm_walk.py` (`--view`). The controller U13 called for:

1. **Plan** (offline, from `walk_model.LIPM`): a footstep sequence, the CoP held
   at each foot centre through single support, and the DCM reference by backward
   recursion `ξ_ini[i] = p[i] + (ξ_ini[i+1] − p[i])·e^{−ω₀T}`.
2. **Feed-forward** (per step): the swing leg tracks a trajectory to the planned
   foothold (`gait.py`'s per-step IK table); the roll joints track the LIPM
   **COM-y arc** rolled forward from the *measured* state each step (a
   hand-picked half-sine was U12's mistake).
3. **DCM feedback** (per sim step): `ξ_meas = COM + COṀ/ω₀`;
   `p_cmd = p_ref + (1 + k/ω₀)(ξ_meas − ξ_ref)`, clamped into the stance foot,
   realised as stance `ankle_roll` / `ankle_pitch` trims.
4. **Step adjustment** (per footfall): shift the next foothold to null the DCM
   error predicted at end of step — the capture point *is* a foothold target.

### Result — the planner is right, realisation is blocked

`dcm_walk.py` **does not produce a stable walk** on the position-PD model — she
topples in step 0 (`baselines/full_body_dcm_walk.json`). Two concrete blockers,
both about *realising* the plan rather than the plan itself:

1. **From-rest start.** The plan's first DCM sits at `ξ_ini[0]_y ≈ −41 mm`, well
   past the **~20 mm double-support weight-shift envelope**. A LIPM walk is
   entered *with lateral momentum* (the limit cycle crosses the midline moving
   toward the stance foot); from a standing rest Cara has none, and a lead-in
   that shifts the COM out that far topples her before step 1.
2. **CoP authority.** Placing the LIPM CoP is an **ankle-torque** action. The
   position servos (`kp` 30) move the CoP only ≈ `1.5·d` per radian of trim and
   lag a 0.4 s step — the U9/U12 realisation problem, now precisely located.

### The gap, precisely

The **theory (U13) and the controller structure (U14) are in place**; what's
missing is:

- **torque-controlled ankles** (a `<motor>` actuator on `ankle_roll` /
  `ankle_pitch` instead of the position `<position>` servo) so the CoP command
  is realised directly, and
- a **limit-cycle warm-start** — a few rocking half-steps that build the lateral
  momentum before the forward walk begins (or a ZMP-preview controller that
  plans the CoP over a horizon rather than step-by-step).

Both are follow-on work. `dcm_walk.py` stays in the tree as the planner +
feedback law + step-adjustment implementation, and prints exactly which blocker
it hit. **Quasi-static stepping (U11) remains Cara's locomotion.**

---

## Phase U15 — torque-controlled ankles ✅ + a warm-start 🔶

U14 said a dynamic walk needs the CoP to be an **ankle-torque** action, not a
position target. U15 delivers that.

### Torque-controlled ankles — done and validated

`dynamics.actuators.torque_joints: [...]` (set at runtime by `dcm_walk.py`)
makes `generate_mjcf` emit a direct-torque **`<motor>`** actuator for the listed
joints instead of the PD `<position>` servo, plus a little passive
`<joint damping>` (`torque_joint_damping`, default 0.06) so a software attitude
PD on top stays stable under the stiff ground contact. `dcm_walk.py` flags the
four ankle joints; `data.ctrl[ankle]` is then a torque in N·m, and it applies

```
τ_ankle = kp_att·(θ_des − θ) − kd_att·θ̇   +   Fz·(p_cmd − p_ankle)
          └────── keep the foot behaving ──────┘   └── place the CoP ──┘
```

- **Default MJCF is byte-identical** — no config opts in, so `<position>`
  everywhere and the hard gate holds.
- **Standing verified** with the torque ankles + software PD (tilt 0.4°).

So the U14 realisation blocker is removed: the CoP command *is* now realisable.

### Warm-start — the remaining piece

`dcm_walk.py` prefixes the walk with `warmup_steps` rocking half-steps (both feet
planted, CoP oscillating with growing amplitude) to build the lateral limit
cycle before any forward progress.

**It doesn't yet produce a walk** — she stands, does ~1 rocking half-step, then
topples (`baselines/full_body_dcm_walk.json`). The failure is now purely **gait
initiation**: from rest at the midline the inverted pendulum diverges *away* from
the intended stance foot (`ẍ = ω₀²(x − p)` with `x` on the far side of `p`), and
the steady-state lateral displacement (~40 mm) is past the ~20 mm double-support
envelope, so a simple "rock toward the stance foot" doesn't converge to the
limit cycle.

### What's left (U16)

- A proper **CoP-leads-motion gait-initiation** sequence — the CoP placed to
  *accelerate* the COM toward the first stance foot, then caught — rather than
  the current fixed-amplitude rock.
- **DCM-controller tuning** on the (now correct) torque-ankle model, or a
  **ZMP-preview / MPC** formulation that plans the CoP over a horizon instead of
  step-by-step.

None of this is RL or a hardware change — it's controller work on a model that
now has the right actuation. **Quasi-static stepping (U11) stays Cara's
locomotion meanwhile.**

---

## Phase U16 — gait initiation fixed, a narrower blocker remains 🔶

U15 left one open question: why does the warm-up rock topple on its very first
half-step? Two independent bugs turned out to be hiding in it, both specific to
*double support* (both feet planted, no lift) rather than the single-support
mechanics U9–U15 already validated.

### Bug 1 — the CoP torque went to the wrong (unloaded) foot

`dcm_walk.py`'s warm-up alternates a "stance" label by step index (`stance =
OTHER[lead]`, `lead` flipping every half-step) and realised the CoP as torque
on *only* that foot's ankle. But both feet stay planted the whole warm-up, and
as the rock's amplitude grows, weight transfers fully onto one foot *before*
its half-step officially ends — measured with `DCM_DBG` instrumentation,
`Fz` on the labelled "stance" foot was already 0 N several tenths of a step
before the code stopped trying to control it. Realising a CoP through a foot
carrying no weight is realising nothing: zero authority exactly when the
pendulum most needs correcting.

**Fix:** during warm-up, drive *each* ankle from its *own* measured contact
force and its *own* local CoP clamp (`min/max` around that foot's own sole,
not a single shared target):

```python
for fp_ in ("l_", "r_"):
    Fz_ = foot_normal_force(foot_gid[fp_])          # that foot's own load
    py_l = clamp(p_cmd[1], sf_y[fp_] ± a_y)          # that foot's own reach
    cop_tau[fp_+"ankle_roll"] = cop_gain * (-SIDE[fp_]) * Fz_ * (py_l - sf_y[fp_])
```

Whichever foot is actually loaded does the work; an unloaded foot's Fz gates
its own contribution to ~0 automatically, so nothing needs to know in advance
which foot that will be.

### Bug 2 — an exponentially ill-conditioned excitation

The DCM equation is exponential in `ω₀·T`. The warm-up used the *same*
step duration as a real forward step (`t_step` ≈ 0.5 s ≈ 2.9 time constants at
ω₀ ≈ 5.7 rad/s) to hold a small "CoP-leads-the-COM" nudge. Solving the DCM
equation backward for the *exact* CoP that would land on-target from true rest
showed the required precision: **~1 mm**, versus the ~15 mm nudge the
heuristic amplitude schedule was actually commanding — any real discrepancy
that size gets amplified `e^(ω₀T) ≈ 18×` by the end of the 0.5 s hold. That is
the "diverges away from the stance foot" failure mode U15 documented: not a
sign error, an **ill-conditioned control problem** at that duration.

**Fix:** give the warm-up its own, much shorter step duration
(`warmup_t_step`, 0.12 s here — chosen so `e^(ω₀T) ≈ 2.4×`, keeping the
feedback law and the foot-sized CoP clamp inside their well-behaved linear
regime) and cap the last rock's peak CoP excursion below the full foot
half-width (`warmup_amp_hi = 0.035 m` of `w_hip_half = 0.05 m`) so it doesn't
run out of margin exactly when the amplitude — and therefore the momentum
carried into the handoff — is largest.

### Result

With both fixes, the warm-up rock is now **robust regardless of length** — it
was re-run at 4, 6, 8, and 10 rocks and never fell during the rock itself, DCM
tracking error staying single-digit-mm early, growing to ~40 mm by the largest
rock (comfortably inside the 25 mm acceptance band for most of it). It then
carries into the **first real forward step** before falling — vs. U15, which
fell on the *first warm-up half-step* with a 411 mm peak error.

```
                          steps survived   peak |DCM error|
U15 (as documented)       1 / 14                411 mm   (fell in warm-up step 1)
U16 (this phase)          7 / 14                143 mm   (fell in the first forward step)
```

**MILESTONE NOT MET** — the walk still doesn't complete — but the failure is
now precisely localised to one place: the **double-support → single-support
handoff**, i.e. the instant the first foot genuinely leaves the ground (every
warm-up "step" keeps both feet planted; `warmup_lift: 0`). The per-step DCM
error trace makes the jump explicit:

```
per-step DCM error: 4, 4, 11, 12, 30, 38, 135 mm   (| = warm-up/forward boundary after 6)
```

six clean rocks, then an order-of-magnitude jump the instant real single
support begins. A parameter sweep (`warmup_t_step`, `warmup_amp_lo`,
`warmup_amp_hi`, `cop_torque_gain`, `k_dcm`, `ankle_kp`, `step_adjust_gain`,
`warmup_steps` 4–10) never broke this pattern — every configuration tried
survives the *entire* warm-up (however many rocks) and always fails at
exactly the same place, the first real liftoff. That consistency is itself
informative: this isn't a warm-up tuning problem any more, it's a distinct,
narrower problem at the handoff.

### What's left (U17)

- Treat the double-support → single-support handoff as its own sub-phase —
  e.g. widen (or slow) the double-support fraction specifically for the first
  real step, so the first liftoff doesn't coincide with the rock's peak
  momentum.
- Or replace the step-by-step DCM feedback with a **ZMP-preview / MPC**
  formulation that plans the CoP over a receding horizon spanning the
  handoff, rather than reacting to it one step at a time.

Still not RL, not a hardware change. **Quasi-static stepping (U11) stays
Cara's locomotion meanwhile.**
