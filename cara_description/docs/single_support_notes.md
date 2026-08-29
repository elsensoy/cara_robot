# Cara — Toward single support (U7 → U9)

Companion to [`weight_shift_notes.md`](weight_shift_notes.md). This is the first
work **past the morphology boundary** — U1–U6 validated the whole-body mass model
(it stands and weight-shifts); U7 onward is **balance / control** on that model.

```
… static standing ✅ → weight shifting ✅ → morphology U1–U6 ✅  ──┼── boundary
    U7 unload one foot ✅ (this doc) → U8 lift one foot → U9 single-support balance → stepping → …
```

Still quasi-static, still transparent (the same frontal-plane IK from
`weight_shift.py`), still no RL and no gain tuning. The model under test is the
**complete** Cara — `cara_full_body.yaml`, 4.43 kg.

---

## Phase U7 — controlled single-foot unloading

Script: `scripts/unload_foot.py` (`--view` loops the maneuver in the MuJoCo
viewer). Milestone question:

> **Can Cara reach a physically valid *pre-single-support* configuration?** —
> transfer weight toward one foot until the other foot's vertical load reaches
> ~0, *without committing to a lift*, while the whole-body COM sits inside the
> **stance foot's own** support polygon.

### The maneuver (two quasi-static phases, `analysis.unload_foot`)

1. **COM shift** — the `weight_shift.py` frontal-plane IK (free = {hip_roll,
   ankle_roll} per leg, both feet flat + planted) ramps the lateral COM target
   toward the stance foot.
2. **Swing-leg unweight** — the swing leg is then *shortened in the sagittal
   plane only* (a 3-DoF foot-position IK on {hip_pitch, knee_pitch,
   ankle_pitch}, seeded from the shifted config so the frontal balance is
   untouched), raising its foot target a fraction of a millimetre at a time
   until its `Fz` crosses `accept.unloaded_frac_target` (5 % of body weight).
   The clearance is then **frozen** — the foot is not raised any further.

A state is valid pre-single-support when, at that frozen point:

| criterion | `accept` key | value |
|---|---|---|
| unloaded foot `Fz` ≤ 5 % weight | `unloaded_frac_target` | 0.05 (≈ 2.2 N) |
| reached before the swing foot rose > 5 mm (i.e. not a U8-style deliberate lift) | `not_lifted_rise` | 0.005 m |
| whole-body COM inside the **stance** foot polygon with margin | `min_stance_margin` | 0.005 m |
| pelvis tilt | `max_pelvis_tilt_deg` | 6° |
| foot slip over the maneuver | `max_foot_slip` | 0.006 m |
| stance sole keeps ≥ 3 contact corners | `min_stance_corners` | 3 |
| no actuator saturation | `max_torque_frac` | 1.0 |

All seven values are **provisional** (the acceptance thresholds for this phase,
not the tighter double-support ones from `weight_shift`).

### Result — `unload_foot.py config/cara_full_body.yaml`

COM-target sweep, per side (Cara is sagittally symmetric, so l\_ and r\_ rows
are identical — running both is the symmetry check):

| COM target | unloaded `Fz` | swing rise at crossing | stance margin | pelvis tilt | slip | verdict |
|---|---|---|---|---|---|---|
| 0.024 m | 0.5 N (1 %) | 2.4 mm | **−1.0 mm** | 2.5° | 2.4 mm | COM not yet over the stance foot |
| 0.027 m | 1.1 N (2.5 %) | 3.0 mm | **+4.7 mm** | 3.1° | 3.2 mm | margin still < 5 mm |
| **0.030 m** | **0.0 N** | **3.9 mm** | **+8.5 mm** | 3.8° | 3.9 mm | **valid pre-single-support** |
| 0.033 m | — | 4.8 mm | −21.6 mm | topples | 550 mm | falls over |

**MILESTONE MET** at COM target **0.030 m**, on both feet:

- the unloaded foot carries **0 N** — fully unweighted;
- it got there with the swing foot only **3.9 mm** off the ground — an
  incidental rise as load transferred, well below U8's deliberate 5–10 mm lift;
- the whole-body COM is **inside the stance foot's own polygon with +8.5 mm
  margin** — she could now lift the other foot and remain statically balanced;
- pelvis tilt 3.8° (of 6°), stance sole fully planted (4/4 corners), peak
  actuator torque 16 % of the provisional limit.

### What the sweep shows

The valid window is **narrow** (~0.030 m only):

- below it the COM has not travelled far enough to sit inside the small stance
  foot polygon (the polygon half-width is ~30 mm around a foot centre ~50 mm off
  the midline);
- above it Cara is past her controlled double-support envelope (`weight_shift`
  found that limit at ±0.020 m) and topples during the COM ramp.

So pre-single-support sits **right at the edge** of the quasi-static
double-support envelope — which is exactly what "about to enter single support"
means. Slip (~4 mm) and the incidental swing-foot rise (~4 mm) are both near
their provisional limits; a real single-support hold (U8/U9) will need either
better foot friction/geometry, a wider foot, or a genuine balance controller —
not just the open-loop quasi-static shift.

`baselines/full_body_unload.json` is the frozen U7 result for regression.

### Open TODOs

- [ ] Foot friction (`analysis.ground.friction` slide = 1.0) and foot size
      (`provisional_geometry`) are provisional — both bound how cleanly Cara can
      unweight a foot.
- [ ] The maneuver is open-loop quasi-static; U9 will need COM feedback.
- [ ] U8: lift the unloaded foot 5–10 mm, hold, return to double support.
