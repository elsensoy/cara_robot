# Cara — Upper body (staged mass/inertia analysis)

Companion to [`standing_notes.md`](standing_notes.md) and
[`weight_shift_notes.md`](weight_shift_notes.md).

The upper body is added as a **design-analysis tool**, not decoration. Each
piece answers: where is its mass, where is its COM, what inertia does it add,
how does it move the whole-body COM, how does it change required lower-body
torque, and how does it change the stable weight-shifting range?

Boundary: **U1–U6 are morphology / design validation** (the lower body just has
to stand and weight-shift with the existing joint PD, no new controllers).
**U7 onward is balance / control.**

```
lower body ✅  →  U1 torso ✅  →  U2 head/neck  →  U3 Jetson+battery  →
U4 passive arms  →  U5 ears + head inertia  →  U6 full regression  ──┼── boundary
                                              U7 unload a foot  →  U8 lift a foot  → …
```

---

## Config hierarchy

```
config/
├── left_leg.yaml          SSOT: one leg + pelvis, fixed base
├── cara_lower_body.yaml    extends left_leg + mirror l→r + floating pelvis + standing/shift poses
├── cara_upper_body.yaml    FRAGMENT (not runnable alone): torso (U1) [+ head/arms/ears later]
│                           -- links + joints + dynamics + an `upper_body:` param block
└── cara_full_body.yaml     extends cara_lower_body + include cara_upper_body
```

**Load pipeline** (`leg_model.load_spec`): `extends` (deep-merge) → `include`
(additive: append `links`/`joints`/`frames_of_interest`, merge `dynamics.links`
and the `upper_body:` param block) → `mirror` (once — expands l→r for legs and,
later, arms and ears together). Existing single-file / lower-body paths are
byte-identical.

`upper_body.*` numeric leaves become expression symbols
(`upper_body.torso.com_z` → `torso_com_z`), referenced by joint origins and
`dynamics.links` `com` / `box`. So a geometry sweep touches one place; a mass
sweep touches `dynamics.links.<link>.mass` (like the leg links).

**`type: fixed` / `locked: true` joints** (the U1 torso weld, later a
locked-at-zero neck): 0 DOF, no PD servo, no MJCF `<joint>` element / URDF
`type="fixed"`. `leg_model.actuated_joint_names` is the servo/qpos list;
`joint_names` is every joint (for FK). Flip `locked` to unlock later.

---

## Preserved regression baseline

`baselines/lower_body_standing.json` and `baselines/lower_body_weightshift.json`
are frozen snapshots. Every U-phase runs:

```bash
python3 scripts/stand_check.py  config/cara_full_body.yaml --baseline baselines/lower_body_standing.json
python3 scripts/weight_shift.py config/cara_full_body.yaml --baseline baselines/lower_body_weightshift.json
```

which prints the current value **and its delta** for: pelvis tilt, COM support
margin, peak/RMS actuator torque (standing); loaded/unloaded foot force, COM
margin, pelvis roll, foot slip, and the quasi-static shift limit (weight-shift).
Nothing in the baseline is dropped. `--json` writes a fresh snapshot.

The single-leg and lower-body suites (`validate_description` 201 / 331,
`fk_sanity`, `validate_mjcf`, `dynamic_check`, both generators `--check`) must
stay **byte-identical** through every phase — that is the hard gate.

---

## Phase U1 — rigid torso lump

`cara_upper_body.yaml`:

| param | provisional value | `TODO` |
|---|---|---|
| torso mass | 1.20 kg | measured/CAD (shell + structure) |
| weld height (`torso_attach_z`) | 0.035 m above pelvis origin | measured/CAD |
| torso COM (`torso_com_x/z`) | (0, 0, +0.075) m above the weld | measured/CAD |
| torso box (inertia approx) | 0.11 × 0.15 × 0.16 m | measured/CAD |

Welded to the pelvis (`base_to_torso`, `type: fixed`) — no waist joint yet.

### Effect vs the lower-body-only baseline

Whole-body mass **2.06 → 3.26 kg**. Whole-body COM (pelvis frame) rises from
**72 mm below** the pelvis to **5 mm below** — the torso lifts the COM ≈ 67 mm.

**Standing** (`stand_check.py --baseline`):

| pose | tilt | ΔCOM margin | Δpeak torque |
|------|------|-------------|--------------|
| stand_nominal | 0.33° (+0.17) | −2.1 mm → +31.0 | +0.084 N·m → 0.162 |
| semi_squat | 0.93° (+0.64) | −6.7 mm → +36.4 | +0.403 N·m → 0.805 |
| stand_wide | 0.35° (+0.18) | −2.1 mm → +31.0 | +0.161 N·m → 0.363 |

All three poses still PASS (no saturation — 0.8 N·m of ±3).

**Weight shift** (`weight_shift.py --baseline`):

| metric | lower body | + torso | Δ |
|---|---|---|---|
| centred foot force | 10.1 N | 16.0 N | +5.9 |
| ±0.03 m: loaded / unloaded foot | 14.1 / 6.1 N | 22.7 / 9.3 N | +8.6 / +3.2 |
| ±0.03 m: pelvis roll | 0.35° | 0.56° | +0.21 |
| ±0.03 m: 8 validation checks | 8/8 PASS | 8/8 PASS | — |
| **quasi-static shift limit** | **0.040 m** | **0.030 m** | **−0.010** |

The torso's mass, sitting well above the pelvis, **cuts the weight-shift
envelope by ~25 %** (0.04 → 0.03 m COM). At 0.04 m the full body now fails
(opposite foot unplants, 5 mm slip).

### Sweeps (acceptance: mass varies cleanly, effect measurable & monotonic)

```
dynamics.links.torso.mass  0.6 → 2.4 kg  :  whole-body COM +31 mm → −26 mm below pelvis (rises 57 mm)
upper_body.torso.com_z     0.04 → 0.12 m :  whole-body COM +18 mm → −11 mm below pelvis (rises 30 mm)
```

Both monotonic, no model-consistency breakage. **U1 acceptance criterion met.**

---

## Open TODOs (U1)

- [ ] Replace all `upper_body.torso.*` and `dynamics.links.torso.*` values with CAD / measured.
- [ ] The weld height / COM are guesses — the real number changes both COM and the shift limit.
- [ ] U2: neck link + head lump; add neck joints `locked: true` if they belong in the 20-DoF design.
