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
lower body ✅  →  U1 torso ✅  →  U2 head/neck ✅  →  U3 Jetson+battery ✅  →
U4 passive arms  →  U5 ears + head inertia  →  U6 full regression  ──┼── boundary
                                              U7 unload a foot  →  U8 lift a foot  → …
```

---

## Config hierarchy

```
config/
├── left_leg.yaml          SSOT: one leg + pelvis, fixed base
├── cara_lower_body.yaml    extends left_leg + mirror l→r + floating pelvis + standing/shift poses
├── cara_upper_body.yaml    FRAGMENT (not runnable alone): torso (U1) + head/neck (U2) + electronics (U3) [+ arms/ears later]
│                           -- links + joints + dynamics + `upper_body:` / `electronics:` param blocks
└── cara_full_body.yaml     extends cara_lower_body + include cara_upper_body
```

**Load pipeline** (`leg_model.load_spec`): `extends` (deep-merge) → `include`
(additive: append `links`/`joints`/`frames_of_interest`, merge `dynamics.links`
and the `upper_body:` / `electronics:` param blocks) → `_resolve_mounts` (wire
`mount_from` joints to a mount preset) → `mirror` (once). Existing single-file
and lower-body paths are byte-identical.

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

## Phase U2 — head + neck

Added in the same fragment:

| param | provisional value | `TODO` |
|---|---|---|
| head mass (`dynamics.links.head.mass`) | 0.35 kg | measured/CAD (plush + skull + eye mechanics) |
| head COM (`head_com_x/z`) | (+0.010, 0, +0.050) m from the neck axes | measured/CAD |
| head shape | `solid_sphere`, radius 0.060 m | measured/CAD |
| neck base (`neck_base_z`) | 0.155 m up the torso frame | measured/CAD |
| neck length (`neck_length`) | 0.040 m to the coincident yaw/roll/pitch axes | measured/CAD |

**Neck joints** `neck_yaw` (+Z), `neck_roll` (+X), `neck_pitch` (+Y) exist
structurally (3 DoF in the long-term 20-DoF design) but are **`locked: true`** —
0 DoF, no servo, MJCF weld / URDF `type="fixed"` — until balance work needs
them. `leg_model.actuated_joint_names` still returns the 12 leg joints.

### Effect vs the lower-body-only baseline (torso + head)

Whole-body mass **2.06 → 3.61 kg**. Whole-body COM (pelvis frame): −72 mm →
**+22 mm** — now *above* the pelvis. The 0.35 kg head at ~0.28 m above the
pelvis adds ≈ 28 mm to the COM height on top of the torso's 67 mm.

`stand_check.py --baseline` (all poses still PASS):

| pose | tilt | Δ | Δpeak torque |
|------|------|---|--------------|
| stand_nominal | 0.29° | +0.13 | +0.096 → 0.174 N·m |
| semi_squat | 1.07° | +0.78 | +0.526 → 0.928 N·m |
| stand_wide | 0.32° | +0.16 | +0.211 → 0.413 N·m |

`weight_shift.py --baseline`: ±0.03 m still 8/8 PASS (loaded/unloaded foot
25.4 / 10.0 N, pelvis roll 0.65°); **quasi-static shift limit still ~0.030 m**
(the head keeps it there but more marginal — slip at 0.03 m is now 2.6 mm of
the 3 mm budget).

### Head-mass sweep — `subsystem_sweep.py` (light / nominal / heavy)

```
python3 scripts/subsystem_sweep.py config/cara_full_body.yaml \
    --param dynamics.links.head.mass --values 0.20,0.35,0.60
```

| head mass | whole-body mass | COM height (floor) | COM vs pelvis | worst standing torque (hip / knee / ankle) | shift limit |
|---|---|---|---|---|---|
| 0.20 kg | 3.46 kg | 0.284 m | +11.3 mm | 0.386 / 0.894 / 0.166 N·m | 0.030 m |
| 0.35 kg | 3.61 kg | 0.296 m | +22.5 mm | 0.407 / 0.971 / 0.174 N·m | 0.030 m |
| 0.60 kg | 3.86 kg | 0.312 m | +39.2 mm | 0.443 / **1.079** / 0.192 N·m | 0.030 m |

- **COM height rises +28 mm** across the sweep — linear in head mass.
- **Knee torque (worst pose = `semi_squat`) rises +0.19 N·m** — a heavier head is
  a real knee-servo-sizing concern in a squat; at 0.6 kg it's ~1.1 N·m
  steady-state (of a ±3 N·m provisional limit; PD-transient peaks reach ~2.4).
- Standing tilt and support margin barely move (COM stays centred at
  `stand_nominal`); the weight-shift limit holds at ~0.03 m.
- `upper_body.neck.length` 0 → 0.10 m: COM +10 mm, torque ±0.01 N·m — head
  *height* matters far less than head *mass*.

**U2 acceptance criterion met:** a heavier / higher head's effect on
whole-body COM and actuator demand is quantified and monotonic.

---

## Phase U3 — Jetson + battery, placement study

Two lumped masses welded to a **switchable** mount point:

| item | provisional mass | `TODO` |
|---|---|---|
| Jetson (module + carrier + heatsink) | 0.15 kg | measured |
| battery (2S LiPo ~5000 mAh) | 0.25 kg | measured |

`electronics.mounts` is a table of candidate points `{link, x/y/z offset}`;
`electronics.jetson.mount` / `.battery.mount` name one. A joint with
`mount_from: electronics.jetson.mount` gets its parent + origin resolved from
that preset (`leg_model._resolve_mounts`); `leg_model.apply_electronics_layout`
switches it at run time. Mass is `mass: jetson_mass` — one source of truth
(`link_inertials` now evaluates `mass` as an expression too).

Candidate mount points (`z` in the parent link's frame): `pelvis_low` (−0.030,
in the pelvis near the hips), `pelvis_top` (+0.020), `torso_low` (+0.015),
`torso_mid` (+0.075), `torso_high` (+0.130).

### Placement comparison — `placement_study.py` (0.40 kg to place)

```
python3 scripts/placement_study.py config/cara_full_body.yaml
```

| layout | jetson / battery | COM vs pelvis | worst knee τ | shift limit |
|--------|------------------|---------------|--------------|-------------|
| `both_pelvis_low` | pelvis_low / pelvis_low | **+17.3 mm** | 1.126 N·m | 0.030 m |
| `battery_low_jetson_torso` | torso_mid / pelvis_low | +22.5 mm | 1.137 N·m | 0.030 m |
| `both_torso_low` | torso_low / torso_low | +25.3 mm | 1.146 N·m | 0.030 m |
| `both_torso_mid` | torso_mid / torso_mid | +31.2 mm | 1.153 N·m | 0.030 m |
| `both_high` | torso_high / torso_high | +36.7 mm | 1.159 N·m | **0.020 m** |

- **COM spread +19 mm** across the placement options — putting the 0.40 kg low
  in the pelvis keeps the whole-body COM ~20 mm lower than "everything high".
- **Knee torque barely moves** (+0.03 N·m) — the electronics are proximal to
  the knee (they hang off the pelvis/torso, short moment arm at these poses).
- **Weight-shift envelope: `both_high` costs 0.01 m** (0.030 → 0.020) — a third
  of the envelope — while every low/mid placement keeps it at ~0.030 m.
- Read-off: **low in the pelvis** is the stability-preferred placement; the
  penalty for going higher is COM height and (past `torso_mid`) the shift
  envelope, not standing torque. `placement_study.py` reports; it does **not**
  choose.

### Full upper body vs the frozen lower-body baseline

Torso + head + electronics = **2.06 → 4.01 kg** (nearly ×2); whole-body COM
−72 mm → **+17 mm** (above the pelvis).

- **Standing** (`stand_check.py --baseline`): all 3 poses PASS. semi_squat
  tilt 1.30° (+1.00), peak knee torque 1.07 N·m (+0.67 — worst pose; PD
  transients reach ~2.4 N·m of the ±3 N·m provisional limit).
- **Weight shift**: the ±0.03 m demo now **fails on foot slip** (3.1 mm of a
  3 mm budget); ±0.025 m passes cleanly (2.0 mm). The full-body quasi-static
  envelope is **~0.025 m** — down from the lower body's 0.04 m and U2's 0.03 m.
  Standing is unaffected; only the *dynamic* shift margin tightens.

**U3 acceptance criterion met:** for every candidate placement the whole-body
COM, support margin, joint torque, tilt and weight-shift envelope are reported;
no placement is chosen automatically.

---

## Open TODOs (U1–U3)

- [ ] Replace every `upper_body.*` / `electronics.*` / `dynamics.links.*` value with CAD / measured.
- [ ] Mount offsets are guesses — `pelvis_low` at −30 mm assumes there is room below the pelvis frame.
- [ ] U4: left/right arm masses in a neutral pose (mirror-generated), no articulation.
