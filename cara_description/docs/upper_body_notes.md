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
U4 passive arms ✅  →  U5 ears + head inertia ✅  →  U6 full regression ✅  ──┼── boundary
                                              U7 unload a foot  →  U8 lift a foot  → …
```

---

## Config hierarchy

```
config/
├── left_leg.yaml          SSOT: one leg + pelvis, fixed base
├── cara_lower_body.yaml    extends left_leg + mirror l→r + floating pelvis + standing/shift poses
├── cara_upper_body.yaml    FRAGMENT (not runnable alone): torso (U1) + head/neck (U2) + electronics (U3) + arms (U4) + ears (U5)
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

Through U6 the single-leg and lower-body generated URDF/MJCF stay
byte-identical (`git status` shows only `cara_full_body.*` changed). The
single-leg and lower-body validators (`validate_description` 201 / 340 —
U5 added an `l_*`↔`r_*` physical-link mass/COM symmetry check —
`fk_sanity`, `validate_mjcf`, `dynamic_check`, both generators `--check`) all
still pass unchanged — that is the hard gate. U6's `subsystem_summary.py`
independently confirms it: its pruned "lower body" stage reproduces
`cara_lower_body.yaml` to 0.0 g / 0.000°.

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

## Phase U4 — arms as rigid passive masses

One lumped mass per side, **welded at the shoulder** in a neutral hanging pose —
no shoulder joint, no arm swing, no manipulation. The mass covers the upper arm
+ forearm + hand + shoulder mechanics. `l_arm` is authored; `mirror` generates
`r_arm`, so the two are identical by construction.

| param | provisional value | `TODO` |
|---|---|---|
| arm mass (`upper_body.arm.mass`) | 0.180 kg **per side** | measured/CAD |
| shoulder (`arm_shoulder_x/y/z`) | (0, ±0.090, +0.120) m in the torso frame | measured/CAD |
| arm length (`arm_length`, inertia rod) | 0.190 m | measured/CAD |
| arm COM drop (`arm_com_z`) | −0.095 m below the shoulder (neutral hang) | measured/CAD |
| arm shape | `uniform_rod_z`, radius 0.022 m | measured/CAD |

Mass is a **single source of truth**: `dynamics.links.l_arm.mass: arm_mass`
resolves to `upper_body.arm.mass`, so a sweep of `upper_body.arm.mass` moves
*both* arms and the whole-body COM stays on the midline. (Sweeping
`dynamics.links.l_arm.mass` alone would desymmetrise the model — don't.)

**`l_shoulder`** (`type: fixed`, parent `torso`) — 0 DoF, welded, no servo. The
shoulder becomes 3 DoF in the long-term 20-DoF design; `actuated_joint_names`
still returns the 12 leg joints.

### Symmetry check (the U4 acceptance gate)

`l_arm` and `r_arm` resolve to identical mass (0.180 kg), COM (0, 0, −0.095) and
inertia diag, mirrored to y = ±0.090. Whole-body COM y = **−0.0000 m at every
standing pose** (`center_of_mass.py config/cara_full_body.yaml`) — the arms do
not shift the COM off the sagittal plane.

### Effect vs the frozen lower-body baseline (full upper body + arms)

Whole-body mass **2.06 → 4.37 kg**; whole-body COM (pelvis frame) −72 mm →
**+21 mm**. The arm COMs sit at z ≈ +60 mm (pelvis frame) — near pelvis height —
so +0.36 kg of arms lifts the COM only ≈ 4 mm on top of U3.

Whole-body inertia about the COM at `stand_nominal`
(`center_of_mass.py --pose`): **Ixx 0.0827 (roll), Iyy 0.0756 (pitch),
Izz 0.0138 (yaw) kg·m²**. The arms at ±0.09 m are the first subsystem that adds
noticeably to **roll and yaw** inertia rather than just raising the COM.

**Standing** (`stand_check.py --baseline`, all 3 poses PASS):

| pose | tilt | ΔCOM margin | Δpeak torque |
|------|------|-------------|--------------|
| stand_nominal | 0.42° (+0.26) | −2.6 mm → +30.5 | +0.157 → 0.235 N·m |
| semi_squat | 1.54° (+1.24) | −10.6 mm → +32.6 | +0.803 → 1.205 N·m |
| stand_wide | 0.50° (+0.33) | −2.7 mm → +30.4 | +0.331 → 0.533 N·m |

**Weight shift** (`weight_shift.py --baseline`): quasi-static envelope now
**~0.020 m** (lower body 0.040 → U2 0.030 → U3 0.025 → +arms 0.020). The ±0.03 m
demo fails on foot slip (3.7 mm of a 3 mm budget); ±0.020 m passes cleanly.
Standing is unaffected — only the *dynamic* shift margin keeps tightening as
upper-body mass accumulates.

| metric | lower body | + full upper body + arms | Δ |
|---|---|---|---|
| ±A loaded / unloaded foot | 14.1 / 6.1 N | 31.3 / 11.6 N | +17.2 / +5.4 |
| ±A pelvis roll | 0.35° | 0.90° | +0.55 |
| foot slip at ±0.03 m | 1.3 mm | 3.7 mm | +2.4 |
| **quasi-static shift limit** | **0.040 m** | **0.020 m** | **−0.020** |

### Sweeps — `subsystem_sweep.py` (acceptance: effect measurable & monotonic)

```
python3 scripts/subsystem_sweep.py config/cara_full_body.yaml \
    --param upper_body.arm.mass --values 0.0,0.10,0.18,0.30
```

| arm mass /side | whole-body mass | COM vs pelvis | worst knee τ | Ixx (roll) | Izz (yaw) | shift limit |
|---|---|---|---|---|---|---|
| 0.00 kg | 4.01 kg | +17.3 mm | 1.126 N·m | 0.0781 | 0.0108 | 0.030 m |
| 0.10 kg | 4.21 kg | +19.3 mm | 1.218 N·m | 0.0807 | 0.0125 | 0.020 m |
| 0.18 kg | 4.37 kg | +20.8 mm | 1.283 N·m | 0.0827 | 0.0138 | 0.020 m |
| 0.30 kg | 4.61 kg | +22.8 mm | 1.382 N·m | 0.0858 | 0.0158 | 0.020 m |

Arm mass 0 → 0.3 kg/side (+0.6 kg):

- **COM height +5.5 mm only** — arms hang near pelvis height, the smallest
  COM-per-kg of any U-subsystem so far.
- **Whole-body inertia** Ixx (roll) **+0.0076**, Izz (yaw) **+0.0050 kg·m²** —
  roughly linear (≈ +0.0017 Izz per 0.1 kg/side). This is the U4 headline: the
  arms' contribution is to the **inertia tensor**, and it must be understood
  before the shoulder is articulated (arm swing will then modulate exactly this
  roll/yaw inertia dynamically).
- **Worst-pose knee torque +0.26 N·m** (`semi_squat`) — arms loaded in a squat.
- **Weight-shift envelope drops 0.030 → 0.020 m** as soon as *any* arm mass is
  added, then holds — the step is the extra mass + roll inertia, not its size.

```
python3 scripts/subsystem_sweep.py config/cara_full_body.yaml \
    --param upper_body.arm.com_z --values -0.14,-0.095,-0.05
```

Arm COM drop −0.14 → −0.05 m (same mass, arm hangs higher): COM height +7.4 mm,
Ixx ±0.0025, **Izz unchanged**, torque ±0.01 N·m, shift limit unchanged — *how
far the arm hangs* barely matters next to *how much it weighs* and *how far it
is from the midline*.

**U4 acceptance criterion met:** the effect of arm mass on whole-body COM
(small, +5.5 mm across the sweep) and on the inertia tensor (roll +0.0076, yaw
+0.0050 kg·m²) is quantified and monotonic; the model stays sagittally
symmetric. No articulation was added.

---

## Phase U5 — ears and head asymmetry study

One plush-ear lump **and** a micro ear-twitch servo per side, both **welded to
the head** (`l_ear_joint` is `revolute` + `locked: true` — the 1-DoF ear twitch
of the 20-DoF design, held at 0; `l_ear_servo_mount` is `type: fixed`). `l_*`
authored, `mirror` → `r_*`. No ear motion yet — the fixed-mass inertia is the
thing to understand first.

| param (`upper_body.ear.*`) | provisional | `TODO` |
|---|---|---|
| `mass` | 0.020 kg/side | measured/CAD (plush ear shell) |
| `servo_mass` | 0.010 kg/side | measured (micro servo + horn) |
| `offset_x / offset_y / offset_z` | (0.005, **±0.090**… default 0.055, 0.050) m in the head frame | measured/CAD |
| `size` | 0.035 m cube (inertia approx) | measured/CAD |
| `servo_offset_x/y/z`, `servo_box_x/y/z` | (0.005, 0.030, 0.045) m / 0.020×0.012×0.023 m | measured |

Offsets are in the **head link frame**, whose origin **is** the coincident neck
yaw/roll/pitch axes — so `offset_y` is literally the ear's lateral moment arm
about the neck-yaw servo.

### The I ~ m r² study — `ear_inertia_study.py`

`leg_model.whole_body_inertia(spec, q, about=<frame>, links=<subset>)` (U5:
gained the `links` subset filter and an arbitrary reference frame) computes the
**head-subsystem** tensor — `{head, l_ear, r_ear, l_ear_servo, r_ear_servo}` —
about the neck axes (`about="head"`).

**Ears vs no ears**, about the neck axis, at `stand_nominal`:

| | Ixx (roll) | Iyy (nod) | Izz (yaw) | mass |
|---|---|---|---|---|
| head only | 0.001379 | 0.001414 | 0.000539 | 0.350 kg |
| head + ears (nominal) | 0.001668 | 0.001566 | 0.000689 | 0.410 kg |
| **ears add** | **+21 %** | +11 % | **+28 %** | +0.060 kg (+17 %) |

The ears are 17 % of the head *mass* but add **28 % of the yaw inertia** the
neck-yaw servo has to accelerate — the headline U5 result.

**Lateral-offset sweep** (`upper_body.ear.offset_y`, 20 → 90 mm):

| offset_y | Izz (yaw) | measured ΔIzz | m·r² prediction | roll Ixx |
|---|---|---|---|---|
| 0.020 m | 0.000584 | — | — | 0.001563 |
| 0.055 m (nominal) | 0.000689 | +0.000105 | +0.000105 | 0.001668 |
| 0.090 m | 0.000892 | +0.000308 | +0.000308 | 0.001871 |

The measured change tracks the point-mass parallel-axis prediction
`ΔIzz = 2·m_ear·(y² − y₀²)` **exactly** (ear COM is lumped at the joint, own-box
inertia is ~4 µkg·m² and constant). Moving a 20 g ear from 20 mm to 90 mm off
the axis **quintuples** its yaw-inertia contribution. Nod inertia (Iyy) is
unaffected by lateral offset, as expected (it depends on x² + z²).

Mass sweeps: `ear.mass` 10 → 60 g/side raises head yaw inertia +0.00033; the
servo (kept nearer the skull at 30 mm) matters ~3× less per gram than the ear.

### Whole-body effect (vs the frozen lower-body baseline)

Full upper body + arms **+ ears** = **2.06 → 4.43 kg** (+0.06 kg for the ears);
whole-body COM −72 mm → **+24 mm** (pelvis frame). Whole-body inertia about the
COM barely moves — the ears are near the yaw axis of the *whole body*
(Izz@COM +0.00001), high but light.

- **Standing** (`stand_check.py --baseline`): all 3 poses PASS — tilt +0.01°,
  margin −0.1 mm, peak torque +0.02 N·m vs U4. Ears are a rounding error for
  standing.
- **Weight shift**: quasi-static envelope **unchanged at ~0.020 m** — the ears
  add no measurable balance cost.

**U5 acceptance criterion met:** we now know the ear design *does* materially
affect head/neck rotational inertia (yaw +28 % at nominal, growing with the
square of the lateral offset — a real neck-yaw-servo sizing input) but does
**not** affect whole-body standing or weight-shift balance. No ear geometry is
chosen. Ear *motion* is the sanctioned next step once this fixed-mass model is
accepted.

---

## Phase U6 — full-body regression + per-subsystem summary

No new hardware. The whole standing / weight-shift suite is re-run on the
complete model, and `subsystem_summary.py` builds the full body up **one
subsystem at a time** — by pruning the composed `cara_full_body` spec down to
each stage — and measures the same MuJoCo metrics at every stage. Stage 0
("lower body", everything upper pruned) is cross-checked against
`cara_lower_body.yaml` loaded directly: Δmass 0.0 g, Δtilt 0.000° — the prune is
clean. The table lives at [`subsystem_summary.md`](subsystem_summary.md)
(regenerate with `python3 scripts/subsystem_summary.py --md docs/subsystem_summary.md`).

### The summary table (worst over the 3 standing poses; shift values at the limit)

| stage | kg | COM z vs pelvis | tilt° | margin mm | knee τ | Ixx@COM | Izz@COM | shift limit |
|---|---|---|---|---|---|---|---|---|
| lower body | 2.06 | −72 mm | 0.16 | 33.1 | 0.40 | 0.0208 | 0.0064 | 0.040 m |
| + torso | 3.26 | −5 mm | 0.33 | 31.1 | 0.83 | 0.0508 | 0.0099 | 0.030 m |
| + head/neck | 3.61 | +23 mm | 0.29 | 31.9 | 0.97 | 0.0770 | 0.0104 | 0.030 m |
| + electronics | 4.01 | +17 mm | 0.37 | 31.3 | 1.13 | 0.0781 | 0.0108 | 0.030 m |
| + arms | 4.37 | +21 mm | 0.42 | 30.9 | 1.28 | 0.0827 | 0.0138 | 0.020 m |
| + ears | 4.43 | +24 mm | 0.40 | 31.0 | 1.31 | 0.0868 | 0.0139 | 0.020 m |

Marginal contribution of each subsystem:

| added | Δmass | ΔCOM z | Δknee τ | ΔIxx@COM (roll) | ΔIzz@COM (yaw) | Δshift limit |
|---|---|---|---|---|---|---|
| torso | +1.20 kg | **+67 mm** | +0.43 | +0.030 | +0.0035 | **−0.010 m** |
| head/neck | +0.35 kg | +28 mm | +0.14 | +0.026 | +0.0005 | 0 |
| electronics (`pelvis_low`) | +0.40 kg | −5 mm | +0.16 | +0.001 | +0.0004 | 0 |
| arms | +0.36 kg | +4 mm | +0.16 | +0.005 | **+0.0030** | **−0.010 m** |
| ears | +0.06 kg | +4 mm | +0.03 | +0.004 | +0.0002 | 0 |

Read-off:

- **COM height** is dominated by the torso (+67 mm) then the head (+28 mm);
  putting the electronics low in the pelvis *pulls the COM back down* 5 mm.
- **Roll inertia (Ixx@COM)** jumps with the torso and head — mass placed high
  above the whole-body COM — and barely moves for the low electronics.
- **Yaw inertia (Izz@COM)** is driven by the **arms** (+0.0030, the widest
  masses); the ears are negligible for the *whole body* (+0.0002) even though
  they are +28 % of the *head-subsystem* yaw inertia (U5).
- **Weight-shift envelope** is cut only by the two big lateral-mass additions:
  the torso (0.040 → 0.030 m) and the arms (0.030 → 0.020 m). Head, electronics
  and ears cost nothing.

### Full-body milestones vs the frozen lower-body baseline

| milestone | lower body | full body (4.43 kg) | verdict |
|---|---|---|---|
| **Standing** — hold 3 poses 10 s (`stand_check.py --baseline`) | ✅ | tilt ≤ 1.6°, margin ≥ 30 mm, peak τ 1.23 N·m (41 % of ±3), FK < 1e-15 | **MET** |
| **Weight shift** ±0.020 m (`weight_shift.py --amplitude 0.02 --baseline`) | ✅ at ±0.040 m | 8/8 checks PASS — Fn 21.7 → 28.3 / 15.2 N, roll 0.36°, slip 1.5 mm, 15 % torque | **MET at ±0.020 m** |
| Weight shift ±0.030 m | ✅ | fails (foot slip 3.8 mm of 3 mm) | envelope now ±0.020 m |

**U6 acceptance criterion met:** the full standing + weight-shift suite passes
on the complete model, every metric is compared to the lower-body-only baseline,
and the saved per-subsystem table makes each addition's effect on COM, inertia,
torque and the shift envelope explicit. The whole-body model is physically
reasonable — this closes the morphology-validation phase (U1–U6). The next step
(U7, unloading one foot toward `Fz → 0`) is balance/control, not morphology.

---

## Open TODOs (U1–U6)

- [ ] Replace every `upper_body.*` / `electronics.*` / `dynamics.links.*` value with CAD / measured.
- [ ] Mount offsets are guesses — `pelvis_low` at −30 mm assumes there is room below the pelvis frame.
- [ ] Arm mass/length/COM/radius and the shoulder position are all provisional single-lump guesses.
- [ ] Ear mass / servo mass / offsets are provisional — the neck-yaw servo spec depends on them.
- [ ] The ±3 N·m actuator limit is provisional — semi-squat already needs 1.3 N·m steady-state
      (PD transients ~2.4 N·m); revisit once a servo is chosen.
- [ ] Follow-on: locked→driven neck / shoulder / ear motion, now that the fixed-mass model is validated.
- [ ] U7+: unload one foot → lift → single-support (balance/control — new controllers).
