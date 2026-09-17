#!/usr/bin/env python3
"""U35 -- bounded 12-case test of the reference-offset hook's authority,
with NO external disturbance. Answers: does nudging the U9 roll-trim's
balance reference offer usable correction in BOTH directions, at the phases
where a learned corrector would need it?

This is deliberately separated from the latch-causality question (answered
in u35_latch_causality.py): a pulse test here shows whether the INTERFACE
has bidirectional authority; it says nothing by itself about whether a
biased latch CAUSES the earlier residual-noise failures. Neither result
justifies calling anything a "mechanical bifurcation" on its own.

Design (12 cases = 2 stance legs x 3 timings x 2 signs):
  stance legs:  step_idx=0 (stance=r_, lead=l_), step_idx=1 (stance=l_, lead=r_)
  timings:      early single support (~0.2s after liftoff, phase B)
                mid single support   (middle of the swing, phase C)
                before touchdown     (~0.2s before the foot lands, late phase D)
  signs:        +0.2mm and -0.2mm, in WORLD Y-axis metres (ey's own convention:
                ey = com_y - stance_foot_y), plus a data-derived toward/away
                label (see `toward_away_label` below) -- amplitude is NOT
                expanded beyond +/-0.2mm this round, per instruction.

Each case: a brief, smooth, FIXED-WIDTH pulse (smoothstep on/hold/off,
total ~0.2s) is applied via gait.py's reference_offset_fn, then the run
continues through the REST of the walk (not just the pulse window). Logged
per case: reference tracking error (dy, using the reconstructed effective
reference), the trim's own command (ctrl delta on the stance ankle/hip roll
vs. the zero-offset baseline), actuator force/ctrl saturation on those two
actuators, peak stance-foot/floor contact-force change vs. baseline, the
first substep of qpos divergence from baseline, and the 40deg fall verdict
(see FALL_TILT_DEG in u35_reference_hook_verify.py -- values after that are
post-failure motion, not reported as a graded signal).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import gait
import leg_model as lm
from cara_residual_env import DT, phase_schedule
from u35_reference_hook_verify import (
    run_recorded, _build_probe_geom, _gait_balance_gains, first_exceed_40deg, FALL_TILT_DEG,
)


def smoothstep(u):
    u = max(0.0, min(1.0, u))
    return 3 * u * u - 2 * u * u * u


def pulse_envelope(k, center, half_width, ramp):
    on_start, on_end = center - half_width - ramp, center - half_width
    off_start, off_end = center + half_width, center + half_width + ramp
    if k < on_start or k >= off_end:
        return 0.0
    if k < on_end:
        return smoothstep((k - on_start) / ramp)
    if k < off_start:
        return 1.0
    return 1.0 - smoothstep((k - off_start) / ramp)


def make_offset_fn(local_center, mag, half_width, ramp):
    counter = {"n": -1}
    trace = []

    def fn(ref_ey):
        counter["n"] += 1
        f = pulse_envelope(counter["n"], local_center, half_width, ramp)
        off = mag * f
        trace.append(off)
        return off

    return fn, trace


def step_calls(config_path):
    spec = lm.load_spec(config_path)
    gc = (spec.get("analysis", {}) or {}).get("gait", {}) or {}
    ramp_calls = int(float(gc.get("ramp_seconds", 4.0)) / DT)
    swing_calls = int(float(gc.get("swing_seconds", 4.0)) / DT)
    dhold_calls = int(0.4 / DT)  # gait.py's own hardcoded 0.4s post-D settle, not a YAML field
    calls_per_step = ramp_calls + swing_calls + ramp_calls + dhold_calls  # B + C + (D-ramp + D-hold)
    return dict(ramp=ramp_calls, swing=swing_calls, dhold=dhold_calls, per_step=calls_per_step)


def timing_center(calls, label):
    if label == "early":
        return 100  # ~0.2s after phase-B onset
    if label == "mid":
        return calls["ramp"] + calls["swing"] // 2  # middle of phase C
    if label == "late":
        return calls["ramp"] + calls["swing"] + calls["ramp"] - 100  # ~0.2s before D-ramp ends (touchdown)
    raise ValueError(label)


def analyze_case(trial, base, offset_trace, step_idx, stance, a_roll, h_roll, ref_ey_baseline, mag_mm, sign_label,
                  calls, timing_label):
    n = min(len(trial["ctrl"]), len(base["ctrl"]))
    # first state departure from baseline (qpos), across the full compared trace
    diffs = np.max(np.abs(trial["qpos"][:n] - base["qpos"][:n]), axis=1)
    nz = np.nonzero(diffs > 1e-9)[0]
    first_departure_substep = int(nz[0]) if len(nz) else None

    # ss_step (and therefore offset_trace) is only called during B/C/D -- NOT during A or E,
    # which also produce "ey" rows here since our own recording doesn't depend on gait.py
    # calling ss_step. Filtering to phase in (B,C,D) keeps this 1:1 aligned with offset_trace
    # AND ensures index 0 is the actual latch instant (start of B), not the start of phase A.
    case_rows = [r for r in trial["rows"] if r.get("step_idx") == step_idx and r.get("phase") in ("B", "C", "D")]
    base_rows = [r for r in base["rows"] if r.get("step_idx") == step_idx and r.get("phase") in ("B", "C", "D")]
    m = min(len(case_rows), len(offset_trace))
    dy_eff = [case_rows[k]["ey"] - (ref_ey_baseline + offset_trace[k]) for k in range(m)]
    peak_dy = max(abs(v) for v in dy_eff) if dy_eff else float("nan")

    # trim command = ctrl on a_roll/h_roll (not force); pull from the full ctrl arrays via absolute substep
    substeps = [r["substep"] for r in case_rows[:m]]
    jn = gait_joint_names_cache.get("jn")
    a_idx, h_idx = jn.index(a_roll), jn.index(h_roll)
    delta_ctrl_a = trial["ctrl"][substeps, a_idx] - base["ctrl"][substeps, a_idx]
    delta_ctrl_h = trial["ctrl"][substeps, h_idx] - base["ctrl"][substeps, h_idx]
    peak_delta_ctrl_a = float(np.max(np.abs(delta_ctrl_a))) if len(delta_ctrl_a) else float("nan")
    peak_delta_ctrl_h = float(np.max(np.abs(delta_ctrl_h))) if len(delta_ctrl_h) else float("nan")

    sat_a = any(r.get(f"force_sat_{a_roll}") or r.get(f"ctrl_clip_{a_roll}") for r in case_rows)
    sat_h = any(r.get(f"force_sat_{h_roll}") or r.get(f"ctrl_clip_{h_roll}") for r in case_rows)

    contact_trial = [r.get("contact_force_n") for r in case_rows if "contact_force_n" in r]
    contact_base = [r.get("contact_force_n") for r in base_rows if "contact_force_n" in r]
    cN = min(len(contact_trial), len(contact_base))
    peak_contact_delta = float(max(abs(contact_trial[k] - contact_base[k]) for k in range(cN))) if cN else float("nan")

    fell, fell_substep, fell_time = first_exceed_40deg(trial["qpos"])

    away = (mag_mm * (1 if sign_label == "+" else -1)) * ref_ey_baseline > 0
    toward_away = "away" if away else "toward"

    return dict(
        step_idx=step_idx, stance=stance, timing=timing_label, sign=sign_label, mag_mm=mag_mm,
        toward_away=toward_away, ref_ey_baseline_mm=ref_ey_baseline * 1e3,
        peak_reference_error_mm=peak_dy * 1e3,
        peak_delta_ctrl_ankle_roll_rad=peak_delta_ctrl_a, peak_delta_ctrl_hip_roll_rad=peak_delta_ctrl_h,
        actuator_saturation_ankle_roll=sat_a, actuator_saturation_hip_roll=sat_h,
        peak_contact_force_delta_n=peak_contact_delta,
        first_state_departure_substep=first_departure_substep,
        fell_40deg=fell, fell_substep=fell_substep, fell_time_s=fell_time,
    )


gait_joint_names_cache = {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=gait.DEFAULT_CONFIG)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    geom = _build_probe_geom(args.config)
    gains = _gait_balance_gains(args.config)
    calls = step_calls(args.config)
    spec = lm.load_spec(args.config)
    gait_joint_names_cache["jn"] = lm.actuated_joint_names(spec)
    schedule2, _ = phase_schedule(2, (spec.get("analysis", {}) or {}).get("gait", {}) or {})

    print(f"calls per step: {calls}  (B={calls['ramp']}, C={calls['swing']}, D={calls['ramp']+calls['dhold']})")

    legs = [(0, "r_"), (1, "l_")]
    baselines = {}
    ref_ey_baseline = {}
    for step_idx, stance in legs:
        n_steps = step_idx + 1
        a_roll, h_roll = stance + "ankle_roll", stance + "hip_roll"
        base = run_recorded(gait, args.config, n_steps, geom=geom, schedule=schedule2,
                             track_actuators=[a_roll, h_roll], track_contact_stance={step_idx: stance})
        baselines[step_idx] = base
        # the latch happens at the FIRST ss_step call, i.e. the first substep of phase B --
        # not the first substep of phase A (which also has "ey" computed here, but gait.py
        # never calls ss_step, hence never latches, during A).
        step_rows = [r for r in base["rows"] if r.get("step_idx") == step_idx and r.get("phase") == "B"]
        ref_ey_baseline[step_idx] = step_rows[0]["ey"]
        print(f"  baseline step_idx={step_idx} (stance={stance}): ref_ey={ref_ey_baseline[step_idx]*1e3:+.3f}mm  "
              f"exit_code={base['exit_code']}")

    results = []
    for step_idx, stance in legs:
        base = baselines[step_idx]
        a_roll, h_roll = stance + "ankle_roll", stance + "hip_roll"
        for timing_label in ("early", "mid", "late"):
            for sign_label, sign in (("+", +1), ("-", -1)):
                mag_mm = 0.2
                mag = sign * mag_mm * 1e-3
                n_steps = step_idx + 1
                center = timing_center(calls, timing_label)
                offset_fn, offset_trace = make_offset_fn(center, mag, half_width=25, ramp=25)
                trial = run_recorded(gait, args.config, n_steps, reference_offset_fn=offset_fn, geom=geom,
                                      schedule=schedule2, track_actuators=[a_roll, h_roll],
                                      track_contact_stance={step_idx: stance})
                result = analyze_case(trial, base, offset_trace, step_idx, stance, a_roll, h_roll,
                                       ref_ey_baseline[step_idx], mag_mm, sign_label, calls, timing_label)
                results.append(result)
                print(f"  step{step_idx}/{stance}  {timing_label:>5}  {sign_label}{mag_mm}mm ({result['toward_away']}): "
                      f"peak_dy={result['peak_reference_error_mm']:+.3f}mm  "
                      f"d_ctrl(ankle_roll)={result['peak_delta_ctrl_ankle_roll_rad']:.4f}rad  "
                      f"sat(ankle/hip)={result['actuator_saturation_ankle_roll']}/{result['actuator_saturation_hip_roll']}  "
                      f"contact_dF={result['peak_contact_force_delta_n']:.3f}N  "
                      f"fell@40deg={result['fell_40deg']}"
                      + (f" (t={result['fell_time_s']:.2f}s)" if result["fell_40deg"] else ""))

    n_fell = sum(1 for r in results if r["fell_40deg"])
    print(f"\n=== summary: {n_fell}/12 cases exceeded {FALL_TILT_DEG}deg ===")
    by_dir = {}
    for r in results:
        by_dir.setdefault(r["toward_away"], []).append(r["fell_40deg"])
    for d, fells in by_dir.items():
        print(f"  {d}: {sum(fells)}/{len(fells)} fell")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
