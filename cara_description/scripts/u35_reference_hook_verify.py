#!/usr/bin/env python3
"""U35 -- verify the new reference-offset hook in gait.py before any learning.

gait.py's U9 lateral roll-trim latches a balance reference once per step
(st["ref_ey"], the CoM-to-stance-foot lateral offset at the first controller
update of phase B) and regulates dy = ey - ref_ey for the rest of the step.
Earlier phase-indexed probing (see probe_phase_pulse.py and the U34 manifest)
found that brief, symmetric ankle-roll residual pulses are safe almost
everywhere in the gait EXCEPT a narrow, magnitude-insensitive window right at
this latch moment -- consistent with the latch picking up a biased sample
under perturbation and then anchoring the whole step's trim to it.

Per instruction, direct ankle-offset residual learning is paused (that
architecture has no non-invasive access to this reference at all). Instead,
gait.py itself now exposes an explicit, zero-default `reference_offset_fn`
hook (see gait.py's `run()` and `ss_step()`): an external controller can add
a correction, in the SAME units/sign as ey, to the latched reference at every
controller update, without ever overwriting the stored latch itself. This
script verifies that hook, in three steps, before any RL is attempted:

  1. Zero-offset equivalence: the hook's default must reproduce
     gait_frozen_baseline.py (a byte-for-byte pre-hook snapshot) bit-for-bit,
     across the full state AND control trace, not just summary metrics.
  2. Causal latch test: from IDENTICAL pre-latch states (same one-time lateral
     CoM-velocity kick, same seed/determinism), does substituting the nominal
     (unperturbed) run's own latched reference for the perturbed run's own
     (biased) latch change the outcome? This is a diagnostic use of the hook,
     not a deployable correction -- it requires knowing the nominal run's
     latch value in advance, which a real controller would not have.
  3. Small, smooth reference-offset probes on the UNPERTURBED nominal run:
     ramp a constant offset on and back off within a single step, recording
     the effective reference, resulting tracking error (dy), the teacher's
     own control response, and whether normal touchdown/support continues --
     plus an internal consistency check against the trim law's own algebra
     (delta_ctrl[ankle_roll] should equal -sfn*KPA*offset exactly).

No PPO, no training. If this hook doesn't behave as intended, that must be
found here, not downstream in a training run.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

import gait
import gait_frozen_baseline
import leg_model as lm
from cara_residual_env import DT, phase_schedule, phase_at


def _quat_rpy(q):
    """Local copy, cross-checked against cara_env.py's _quat_rpy -- same
    [w,x,y,z] -> (roll, pitch, yaw) convention used throughout this project."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    return roll, pitch


def _build_probe_geom(config_path):
    import mujoco
    import generate_mjcf
    spec = lm.load_spec(config_path)
    xml = generate_mjcf.build_mjcf(spec, dynamic=True)
    model = mujoco.MjModel.from_xml_string(xml)
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    return dict(floor_gid=gid("floor"), foot_gid={"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")})


def _foot_touch(data, floor_gid, foot_gid):
    touch = {p: False for p in foot_gid}
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if floor_gid not in pair:
            continue
        other = (pair - {floor_gid}).pop()
        for p, gid in foot_gid.items():
            if other == gid:
                touch[p] = True
    return touch


def _gait_balance_gains(config_path):
    spec = lm.load_spec(config_path)
    gc = (spec.get("analysis", {}) or {}).get("gait", {}) or {}
    bal = gc.get("balance", {}) or {}
    return dict(KPA=float(bal.get("kp_ankle_roll", 50.0)), KDA=float(bal.get("kd_ankle_roll", 10.0)),
                KPH=float(bal.get("kp_hip_roll", 15.0)))


def run_recorded(module, config_path, n_steps, reference_offset_fn=None,
                  kick_substep=None, kick_qvel_y=0.0, geom=None, schedule=None,
                  track_actuators=None, track_contact_stance=None, jn=None):
    """Runs module.run(...) (gait.py or gait_frozen_baseline.py) with
    mujoco.mj_step monkeypatched (non-invasive; module source untouched by
    this wrapper) to record qpos/ctrl every substep, and -- only when geom
    and schedule are supplied -- ey and foot-touch per substep too.

    track_actuators: optional list of joint names -- also records
    actuator_force and whether it saturates model.actuator_forcerange
    (same check gait.py's own M["sat"] uses), plus whether ctrl itself
    saturates actuator_ctrlrange, for just these actuators (kept selective
    to bound memory over long runs).
    track_contact_stance: optional dict step_idx -> stance prefix ("l_"/"r_")
    -- when the current substep's phase_at().step_idx matches, records the
    peak normal contact force between that stance foot and the floor.

    Restores mj_step afterward regardless of outcome."""
    import mujoco
    orig = mujoco.mj_step
    rec = {"substep": 0, "qpos": [], "ctrl": [], "rows": []}
    aid_cache = {}

    def wrapped(model, data):
        i = rec["substep"]
        if kick_substep is not None and i == kick_substep:
            data.qvel[1] += kick_qvel_y
        rec["qpos"].append(np.array(data.qpos, dtype=np.float64).copy())
        rec["ctrl"].append(np.array(data.ctrl, dtype=np.float64).copy())
        if geom is not None and schedule is not None:
            info = phase_at(i, schedule)
            row = {"substep": i, "phase": info["phase"], "lead": info["lead"], "step_idx": info["step_idx"]}
            if info["lead"] is not None:
                stance = gait.OTHER[info["lead"]]
                row["ey"] = float(data.subtree_com[0][1] - data.geom_xpos[geom["foot_gid"][stance]][1])
                row["touch"] = _foot_touch(data, geom["floor_gid"], geom["foot_gid"])
            if track_actuators:
                if not aid_cache:
                    for n in track_actuators:
                        aid_cache[n] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in track_actuators:
                    aidx = aid_cache[n]
                    force = float(data.actuator_force[aidx])
                    frange = model.actuator_forcerange[aidx]
                    crange = model.actuator_ctrlrange[aidx]
                    ctrl_val = float(data.ctrl[aidx])
                    row[f"force_{n}"] = force
                    row[f"force_sat_{n}"] = bool(abs(force) >= frange[1] - 0.02)
                    row[f"ctrl_clip_{n}"] = bool(ctrl_val <= crange[0] + 1e-9 or ctrl_val >= crange[1] - 1e-9)
            if track_contact_stance is not None and info["step_idx"] in track_contact_stance:
                stance_p = track_contact_stance[info["step_idx"]]
                # SUM simultaneous contacts, not max -- a foot can rest on several
                # contact points (e.g. corners of a box collision geom) at once, and
                # the load-bearing quantity is their total normal force, not the
                # single largest point. An earlier version took max() here, which
                # undercounts total stance-foot load whenever more than one contact
                # point is active. Still lags data.qpos/data.ctrl by one substep in
                # this row (contacts/forces here are computed by the PRECEDING
                # mj_step call's constraint solve, read before this call's own
                # orig(model, data) runs) -- a well-defined quantity, just not
                # synchronized to the same instant as the ctrl value in this row;
                # not fixed here, disclosed instead.
                total_f = 0.0
                fgid = geom["foot_gid"][stance_p]
                for ci in range(data.ncon):
                    c = data.contact[ci]
                    pair = {c.geom1, c.geom2}
                    if geom["floor_gid"] in pair and fgid in pair:
                        result = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, result)
                        total_f += abs(float(result[0]))
                row["contact_force_n"] = total_f
            rec["rows"].append(row)
        orig(model, data)
        rec["substep"] = i + 1
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            raise RuntimeError(f"non-finite state at substep {i}")

    mujoco.mj_step = wrapped
    try:
        kwargs = {} if reference_offset_fn is None else {"reference_offset_fn": reference_offset_fn}
        rec["exit_code"] = module.run(config_path, n_steps, False, None, None, **kwargs)
    finally:
        mujoco.mj_step = orig
    rec["qpos"] = np.array(rec["qpos"])
    rec["ctrl"] = np.array(rec["ctrl"])
    return rec


def peak_tilt_deg(qpos_trace):
    peaks = []
    for q in qpos_trace:
        roll, pitch = _quat_rpy(q[3:7])
        peaks.append(math.degrees(max(abs(roll), abs(pitch))))
    return max(peaks)


FALL_TILT_DEG = 40.0  # matches CaraResidualEnv/cara_env's own fall threshold -- use this for
                       # pass/fail comparisons; tilt values seen AFTER this point (up to 180deg,
                       # a collapsed/inverted pose) describe post-failure motion, not a graded signal.


def first_exceed_40deg(qpos_trace, dt=DT):
    """Returns (fell: bool, substep_idx or None, time_s or None) for the FIRST
    substep at which tilt exceeds FALL_TILT_DEG -- the reportable failure
    point, instead of a 180deg post-collapse peak that adds no diagnostic
    value once the robot has already tipped over."""
    for i, q in enumerate(qpos_trace):
        roll, pitch = _quat_rpy(q[3:7])
        if math.degrees(max(abs(roll), abs(pitch))) > FALL_TILT_DEG:
            return True, i, i * dt
    return False, None, None


# --------------------------------------------------------------------- #
def test1_zero_offset_equivalence(config_path, n_steps):
    print("=== test 1: zero-offset equivalence vs frozen baseline ===")
    live = run_recorded(gait, config_path, n_steps)
    frozen = run_recorded(gait_frozen_baseline, config_path, n_steps)
    same_shape = live["qpos"].shape == frozen["qpos"].shape and live["ctrl"].shape == frozen["ctrl"].shape
    if not same_shape:
        print(f"  FAIL: shape mismatch qpos {live['qpos'].shape} vs {frozen['qpos'].shape}")
        return False
    max_qpos_diff = float(np.max(np.abs(live["qpos"] - frozen["qpos"])))
    max_ctrl_diff = float(np.max(np.abs(live["ctrl"] - frozen["ctrl"])))
    ok = max_qpos_diff == 0.0 and max_ctrl_diff == 0.0
    print(f"  substeps compared: {live['qpos'].shape[0]}  qpos dims: {live['qpos'].shape[1]}")
    print(f"  max |qpos diff| = {max_qpos_diff:.3e}   max |ctrl diff| = {max_ctrl_diff:.3e}")
    print(f"  {'PASS -- bit-for-bit identical' if ok else 'FAIL -- hook changes default behaviour'}")
    return ok


# --------------------------------------------------------------------- #
def test2_causal_latch(config_path, n_steps, geom, schedule):
    print("\n=== test 2: causal latch test ===")
    b_onset = next(s for s, e, name, lead, si in schedule if name == "B" and si == 0)
    kick_substep = b_onset - 20  # 40ms before the phase-B onset (latch happens at the first B call)
    print(f"  step_idx=0 phase-B onset at substep {b_onset} ({b_onset*DT:.2f}s); "
          f"kick applied at substep {kick_substep} ({kick_substep*DT:.2f}s), 40ms earlier")

    # (a) nominal (unperturbed) run -- records its own latch value via the hook itself
    nominal_latch = {}

    def record_only(ref_ey):
        nominal_latch.setdefault("value", ref_ey)
        return 0.0

    nominal = run_recorded(gait, config_path, n_steps, reference_offset_fn=record_only, geom=geom, schedule=schedule)
    ref_ey_nominal = nominal_latch["value"]
    print(f"  nominal (unperturbed) latch ref_ey = {ref_ey_nominal*1e3:+.3f} mm")

    # calibrate the kick: smallest magnitude (of a short list) that visibly biases the
    # latch and measurably changes the outcome, without itself blowing up the state
    # before the latch is even reached.
    candidates = [0.001, 0.002, 0.004, 0.008, 0.015]
    chosen = None
    for kick in candidates:
        perturbed_latch = {}

        def record_only_p(ref_ey, _store=perturbed_latch):
            _store.setdefault("value", ref_ey)
            return 0.0

        try:
            trial = run_recorded(gait, config_path, n_steps, reference_offset_fn=record_only_p,
                                  kick_substep=kick_substep, kick_qvel_y=kick, geom=geom, schedule=schedule)
        except RuntimeError as e:
            print(f"  kick={kick:.2f} m/s: non-finite before reaching the latch ({e}) -- too large, skip")
            continue
        ref_ey_perturbed = perturbed_latch.get("value")
        if ref_ey_perturbed is None:
            print(f"  kick={kick:.2f} m/s: never reached the latch (fell earlier) -- too large, skip")
            continue
        bias_mm = (ref_ey_perturbed - ref_ey_nominal) * 1e3
        tilt = peak_tilt_deg(trial["qpos"])
        print(f"  kick={kick:.2f} m/s: latch biased by {bias_mm:+.3f} mm, peak_tilt={tilt:.2f}deg, "
              f"exit_code={trial['exit_code']}")
        if chosen is None and (abs(bias_mm) > 0.5 and (tilt > 8.0 or trial["exit_code"] != 0)):
            chosen = dict(kick=kick, ref_ey_perturbed=ref_ey_perturbed, bias_mm=bias_mm,
                          own_latch_trial=trial)
    if chosen is None:
        print("  no tested kick both biased the latch AND produced a materially worse outcome -- "
              "cannot run the causal comparison with this kick family; reporting inconclusive, not fabricating a result.")
        return None

    kick = chosen["kick"]
    ref_ey_perturbed = chosen["ref_ey_perturbed"]
    trial_own_latch = chosen["own_latch_trial"]
    print(f"\n  chosen kick = {kick:.2f} m/s (bias {chosen['bias_mm']:+.3f} mm)")

    # (b) perturbed run, own (biased) latch -- already have it as trial_own_latch
    tilt_own = peak_tilt_deg(trial_own_latch["qpos"])
    print(f"  (b) perturbed, OWN latch:      peak_tilt={tilt_own:.2f}deg  exit_code={trial_own_latch['exit_code']}")

    # (c) perturbed run, SAME kick, but reference_offset_fn substitutes the nominal run's
    # latched value in place of this run's own latch -- identical pre-latch state (same kick,
    # same substep, deterministic physics), diverging only in which reference the trim uses.
    def substitute_nominal(ref_ey):
        return ref_ey_nominal - ref_ey  # forces ref_ey_effective == ref_ey_nominal, every call

    trial_substituted = run_recorded(gait, config_path, n_steps, reference_offset_fn=substitute_nominal,
                                      kick_substep=kick_substep, kick_qvel_y=kick, geom=geom, schedule=schedule)
    tilt_sub = peak_tilt_deg(trial_substituted["qpos"])
    print(f"  (c) perturbed, NOMINAL latch substituted: peak_tilt={tilt_sub:.2f}deg  "
          f"exit_code={trial_substituted['exit_code']}")

    # sanity: (a) vs (c)'s pre-kick substeps must be identical (same physics up to the kick)
    pre_kick_diff = float(np.max(np.abs(
        trial_own_latch["qpos"][:kick_substep] - trial_substituted["qpos"][:kick_substep])))
    print(f"  sanity: trials (b) and (c) identical before the kick? max diff = {pre_kick_diff:.3e} "
          f"({'OK' if pre_kick_diff == 0.0 else 'UNEXPECTED -- investigate'})")

    improved = tilt_sub < tilt_own - 1.0 or (trial_substituted["exit_code"] == 0 and trial_own_latch["exit_code"] != 0)
    print(f"\n  interpretation: substituting the nominal latch "
          f"{'clearly reduced the disturbance response (supports the latch-bias hypothesis)' if improved else 'did NOT clearly help -- the biased latch is not shown to be the (sole) cause; something else may also be at play'}.")
    return dict(kick=kick, tilt_own=tilt_own, tilt_substituted=tilt_sub,
                exit_own=trial_own_latch["exit_code"], exit_substituted=trial_substituted["exit_code"],
                pre_kick_bitexact=(pre_kick_diff == 0.0), improved=improved)


# --------------------------------------------------------------------- #
def test3_small_smooth_offset_probe(config_path, n_steps, geom, schedule, gains):
    print("\n=== test 3: small, smooth reference-offset probes (unperturbed nominal run) ===")
    b_onset = next(s for s, e, name, lead, si in schedule if name == "B" and si == 0)
    d_end = next(s for s, e, name, lead, si in schedule if name == "E" and si == 0)  # end of D+settle == start of E
    call_span = d_end - b_onset  # number of ss_step calls across B+C+D(+settle) for step 0
    ramp_calls = 200  # 0.4s smooth ramp on/off, well inside the ~6200-call span

    def smoothstep(u):
        u = max(0.0, min(1.0, u))
        return 3 * u * u - 2 * u * u * u

    # zero-offset baseline is deterministic -- compute it once, not once per magnitude/sign
    baseline = run_recorded(gait, config_path, n_steps, geom=geom, schedule=schedule)
    jn = lm.actuated_joint_names(lm.load_spec(config_path))
    a_roll_idx = jn.index("r_ankle_roll")  # step_idx=0 lead=l_ -> stance=r_
    sfn = 1.0  # -SIDE["r_"] = -(-1.0) = +1.0

    # magnitudes chosen so KPA*offset stays within the ~0.05 rad envelope
    # probe_phase_pulse.py established as bidirectionally safe for direct
    # ankle_roll residual pulses (KPA=50 rad/rad here, so 1mm -> 0.05 rad,
    # 0.2mm -> 0.01 rad) -- the earlier attempt at 2/5/10mm implied
    # 0.1-0.5 rad corrections, well outside that envelope, and both
    # magnitudes (even the "small" 2mm one, on the sign that made it worse)
    # tipped the robot over -- a scale-calibration miss, not a hook finding.
    results = {}
    for mag_mm in (0.2, 0.5, 1.0):
        for sign, label in ((+1, "+"), (-1, "-")):
            mag = sign * mag_mm * 1e-3
            call_count = {"n": 0}
            offset_trace = []

            def offset_fn(ref_ey, _mag=mag, _cc=call_count, _trace=offset_trace):
                k = _cc["n"]
                _cc["n"] += 1
                if k < ramp_calls:
                    f = smoothstep(k / ramp_calls)
                elif k > call_span - ramp_calls:
                    f = smoothstep((call_span - k) / ramp_calls)
                else:
                    f = 1.0
                off = _mag * max(0.0, f)
                _trace.append(off)
                return off

            trial = run_recorded(gait, config_path, n_steps, reference_offset_fn=offset_fn,
                                  geom=geom, schedule=schedule)
            n = min(len(trial["ctrl"]), len(baseline["ctrl"]))
            delta_ctrl_full = trial["ctrl"][:n] - baseline["ctrl"][:n]
            predicted = np.array([-sfn * gains["KPA"] * off for off in offset_trace[:n]])
            observed = delta_ctrl_full[:n, a_roll_idx]
            n_cmp = min(len(predicted), len(observed))
            pred_err = float(np.max(np.abs(predicted[:n_cmp] - observed[:n_cmp]))) if n_cmp else float("nan")

            step0_rows_trial = [r for r in trial["rows"] if r.get("step_idx") == 0 and "ey" in r]
            step0_rows_base = [r for r in baseline["rows"] if r.get("step_idx") == 0 and "ey" in r]
            peak_offset = max((abs(o) for o in offset_trace), default=0.0)
            peak_dy_trial = max((abs(r["ey"]) for r in step0_rows_trial), default=float("nan"))
            touch_ok = all(any(r["touch"].values()) for r in step0_rows_trial[-50:]) if step0_rows_trial else False
            tilt_trial = peak_tilt_deg(trial["qpos"])
            tilt_base = peak_tilt_deg(baseline["qpos"])

            key = f"{label}{mag_mm}mm"
            results[key] = dict(peak_offset_mm=peak_offset * 1e3, tracking_dy_peak_mm=peak_dy_trial * 1e3,
                                 touch_after_step0=touch_ok, tilt_trial_deg=tilt_trial, tilt_baseline_deg=tilt_base,
                                 exit_code=trial["exit_code"], predicted_vs_observed_ctrl_max_err=pred_err)
            print(f"  offset {label}{mag_mm}mm: peak|offset|={peak_offset*1e3:.2f}mm  "
                  f"touch_after_step0={touch_ok}  tilt={tilt_trial:.2f}deg (baseline {tilt_base:.2f}deg)  "
                  f"exit_code={trial['exit_code']}  trim-law check max_err={pred_err:.2e}")
    survived = [k for k, v in results.items() if v["exit_code"] == 0 and v["touch_after_step0"]]
    print(f"\n  survived (milestone met + touchdown continued): {survived}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=gait.DEFAULT_CONFIG)
    ap.add_argument("--n-steps", type=int, default=2)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    geom = _build_probe_geom(args.config)
    schedule, total_substeps = phase_schedule(args.n_steps, (lm.load_spec(args.config).get("analysis", {}) or {}).get("gait", {}) or {})
    gains = _gait_balance_gains(args.config)

    out = {}
    out["test1_zero_offset_equivalence"] = test1_zero_offset_equivalence(args.config, args.n_steps)
    out["test2_causal_latch"] = test2_causal_latch(args.config, args.n_steps, geom, schedule)
    out["test3_small_smooth_offset_probe"] = test3_small_smooth_offset_probe(args.config, args.n_steps, geom, schedule, gains)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
