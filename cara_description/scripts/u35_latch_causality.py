#!/usr/bin/env python3
"""U35 -- latch-causality test using an ORIGINAL residual-noise failure
(not a new external kick). The earlier velocity-kick version conflated a
different disturbance mechanism with the question we actually need answered:
does the biased latch materially contribute to the residual-noise failures
found in the truncated-window sweep (falls clustered at decisions ~275-345,
phase B, even at std=0.0005 rad)?

Method: find one seed at std=0.0005 that reproduces that failure through
CaraResidualEnv (same mechanism as the original sweep), then run two matched
trials from IDENTICAL prior state (same seed -> same residual draws, same
deterministic physics):
  Run A: normal latch (reference_offset_fn=None).
  Run B: identical perturbations, but reference_offset_fn forces the step's
          effective reference to the UNDISTURBED (zero-residual) run's own
          latch value for that same step, for the whole step.
Both runs are bit-for-bit identical up to the point ss_step is first called
(the latch instant) -- reference_offset_fn plays no role before then, so
"everything before the intervention matches, including the teacher's
internal state" holds by construction, and is verified below, not assumed.

Three possible outcomes, per instruction:
  1. B recovers, A fails  -> the latch value materially contributes.
  2. both fail            -> latch correction alone is insufficient.
  3. B merely delays the failure -> report that limited effect, not full credit.

This is a diagnostic use of the hook (it requires knowing the undisturbed
latch value in advance, which a real controller would not have) -- not a
deployable correction.

Also addresses the earlier kick-based test's outstanding caveat: a
data.qvel[1] increment changes the FLOATING BASE's own linear velocity
component, which is not generally identical to the WHOLE-BODY CoM velocity
(the CoM also depends on leg-joint velocities and is offset from the free
joint's origin). The actual achieved CoM-y velocity change is measured
directly here, not assumed equal to the requested kick.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from cara_residual_env import CaraResidualEnv, CaraResidualEnvConfig, DT
from u35_reference_hook_verify import FALL_TILT_DEG


def find_failing_seed(std, bound, max_decisions, seed_start=4000, n_tries=20):
    """Reproduces the ORIGINAL residual-noise failure mechanism (identical to
    sampled_residual_falltime_check.py's run_and_locate_fall): searches for
    the first seed that falls within max_decisions, and reports exactly
    where."""
    for i in range(n_tries):
        seed = seed_start + i
        env = CaraResidualEnv(CaraResidualEnvConfig(n_steps=1, residual_bound_rad=bound))
        try:
            rng = np.random.default_rng(seed)
            obs, info = env.reset()
            term = trunc = False
            steps = 0
            while not (term or trunc) and steps < max_decisions:
                residual = rng.normal(0.0, std, size=env.n_act)
                obs, r, term, trunc, info = env.step(residual)
                steps += 1
            if term:
                return seed, steps, info
        finally:
            env.close()
    return None, None, None


def get_nominal_latch(n_steps=1):
    """Zero-residual reference: records the step's own natural latch via the
    SAME reference_offset_fn hook, as a pure no-op recorder (returns 0.0)."""
    latch = {}

    def record_only(ref_ey):
        latch.setdefault("value", ref_ey)
        return 0.0

    env = CaraResidualEnv(CaraResidualEnvConfig(n_steps=n_steps, reference_offset_fn=record_only))
    try:
        obs, info = env.reset()
        term = trunc = False
        while not (term or trunc):
            obs, r, term, trunc, info = env.step(np.zeros(env.n_act))
    finally:
        env.close()
    return latch["value"]


def run_with_residual_seed(seed, std, bound, max_decisions, reference_offset_fn=None, n_steps=1):
    env = CaraResidualEnv(CaraResidualEnvConfig(n_steps=n_steps, residual_bound_rad=bound,
                                                 reference_offset_fn=reference_offset_fn))
    try:
        rng = np.random.default_rng(seed)
        obs, info = env.reset()
        term = trunc = False
        steps = 0
        qpos_trace = [np.array(env._teacher.data.qpos, dtype=np.float64).copy()]
        tilt_trace = [info["tilt_deg"]]
        while not (term or trunc) and steps < max_decisions:
            residual = rng.normal(0.0, std, size=env.n_act)
            obs, r, term, trunc, info = env.step(residual)
            steps += 1
            qpos_trace.append(np.array(env._teacher.data.qpos, dtype=np.float64).copy())
            tilt_trace.append(info["tilt_deg"])
        return dict(fell=bool(term), fell_decision=steps if term else None, steps=steps,
                    qpos_trace=np.array(qpos_trace), tilt_trace=tilt_trace,
                    peak_tilt=max(tilt_trace), done_reason=info.get("done_reason"))
    finally:
        env.close()


def measure_kick_com_velocity(kick_qvel_y, kick_substep):
    """Addendum: directly measures the ACHIEVED change in whole-body CoM y-
    velocity from a data.qvel[1] += kick, instead of assuming it equals the
    kick. Uses the same one-time-kick technique as the earlier (superseded)
    causal-latch attempt, purely to characterize what that intervention
    actually did physically."""
    import mujoco
    import gait

    orig = mujoco.mj_step
    com_y_before = {}
    com_y_after = {}
    substep = {"n": 0}

    def w(model, data):
        i = substep["n"]
        if i == kick_substep - 1:
            com_y_before["pre_kick_com_y"] = float(data.subtree_com[0][1])
            com_y_before["pre_kick_qvel1"] = float(data.qvel[1])
        if i == kick_substep:
            data.qvel[1] += kick_qvel_y
            com_y_before["at_kick_com_y"] = float(data.subtree_com[0][1])
        orig(model, data)
        substep["n"] = i + 1
        if i == kick_substep:
            com_y_after["post_kick_com_y"] = float(data.subtree_com[0][1])
            com_y_after["post_kick_qvel1"] = float(data.qvel[1])

    mujoco.mj_step = w
    try:
        gait.run(gait.DEFAULT_CONFIG, 1, False, None, None)
    except Exception:
        pass
    finally:
        mujoco.mj_step = orig

    dt = DT
    v_com_y_before_kick = (com_y_before.get("at_kick_com_y", float("nan")) - com_y_before.get("pre_kick_com_y", float("nan"))) / dt
    v_com_y_after_kick = (com_y_after.get("post_kick_com_y", float("nan")) - com_y_before.get("at_kick_com_y", float("nan"))) / dt
    return dict(requested_kick_qvel1=kick_qvel_y,
                qvel1_before=com_y_before.get("pre_kick_qvel1"), qvel1_after=com_y_after.get("post_kick_qvel1"),
                achieved_qvel1_delta=com_y_after.get("post_kick_qvel1", float("nan")) - com_y_before.get("pre_kick_qvel1", float("nan")),
                com_y_velocity_estimate_before_kick_step=v_com_y_before_kick,
                com_y_velocity_estimate_after_kick_step=v_com_y_after_kick,
                note="finite-difference estimate of d(com_y)/dt across the single substep the kick was applied in; "
                     "compares the requested qvel[1] increment to the resulting whole-body CoM-y velocity change.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--std", type=float, default=0.0005)
    ap.add_argument("--bound", type=float, default=0.05)
    ap.add_argument("--max-decisions", type=int, default=350)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    print("=== finding a reproducible original residual-noise failure ===")
    seed, fell_decision, info = find_failing_seed(args.std, args.bound, args.max_decisions)
    if seed is None:
        print("  no seed in the tried range failed -- cannot run the causal comparison")
        return 1
    print(f"  seed={seed} fell at decision {fell_decision} ({fell_decision*10*DT:.2f}s), "
          f"reason={info.get('done_reason')}")

    print("\n=== nominal (zero-residual) latch for this step ===")
    ref_ey_nominal = get_nominal_latch(n_steps=1)
    print(f"  ref_ey_nominal = {ref_ey_nominal*1e3:+.3f} mm")

    print(f"\n=== Run A: seed={seed}, normal latch ===")
    run_a = run_with_residual_seed(seed, args.std, args.bound, args.max_decisions, reference_offset_fn=None)
    print(f"  fell={run_a['fell']} at decision {run_a['fell_decision']}  peak_tilt={run_a['peak_tilt']:.2f}deg "
          f"(40deg threshold; run truncated at {args.max_decisions} decisions if not fallen)")

    def substitute_nominal(ref_ey):
        return ref_ey_nominal - ref_ey

    print(f"\n=== Run B: seed={seed}, nominal latch substituted ===")
    run_b = run_with_residual_seed(seed, args.std, args.bound, args.max_decisions,
                                    reference_offset_fn=substitute_nominal)
    print(f"  fell={run_b['fell']} at decision {run_b['fell_decision']}  peak_tilt={run_b['peak_tilt']:.2f}deg")

    n = min(len(run_a["qpos_trace"]), len(run_b["qpos_trace"]))
    # everything up to (not including) the latch decision must match bit-for-bit
    b_onset_decision = 275  # step_idx=0 phase-B onset, decisions (see phase_schedule/phase boundary table)
    pre_latch_n = min(n, b_onset_decision + 1)
    pre_latch_diff = float(np.max(np.abs(run_a["qpos_trace"][:pre_latch_n] - run_b["qpos_trace"][:pre_latch_n])))
    print(f"\n  sanity: runs A and B identical through decision {pre_latch_n-1} "
          f"(pre-latch + latch instant)? max diff = {pre_latch_diff:.3e} "
          f"({'OK' if pre_latch_diff == 0.0 else 'UNEXPECTED -- investigate'})")

    if not run_a["fell"]:
        verdict = "Run A (normal latch) did not fail within the tested window -- cannot evaluate the causal claim with this seed/window."
    elif run_b["fell"] and run_b["fell_decision"] is not None and run_a["fell_decision"] is not None \
            and run_b["fell_decision"] <= run_a["fell_decision"] + 5:
        verdict = "BOTH FAIL at essentially the same point -- latch correction alone is insufficient; the biased latch is not the (sole) cause."
    elif run_b["fell"]:
        verdict = (f"Run B fails LATER than Run A (decision {run_b['fell_decision']} vs {run_a['fell_decision']}) "
                   f"but still fails -- LIMITED effect: the latch substitution delays but does not prevent the failure.")
    else:
        verdict = "Run B (nominal latch substituted) SURVIVES the window where Run A fails -- the latch value materially contributes to the failure."
    print(f"\n  interpretation: {verdict}")

    print("\n=== addendum: what did the earlier kick intervention actually do to CoM velocity? ===")
    kick_check = measure_kick_com_velocity(kick_qvel_y=0.008, kick_substep=2730)
    for k, v in kick_check.items():
        print(f"  {k}: {v}")

    out = dict(seed=seed, run_a=dict(fell=run_a["fell"], fell_decision=run_a["fell_decision"], peak_tilt=run_a["peak_tilt"]),
               run_b=dict(fell=run_b["fell"], fell_decision=run_b["fell_decision"], peak_tilt=run_b["peak_tilt"]),
               ref_ey_nominal_mm=ref_ey_nominal * 1e3, pre_latch_bitexact=(pre_latch_diff == 0.0),
               verdict=verdict, kick_velocity_check=kick_check)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
