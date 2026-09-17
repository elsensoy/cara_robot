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
    """Addendum: directly measures the change in whole-body CoM y-velocity
    from a data.qvel[1] += kick, instead of assuming it equals the kick.

    FIXED (was wrong): the original version read data.subtree_com
    immediately before/after mj_step(model, data). MuJoCo's mj_step computes
    all derived/kinematic quantities (including subtree_com) from the qpos
    AS IT STANDS AT THE START of that call (mj_step1/mj_fwdPosition), THEN
    integrates qpos/qvel forward (mj_step2) -- so data.subtree_com read
    right after mj_step returns still reflects the PRE-integration
    configuration, not the state resulting from that step's own physics.
    Reading it again one substep later is comparing kinematics that are
    perpetually one mj_step call behind data.qpos/qvel. That finite
    difference measured motion preceding the kick, not its effect.

    Correct approach (no physics advanced): compute the instantaneous
    v_com = J_com(q) . qdot via mj_jacSubtreeCom, evaluated at the SAME q
    before and after modifying qvel. mj_kinematics + mj_comPos are called
    explicitly first to refresh xpos/subtree_com for the qpos CURRENTLY in
    data (they too would otherwise lag by one mj_step call), so the
    Jacobian is evaluated at a configuration consistent with the qvel it is
    dotted against."""
    import mujoco
    import gait

    result = {}
    orig = mujoco.mj_step
    substep = {"n": 0}

    def w(model, data):
        i = substep["n"]
        if i == kick_substep:
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSubtreeCom(model, data, jacp, 0)  # body 0 = worldbody; subtree_com[0] = whole robot
            v_com_before = jacp @ data.qvel
            qvel1_before = float(data.qvel[1])

            data.qvel[1] += kick_qvel_y  # same q, so the SAME jacp still applies exactly -- no re-evaluation needed

            v_com_after = jacp @ data.qvel
            qvel1_after = float(data.qvel[1])

            result["v_com_y_before"] = float(v_com_before[1])
            result["v_com_y_after"] = float(v_com_after[1])
            result["v_com_y_delta"] = float(v_com_after[1] - v_com_before[1])
            result["qvel1_before"] = qvel1_before
            result["qvel1_after"] = qvel1_after
            result["qvel1_delta"] = qvel1_after - qvel1_before
        orig(model, data)
        substep["n"] = i + 1

    mujoco.mj_step = w
    try:
        gait.run(gait.DEFAULT_CONFIG, 1, False, None, None)
    finally:
        mujoco.mj_step = orig

    if not result:
        raise RuntimeError(f"kick_substep={kick_substep} was never reached (n_steps=1 run ended earlier than expected) "
                            f"-- not silently returning an incomplete measurement.")

    result["requested_kick_qvel1"] = kick_qvel_y
    result["note"] = ("instantaneous v_com = J_com(q).qdot via mj_jacSubtreeCom, evaluated before/after "
                       "modifying qvel[1] at the SAME q (no physics advanced) -- not a cross-substep finite difference.")
    return result


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

    # This nominal-latch value is specific to n_steps=1, step_idx=0 (the only step this
    # script ever simulates) -- it is NOT a general-purpose "the nominal latch" constant
    # and would need recomputing for any other step_idx/n_steps configuration.
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

    # NOTE ON SCOPE: this compares qpos sampled once per DECISION (every 10 substeps,
    # i.e. 50Hz) -- it verifies equivalence AT those sampling points, not at every one
    # of the 2750 physics substeps (500Hz) leading up to the latch. It is not a full
    # substep-resolution state/controller-history equivalence check (that would need
    # per-substep instrumentation inside _TeacherThread, which does not currently
    # exist). What DOES support substep-level equivalence, by construction rather than
    # by this check alone: reference_offset_fn is only ever invoked from inside
    # ss_step, which gait.py never calls during phase A or the settle loop -- so up
    # through decision 275 (phase-B onset), Run A and Run B execute the IDENTICAL code
    # path regardless of reference_offset_fn's value, and with the same seed/RNG draws
    # and deterministic physics, they cannot differ. The decision-level check below is
    # confirming that construction, not the sole evidence for it.
    n = min(len(run_a["qpos_trace"]), len(run_b["qpos_trace"]))
    b_onset_decision = 275  # step_idx=0 phase-B onset, decisions (see phase_schedule/phase boundary table)
    pre_latch_n = min(n, b_onset_decision + 1)
    pre_latch_diff = float(np.max(np.abs(run_a["qpos_trace"][:pre_latch_n] - run_b["qpos_trace"][:pre_latch_n])))
    print(f"\n  sanity: runs A and B identical through decision {pre_latch_n-1} "
          f"(pre-latch + latch instant), AT 50Hz DECISION-LEVEL SAMPLES? max diff = {pre_latch_diff:.3e} "
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
