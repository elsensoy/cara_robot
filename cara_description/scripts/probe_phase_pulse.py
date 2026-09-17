#!/usr/bin/env python3
"""U35 prep -- brief, phase-indexed residual pulses, replacing the U34
constant-bias sweep. That sweep's asymmetry (negative ankle-roll bias
failing even at -0.001 rad) was confounded: joint-sign convention, WHICH
leg, and the phase when the bias began were never isolated from "opposes
the alternating trim."

Design: for each ankle-roll channel, tested at the decision window where
THAT foot is actually the stance/trimmed foot (from gait.py's own a_roll
convention: a_roll = stance_leg + 'ankle_roll'), apply a brief pulse
(smoothstep on, hold, smoothstep off) of EQUAL magnitude in both signs,
then return exactly to zero residual for the rest of the episode. Teacher
targets and actuator limits are completely unchanged -- only the residual
pulse is varied.

Records: immediate tilt response (peak tilt during and shortly after the
pulse), whether normal touchdown/support resumes afterward, and whether
the full 2-step walk still completes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cara_residual_env import CaraResidualEnv, CaraResidualEnvConfig, phase_at

# (joint, step_idx) pairs where that joint is the ACTUAL stance/trimmed
# ankle for that step, per gait.py's own convention (a_roll = stance+"ankle_roll"):
#   step 0: lead=l_, stance=r_  -> r_ankle_roll is trimmed
#   step 1: lead=r_, stance=l_  -> l_ankle_roll is trimmed
STANCE_WINDOWS = {
    "r_ankle_roll": dict(step_idx=0, phase="C"),
    "l_ankle_roll": dict(step_idx=1, phase="C"),
}


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 3 * u ** 2 - 2 * u ** 3


def pulse_profile(center_decision, half_width, ramp):
    """Returns a function f(decision_idx) -> [0,1] pulse envelope: smoothstep
    on over `ramp` decisions, held at 1.0 for the middle, smoothstep off over
    `ramp` decisions, zero elsewhere."""
    on_start = center_decision - half_width - ramp
    on_end = center_decision - half_width
    off_start = center_decision + half_width
    off_end = center_decision + half_width + ramp

    def f(k):
        if k < on_start or k >= off_end:
            return 0.0
        if k < on_end:
            return smoothstep((k - on_start) / ramp)
        if k < off_start:
            return 1.0
        return 1.0 - smoothstep((k - off_start) / ramp)
    return f, on_start, off_end


def run_pulse(n_steps, joint, magnitude, half_width=10, ramp=10):
    # CaraResidualEnv.step() clips the incoming residual to cfg.residual_bound_rad
    # (default 0.02) -- that's correct behavior for a trained policy, but this
    # probe's whole purpose is to test magnitudes UP TO and BEYOND typical
    # bounds, so the env's own bound must be raised to accommodate whatever is
    # being tested here. An earlier version of this script left the default in
    # place, silently clipping every requested magnitude above 0.02 down to
    # 0.02 -- caught because l_ankle_roll and r_ankle_roll pulses showed
    # near-identical, saturated effects across 0.02/0.05/0.1/0.15 rad instead
    # of the expected monotonic response.
    bound = max(0.3, abs(magnitude) * 1.5)
    env = CaraResidualEnv(CaraResidualEnvConfig(n_steps=n_steps, residual_bound_rad=bound))
    ji = env.jn.index(joint)
    win = STANCE_WINDOWS[joint]
    for start, end, name, lead, step_idx in env.schedule:
        if step_idx == win["step_idx"] and name == win["phase"]:
            center = (start // 10 + end // 10) // 2
            break
    else:
        raise ValueError("stance window not found in schedule")
    envelope, on_start, off_end = pulse_profile(center, half_width, ramp)

    obs, info = env.reset()
    x0 = info["pelvis_x"]
    peak_tilt = info["tilt_deg"]
    peak_tilt_during_window = 0.0
    touch_key = "l_" if joint.startswith("l_") else "r_"
    touch_after_pulse = []
    steps = 0
    k = 0
    term = trunc = False
    while not (term or trunc):
        residual = np.zeros(env.n_act)
        residual[ji] = magnitude * envelope(k)
        obs, r, term, trunc, info = env.step(residual)
        steps += 1
        k += 1
        peak_tilt = max(peak_tilt, info["tilt_deg"])
        if on_start <= k <= off_end + half_width:
            peak_tilt_during_window = max(peak_tilt_during_window, info["tilt_deg"])
        if off_end <= k <= off_end + 20:  # ~0.4s after the pulse fully returns to zero
            touch_after_pulse.append(info["foot_touch"][touch_key])
    resumed_support = any(touch_after_pulse) if touch_after_pulse else None
    return dict(steps=steps, total=env.total_decisions, completed=trunc,
                peak_tilt_deg=peak_tilt, peak_tilt_during_window_deg=peak_tilt_during_window,
                fwd_dist_m=info["pelvis_x"] - x0, resumed_support_after_pulse=resumed_support,
                pulse_center_decision=center, fell_reason=info.get("done_reason"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-steps", type=int, default=2)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    print("=== zero-residual reference (for peak-tilt comparison) ===")
    ref = run_pulse(args.n_steps, "l_ankle_roll", 0.0)
    print(f"  completed={ref['completed']} peak_tilt={ref['peak_tilt_deg']:.2f}deg fwd={ref['fwd_dist_m']:+.4f}m")

    magnitudes = [0.01, 0.02, 0.05, 0.1, 0.15]
    results = {"reference": ref}
    for joint in STANCE_WINDOWS:
        for mag in magnitudes:
            for sign, label in ((+1, "+"), (-1, "-")):
                m = sign * mag
                key = f"{joint}_{label}{mag}"
                print(f"\n=== pulse {label}{mag} rad on {joint} at its own stance-C window ===")
                r = run_pulse(args.n_steps, joint, m)
                print(f"  completed={r['completed']}  peak_tilt={r['peak_tilt_deg']:.2f}deg "
                      f"(during-window={r['peak_tilt_during_window_deg']:.2f})  "
                      f"resumed_support={r['resumed_support_after_pulse']}  fwd={r['fwd_dist_m']:+.4f}m"
                      + ("" if r["completed"] else f"  FELL: {r['fell_reason']}"))
                results[key] = r

    print("\n=== summary: largest magnitude where BOTH signs completed AND resumed support ===")
    chosen = {}
    for joint in STANCE_WINDOWS:
        ok_mags = [mag for mag in magnitudes
                   if results[f"{joint}_+{mag}"]["completed"] and results[f"{joint}_+{mag}"]["resumed_support_after_pulse"]
                   and results[f"{joint}_-{mag}"]["completed"] and results[f"{joint}_-{mag}"]["resumed_support_after_pulse"]]
        chosen[joint] = max(ok_mags) if ok_mags else None
        print(f"  {joint}: bidirectionally OK at {ok_mags} rad" if ok_mags else f"  {joint}: none bidirectionally OK")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(dict(results=results, chosen=chosen), f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
