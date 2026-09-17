#!/usr/bin/env python3
"""U34 milestone 2 -- a few physically interpretable, constant residual
corrections, to choose residual_bound_rad from measured effect rather than
copying CaraWalkEnv's action_range_frac=0.30 (a DIFFERENT quantity: a
fraction of full joint range around nominal, not a correction magnitude
around a moving teacher target).

Applies a CONSTANT residual (same value every decision, for the whole
episode) to one joint pair at a time, on top of the unmodified teacher,
and reports survival, peak tilt, and forward distance relative to the
zero-residual reference -- looking for "predictable effects" (a small,
roughly monotonic response) rather than either negligible or destabilizing
ones.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cara_residual_env import CaraResidualEnv, CaraResidualEnvConfig


def run_constant_residual(n_steps, joint_pair, magnitude_rad):
    env = CaraResidualEnv(CaraResidualEnvConfig(n_steps=n_steps))
    idx = [env.jn.index(j) for j in joint_pair]
    obs, info = env.reset()
    x0 = info["pelvis_x"]
    peak_tilt = info["tilt_deg"]
    steps = 0
    term = trunc = False
    while not (term or trunc):
        residual = np.zeros(env.n_act)
        for i in idx:
            residual[i] = magnitude_rad
        obs, r, term, trunc, info = env.step(residual)
        steps += 1
        peak_tilt = max(peak_tilt, info["tilt_deg"])
    return dict(steps=steps, total=env.total_decisions, survived=trunc, peak_tilt_deg=peak_tilt,
                fwd_dist_m=info["pelvis_x"] - x0, fell_reason=info.get("done_reason"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-steps", type=int, default=2)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    print("=== zero-residual reference ===")
    ref = run_constant_residual(args.n_steps, [], 0.0)
    print(f"  steps={ref['steps']}/{ref['total']} survived={ref['survived']} "
          f"peak_tilt={ref['peak_tilt_deg']:.2f}deg fwd_dist={ref['fwd_dist_m']:+.4f}m")

    joint_pairs = {
        "ankle_roll": ["l_ankle_roll", "r_ankle_roll"],
        "hip_roll": ["l_hip_roll", "r_hip_roll"],
    }
    magnitudes = [0.005, 0.01, 0.02, 0.05]

    results = {"reference": ref}
    for name, pair in joint_pairs.items():
        for mag in magnitudes:
            for sign, label in ((+1, "+"), (-1, "-")):
                m = sign * mag
                key = f"{name}_{label}{mag}"
                print(f"\n=== residual {label}{mag} rad on {pair} (constant, whole episode) ===")
                r = run_constant_residual(args.n_steps, pair, m)
                d_tilt = r["peak_tilt_deg"] - ref["peak_tilt_deg"]
                d_fwd = r["fwd_dist_m"] - ref["fwd_dist_m"]
                print(f"  steps={r['steps']}/{r['total']} survived={r['survived']} "
                      f"peak_tilt={r['peak_tilt_deg']:.2f}deg ({d_tilt:+.2f} vs ref)  "
                      f"fwd_dist={r['fwd_dist_m']:+.4f}m ({d_fwd:+.4f} vs ref)"
                      + ("" if r["survived"] else f"  FELL: {r['fell_reason']}"))
                results[key] = r

    print("\n=== summary: largest magnitude that survived, per joint pair ===")
    chosen = {}
    for name in joint_pairs:
        survived_mags = [mag for mag in magnitudes
                          if results[f"{name}_+{mag}"]["survived"] and results[f"{name}_-{mag}"]["survived"]]
        largest_ok = max(survived_mags) if survived_mags else None
        print(f"  {name}: survived at +/-{survived_mags} rad" if survived_mags else f"  {name}: none survived")
        chosen[name] = largest_ok

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(dict(results=results, chosen_largest_surviving_magnitude=chosen), f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
