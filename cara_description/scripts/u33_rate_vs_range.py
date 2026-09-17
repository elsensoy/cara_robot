#!/usr/bin/env python3
"""U33 -- separate RANGE from UPDATE RATE in the U32 teacher-replay failure.

U32 downsampled a single 500Hz-native, unbounded recorded trajectory to
50Hz and clipped it -- frequency and clipping changed TOGETHER, so their
individual effects were confounded. This script runs gait.py's OWN control
law (unmodified source, monkeypatched mujoco.mj_step, same technique as
U32) ONLINE at each of 4 conditions, closed-loop, against the ACTUAL live
state at each condition -- never replaying commands recorded on a
different state trajectory, since a fall in one condition changes the
state the others would see next.

    rate   | bounds           | isolates
    -------|------------------|---------------------------------
    500 Hz | native (joint/actuator limits only) | reference under replay conditions
    500 Hz | current RL bounds (+/-1 normalized)  | effect of restricted authority alone
    50 Hz  | native                                | effect of slower updates alone
    50 Hz  | current RL bounds                     | reproduces U32

Mechanism: gait.py computes a fresh data.ctrl[:] target from LIVE state
before every one of its own mj_step() calls (its native 500Hz cadence).
The monkeypatch intercepts each call: at "decision boundaries" (every Nth
call, N=1 for 500Hz or N=10 for 50Hz) it lets gait.py's freshly-computed
target through (after applying this condition's bounds), and RECORDS it
as the held target; on intermediate calls (50Hz conditions only) it
OVERWRITES data.ctrl with that held target before the real mj_step runs --
so gait.py still computes on schedule, but the actuators only move at the
tested decision rate, exactly like the underlying position-PD servo
tracking a target that CaraWalkEnv would hand it once per control step.

Termination (CaraWalkEnv's own 40deg / 0.15m rule) is checked at every
decision boundary and stops the replay via a raised (and caught) sentinel
exception -- gait.py's own source has no notion of failing early.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from cara_env import CaraWalkEnv, CaraWalkEnvConfig

DEFAULT_CONFIG = "../config/cara_full_body.yaml"
NATIVE_HZ = 500  # 1/model.opt.timestep = 1/0.002


class _Fell(Exception):
    def __init__(self, decision_step, tilt_deg, height_m, reason):
        self.decision_step = decision_step
        self.tilt_deg = tilt_deg
        self.height_m = height_m
        self.reason = reason


def make_clipper(mode, jn_index, nominal, offset_scale, lo, hi):
    """Returns f(target_vec) -> clipped_target_vec, for one of the two
    bounds conditions. 'native' clips only to the model's own joint
    limits. 'rl' additionally passes through CaraWalkEnv's own
    normalize -> clip[-1,1] -> denormalize pipeline, exactly matching what
    CaraWalkEnv.step() does internally."""
    if mode == "native":
        def f(target):
            return np.clip(target, lo, hi)
    elif mode == "rl":
        def f(target):
            a = (target - nominal) / offset_scale
            a = np.clip(a, -1.0, 1.0)
            clipped_target = nominal + a * offset_scale
            return np.clip(clipped_target, lo, hi)
    else:
        raise ValueError(mode)
    return f


def run_condition(config_path, n_steps, rate_hz, bounds_mode, jn, nominal, offset_scale, lo, hi,
                   fall_tilt_deg, fall_height_m):
    import mujoco
    original_mj_step = mujoco.mj_step
    clip_fn = make_clipper(bounds_mode, None, nominal, offset_scale, lo, hi)
    ratio = round(NATIVE_HZ / rate_hz)

    state = dict(substep=0, held_target=None, decision_step=0,
                 pelvis_x0=None, min_x=1e9, max_x=-1e9)
    log = dict(pelvis_x=[], tilt_deg=[], decision_step_of_substep=[])

    def wrapped_mj_step(model, data):
        i = state["substep"]
        if i % ratio == 0:
            # gait.py has just written its freshly-computed, LIVE-state
            # target into data.ctrl -- clip it per this condition's bounds
            # and hold it for the next `ratio` substeps.
            raw_target = np.array(data.ctrl, dtype=np.float64).copy()
            held = clip_fn(raw_target)
            state["held_target"] = held
            state["decision_step"] += 1
        data.ctrl[:] = state["held_target"]
        original_mj_step(model, data)
        state["substep"] = i + 1

        if state["pelvis_x0"] is None:
            state["pelvis_x0"] = float(data.qpos[0])
        # Check termination once per decision boundary (matches how
        # CaraWalkEnv itself only ever evaluates it once per control step).
        if i % ratio == ratio - 1:
            roll = math.atan2(2 * (data.qpos[3] * data.qpos[4] + data.qpos[5] * data.qpos[6]),
                               1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2))
            sinp = 2 * (data.qpos[3] * data.qpos[5] - data.qpos[6] * data.qpos[4])
            sinp = max(-1.0, min(1.0, sinp))
            pitch = math.asin(sinp)
            tilt_deg = math.degrees(max(abs(roll), abs(pitch)))
            height = float(data.qpos[2])
            log["pelvis_x"].append(float(data.qpos[0]))
            log["tilt_deg"].append(tilt_deg)
            if tilt_deg > fall_tilt_deg or height < fall_height_m or not np.all(np.isfinite(data.qpos)):
                raise _Fell(state["decision_step"], tilt_deg, height,
                            "tilt" if tilt_deg > fall_tilt_deg else
                            ("height" if height < fall_height_m else "nonfinite"))

    mujoco.mj_step = wrapped_mj_step
    try:
        import gait
        try:
            gait.run(config_path, n_steps, view=False, json_path=None, baseline_path=None)
            fell = None
        except _Fell as e:
            fell = e
    finally:
        mujoco.mj_step = original_mj_step

    total_decision_steps = state["decision_step"]
    fwd_dist = (log["pelvis_x"][-1] - log["pelvis_x"][0]) if log["pelvis_x"] else 0.0
    peak_tilt = max(log["tilt_deg"]) if log["tilt_deg"] else 0.0
    return dict(
        survived=fell is None,
        fell_at_decision_step=fell.decision_step if fell else None,
        fell_at_time_s=(fell.decision_step / rate_hz) if fell else None,
        fell_reason=fell.reason if fell else None,
        fell_tilt_deg=fell.tilt_deg if fell else None,
        total_decision_steps=total_decision_steps,
        peak_tilt_deg=peak_tilt,
        fwd_dist_m=fwd_dist,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--n-steps", type=int, default=2)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    probe = CaraWalkEnv(CaraWalkEnvConfig(episode_seconds=1.0))
    jn, nominal, offset_scale = probe.jn, probe.nominal, probe.offset_scale
    lo, hi = probe.lo, probe.hi
    fall_tilt_deg = probe.cfg.fall_tilt_deg
    fall_height_m = probe.cfg.fall_height_m

    conditions = [
        (500, "native", "500Hz / native teacher targets (reference under replay conditions)"),
        (500, "rl", "500Hz / current RL bounds (isolates restricted authority)"),
        (50, "native", "50Hz / native teacher targets (isolates slower updates)"),
        (50, "rl", "50Hz / current RL bounds (reproduces U32)"),
    ]

    results = {}
    for rate_hz, bounds_mode, label in conditions:
        print(f"\n=== {label} ===")
        r = run_condition(args.config, args.n_steps, rate_hz, bounds_mode, jn, nominal, offset_scale,
                           lo, hi, fall_tilt_deg, fall_height_m)
        if r["survived"]:
            print(f"  SURVIVED all {r['total_decision_steps']} decision steps "
                  f"({r['total_decision_steps']/rate_hz:.1f}s)  peak_tilt={r['peak_tilt_deg']:.1f}deg  "
                  f"fwd_dist={r['fwd_dist_m']:+.3f}m")
        else:
            print(f"  FELL at decision step {r['fell_at_decision_step']} "
                  f"({r['fell_at_time_s']:.1f}s), reason={r['fell_reason']}, tilt={r['fell_tilt_deg']:.1f}deg")
        results[f"{rate_hz}Hz_{bounds_mode}"] = dict(rate_hz=rate_hz, bounds_mode=bounds_mode, label=label, **r)

    ref = results["500Hz_native"]
    rl_only = results["500Hz_rl"]
    rate_only = results["50Hz_native"]
    both = results["50Hz_rl"]

    print("\n=== decision readout ===")
    if not ref["survived"]:
        print("  Reference (500Hz/native) FAILS under the replay's own conditions -- reconcile these "
              "conditions with gait.py's own validated run before drawing any interface conclusion. "
              f"(gait.py's own run() reported its milestone separately -- see stdout above for its own PASS/FAIL.)")
    elif rate_only["survived"] and not rl_only["survived"]:
        print("  50Hz/native SUCCEEDS, 500Hz/RL-bounds FAILS: RANGE is a demonstrated blocker for this "
              "teacher. Next: a joint-specific ankle-roll expansion, sized from its required targets, "
              "with explicit joint-limit checks.")
    elif rl_only["survived"] and not rate_only["survived"]:
        print("  500Hz/RL-bounds SUCCEEDS, 50Hz/native FAILS: UPDATE RATE is the stronger lead.")
    elif not rate_only["survived"] and not rl_only["survived"]:
        print("  BOTH isolated changes cause failure: this teacher depends on both range AND rate -- "
              "widening ankle range alone will not transfer it.")
    else:
        print("  Both isolated changes (500Hz/RL-bounds, 50Hz/native) SUCCEED -- neither range nor rate "
              "alone explains U32's failure; the combination (both, i.e. 50Hz/RL-bounds) may still fail "
              "from their interaction. See the 50Hz/RL-bounds ('reproduces U32') row above.")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
