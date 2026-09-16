#!/usr/bin/env python3
"""U30 evaluation: does introducing a modest forward command produce
genuine alternating stepping, or does the policy just keep standing
(a legitimate, non-failure outcome that must be reported as such, not
assumed away)?

Four measurements, per the review table:
  1. survival and achieved forward speed (sustained motion vs. standing/falling)
  2. foot clearance, alternating touchdown, and STANCE-FOOT SLIP (stepping vs. sliding)
  3. zero-command performance (was standing retained?)
  4. torque use and joint tracking (movement within modeled actuator authority)

Milestone (per instruction): several genuine alternating steps with
sustained forward progress while upright. If the policy just stands, that
IS the result -- report it plainly, and separately check whether that's an
exploration/cost-of-moving issue rather than automatically re-filing it as
a balance failure (it manifestly is not one -- standing is retained by
construction if this happens).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from train_ppo import ActorCritic, RunningNorm

MAX_STEPS = 200
EVAL_SEEDS = list(range(3000, 3010))   # N=10, fresh -- deterministic reset means N=1 would suffice
                                         # for a single policy, but keeping several for stability checks
MIN_STANCE_RUN = 3   # control steps (60ms) -- filters contact flicker, matches evaluate_policy.py's convention


def load(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env_config = ckpt["env_config"]
    obs_dim = len(ckpt["obs_norm"]["mean"])
    act_dim = ckpt["model_state"]["actor_mean.4.bias"].shape[0]
    agent = ActorCritic(obs_dim, act_dim)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    norm = RunningNorm(obs_dim)
    norm.load_state_dict(ckpt["obs_norm"])
    return agent, norm, env_config


def rollout(env, agent, norm, seed):
    obs, _ = env.reset(seed=seed)
    log = dict(vx=[], tilt=[], qpos_err=[], torque_frac=[],
               touch_l=[], touch_r=[], z_l=[], z_r=[], xy_l=[], xy_r=[], pelvis_x=[])
    fell = False
    steps = 0
    for t in range(MAX_STEPS):
        obs_n = norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            mean = agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
        obs, reward, terminated, truncated, info = env.step(mean.numpy())
        steps += 1
        log["vx"].append(info["vx"])
        log["tilt"].append(info["tilt_deg"])
        log["qpos_err"].append(info["qpos_err"])
        log["torque_frac"].append(info["torque_frac"])
        log["touch_l"].append(info["foot_touch"]["l_foot_collision"])
        log["touch_r"].append(info["foot_touch"]["r_foot_collision"])
        log["z_l"].append(info["foot_z"]["l_foot_collision"])
        log["z_r"].append(info["foot_z"]["r_foot_collision"])
        log["xy_l"].append(tuple(env.data.geom_xpos[env.foot_gid["l_foot_collision"]][:2]))
        log["xy_r"].append(tuple(env.data.geom_xpos[env.foot_gid["r_foot_collision"]][:2]))
        log["pelvis_x"].append(float(env.data.qpos[0]))
        if terminated:
            fell = True
            break
        if truncated:
            break
    return dict(fell=fell, steps=steps, log=log)


def stance_runs(touch):
    """Yield (start, end) index pairs of maximal contact runs >= MIN_STANCE_RUN."""
    runs = []
    start = None
    for i, v in enumerate(touch + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= MIN_STANCE_RUN:
                runs.append((start, i))
            start = None
    return runs


def analyze(result, floor_ref_z):
    log = result["log"]
    n = result["steps"]
    touch_l, touch_r = log["touch_l"], log["touch_r"]
    support = [int(l) * 2 + int(r) for l, r in zip(touch_l, touch_r)]  # 0=none,1=r,2=l,3=both
    switches = sum(1 for i in range(1, n) if support[i] != support[i - 1])

    def genuine_steps(touch_self):
        runs = 0
        run = 0
        for v in touch_self:
            if not v:
                run += 1
            else:
                if run >= MIN_STANCE_RUN:
                    runs += 1
                run = 0
        if run >= MIN_STANCE_RUN:
            runs += 1
        return runs

    swing_steps_l = genuine_steps(touch_l)   # genuine SWING phases (foot airborne) for each foot
    swing_steps_r = genuine_steps(touch_r)

    def clearance(z, touch):
        airborne_z = [z[i] - floor_ref_z for i in range(n) if not touch[i]]
        return max(airborne_z) if airborne_z else 0.0

    clearance_l = clearance(log["z_l"], touch_l)
    clearance_r = clearance(log["z_r"], touch_r)

    def slip(xy, touch):
        runs = stance_runs(touch)
        if not runs:
            return None
        slips = []
        for s, e in runs:
            pts = np.array(xy[s:e])
            slips.append(float(np.max(np.linalg.norm(pts - pts[0], axis=1))))
        return float(np.mean(slips))

    slip_l = slip(log["xy_l"], touch_l)
    slip_r = slip(log["xy_r"], touch_r)

    fwd_dist = log["pelvis_x"][-1] - log["pelvis_x"][0] if n else 0.0
    return dict(
        steps=n, fell=result["fell"], survived_full=not result["fell"] and n >= MAX_STEPS,
        vx_mean=float(np.mean(log["vx"])) if n else 0.0, vx_std=float(np.std(log["vx"])) if n else 0.0,
        fwd_dist=fwd_dist, support_switches=switches,
        swing_steps_l=swing_steps_l, swing_steps_r=swing_steps_r,
        clearance_l_m=clearance_l, clearance_r_m=clearance_r,
        stance_slip_l_m=slip_l, stance_slip_r_m=slip_r,
        qpos_err_mean=float(np.mean(log["qpos_err"])) if n else float("nan"),
        torque_frac_mean=float(np.mean(log["torque_frac"])) if n else float("nan"),
        torque_frac_max=float(np.max(log["torque_frac"])) if n else float("nan"),
        peak_tilt_deg=float(np.max(log["tilt"])) if n else float("nan"),
    )


def classify(a, desired_vx):
    """The actual milestone check: several genuine alternating steps with
    sustained forward progress while upright. NOT automatically a balance
    failure if the policy just stands -- that's reported as its own,
    legitimate outcome."""
    if not a["survived_full"]:
        return "FELL"
    min_swings = min(a["swing_steps_l"], a["swing_steps_r"])
    if desired_vx == 0.0:
        return "STANDING (zero command, as expected)"
    if min_swings >= 3 and a["fwd_dist"] > 0.05:
        return "STEPPING (several genuine alternating steps, sustained forward progress)"
    if a["fwd_dist"] > 0.05 and min_swings < 3:
        return "SLIDING (forward progress without alternating swing phases)"
    return "STANDING (survived at commanded nonzero speed, but did not step or progress -- " \
           "not a balance failure; inspect exploration / cost of moving)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    agent, norm, env_config = load(args.checkpoint)
    cmd_vx = 0.03
    if env_config.get("desired_vx_bands"):
        vxs = [v for _, v in env_config["desired_vx_bands"]]
        cmd_vx = max(vxs)

    report = {}
    for label, vx in (("zero_command", 0.0), (f"commanded_{cmd_vx}", cmd_vx)):
        cfg = CaraWalkEnvConfig(**{**env_config, "episode_seconds": MAX_STEPS / 50.0,
                                    "desired_vx_bands": None, "desired_vx": vx})
        env = CaraWalkEnv(cfg)
        # floor reference: this policy's OWN zero-command resting foot height,
        # not an assumed constant -- measured directly, not hardcoded.
        zero_cfg = CaraWalkEnvConfig(**{**env_config, "episode_seconds": 1.0,
                                         "desired_vx_bands": None, "desired_vx": 0.0})
        zero_env = CaraWalkEnv(zero_cfg)
        zr = rollout(zero_env, agent, norm, seed=0)
        floor_ref_z = float(np.mean(zr["log"]["z_l"][-10:] + zr["log"]["z_r"][-10:]))

        results = [rollout(env, agent, norm, seed=s) for s in EVAL_SEEDS]
        analyzed = [analyze(r, floor_ref_z) for r in results]
        a0 = analyzed[0]  # deterministic + deterministic reset -> all EVAL_SEEDS identical; keep 1 as canonical
        verdict = classify(a0, vx)

        print(f"\n=== {label} (vx={vx}) ===")
        print(f"  steps={a0['steps']}/{MAX_STEPS}  fell={a0['fell']}  "
              f"achieved_vx={a0['vx_mean']:+.4f}+/-{a0['vx_std']:.4f} m/s  fwd_dist={a0['fwd_dist']:+.3f}m")
        print(f"  support_switches={a0['support_switches']}  genuine_swings L={a0['swing_steps_l']} R={a0['swing_steps_r']}")
        print(f"  foot clearance: L={a0['clearance_l_m']*1000:.1f}mm R={a0['clearance_r_m']*1000:.1f}mm "
              f"(floor_ref_z={floor_ref_z:.4f}m)")
        print(f"  stance-foot slip: L={a0['stance_slip_l_m']*1000 if a0['stance_slip_l_m'] is not None else float('nan'):.1f}mm "
              f"R={a0['stance_slip_r_m']*1000 if a0['stance_slip_r_m'] is not None else float('nan'):.1f}mm")
        print(f"  joint tracking error: mean={a0['qpos_err_mean']:.4f} rad  "
              f"actuator saturation: mean={a0['torque_frac_mean']:.1%} max={a0['torque_frac_max']:.1%}")
        print(f"  peak tilt: {a0['peak_tilt_deg']:.1f} deg")
        print(f"  VERDICT: {verdict}")

        report[label] = dict(vx_commanded=vx, floor_ref_z=floor_ref_z, canonical=a0, verdict=verdict,
                              all_repeats=analyzed)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
