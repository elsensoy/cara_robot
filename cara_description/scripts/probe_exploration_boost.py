#!/usr/bin/env python3
"""U31 pre-training probe: does boosting exploration std on moving-command
episodes produce anything USEFUL, before spending a training budget on it?

For each candidate multiplier (1.5x, 2x, 3x over the U30 policy's own
LEARNED std), runs short SAMPLED rollouts at a fixed 0.03 m/s command,
using the exact same command-dependent-std mechanism ActorCritic.act()
uses during real training (same code path -- not a separate noise source).
Zero-command exploration is never touched (the boost only ever applies
when the desired_vx observation is nonzero).

Looked for, per instruction -- NOT just "does it fall less":
  - longer unloading (airborne) intervals than U30's noise-floor baseline
  - clearance clearly above that noise floor (U30 measured 1-3mm at every
    seed/condition -- that's vibration, not a lift)
  - a genuine SUPPORTED touchdown after the airborne interval (not just an
    airborne blip followed immediately by falling)
  - rejected if the setting mainly produces immediate falls, heavy action
    clipping, or jitter without any of the above
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from train_ppo import ActorCritic, RunningNorm

MAX_STEPS = 200
N_REPEATS = 20
MULTIPLIERS = [1.5, 2.0, 3.0]
MIN_RUN = 3            # control steps (60ms), filters flicker -- matches project convention
NOISE_FLOOR_M = 0.003  # 3mm, from U30's own measured 1.3-2.7mm foot "clearance" at rest/no-lift


def load(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env_config = ckpt["env_config"]
    obs_dim = len(ckpt["obs_norm"]["mean"])
    act_dim = ckpt["model_state"]["actor_mean.4.bias"].shape[0]
    norm = RunningNorm(obs_dim)
    norm.load_state_dict(ckpt["obs_norm"])
    return ckpt, obs_dim, act_dim, env_config, norm


def runs(bool_list, want):
    """Maximal runs where bool_list[i]==want, length >= MIN_RUN. Returns
    list of (start, end) index pairs."""
    out = []
    start = None
    seq = bool_list + [not want]
    for i, v in enumerate(seq):
        if v == want and start is None:
            start = i
        elif v != want and start is not None:
            if i - start >= MIN_RUN:
                out.append((start, i))
            start = None
    return out


def probe_one(env, agent, norm, torch_seed, reset_seed=0):
    gen = torch.Generator().manual_seed(torch_seed)
    obs, _ = env.reset(seed=reset_seed)
    touch_l, touch_r, z_l, z_r = [], [], [], []
    clip_frac_steps, delta_steps = [], []
    prev_action = np.zeros(env.n_act)
    fell = False
    steps = 0
    for t in range(MAX_STEPS):
        obs_n = norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32)
            action, _, _, _ = agent.act(obs_t)
        action_np = action.numpy()
        clip_frac_steps.append(float(np.mean(np.abs(action_np) > 1.0)))
        delta_steps.append(float(np.mean(np.abs(action_np - prev_action))))
        prev_action = action_np
        obs, reward, terminated, truncated, info = env.step(action_np)
        steps += 1
        touch_l.append(info["foot_touch"]["l_foot_collision"])
        touch_r.append(info["foot_touch"]["r_foot_collision"])
        z_l.append(info["foot_z"]["l_foot_collision"])
        z_r.append(info["foot_z"]["r_foot_collision"])
        if terminated:
            fell = True
            break
        if truncated:
            break

    def foot_stats(touch, z):
        floor_ref = np.mean([z[i] for i in range(len(z)) if touch[i]]) if any(touch) else min(z) if z else 0.0
        airborne_runs = runs(touch, False)
        best_run, best_clear = None, 0.0
        for s, e in airborne_runs:
            clear = max(z[i] - floor_ref for i in range(s, e))
            if clear > best_clear:
                best_clear, best_run = clear, (s, e)
        supported_landing = False
        if best_run is not None:
            e = best_run[1]
            # Does the step right after the airborne run fall inside a
            # genuine (>=MIN_RUN) contact run -- i.e. did it actually land
            # and hold, not just touch briefly and lift off again?
            land_runs = runs(touch, True)
            supported_landing = any(ls <= e < le for ls, le in land_runs)
        return dict(max_airborne_len=max([e - s for s, e in airborne_runs], default=0),
                    max_clearance_m=best_clear, supported_landing_after_best=supported_landing,
                    n_airborne_runs=len(airborne_runs))

    return dict(steps=steps, fell=fell,
                mean_clip_frac=float(np.mean(clip_frac_steps)), mean_action_delta=float(np.mean(delta_steps)),
                foot_l=foot_stats(touch_l, z_l), foot_r=foot_stats(touch_r, z_r))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    ckpt, obs_dim, act_dim, env_config, norm = load(args.checkpoint)
    desired_vx_obs_idx = 2 * act_dim + 3 + 3

    cfg = CaraWalkEnvConfig(**{**env_config, "episode_seconds": MAX_STEPS / 50.0,
                                "desired_vx_bands": None, "desired_vx": 0.03})
    env = CaraWalkEnv(cfg)

    report = {}
    for mult in MULTIPLIERS:
        agent = ActorCritic(obs_dim, act_dim, command_dependent_std=True,
                             desired_vx_obs_idx=desired_vx_obs_idx, std_boost_multiplier=mult)
        agent.load_state_dict(ckpt["model_state"])
        agent.eval()

        results = [probe_one(env, agent, norm, torch_seed=7000 + i) for i in range(N_REPEATS)]
        fall_rate = np.mean([r["fell"] for r in results])
        clip_rate = np.mean([r["mean_clip_frac"] for r in results])
        jitter = np.mean([r["mean_action_delta"] for r in results])
        max_clear_l = max(r["foot_l"]["max_clearance_m"] for r in results)
        max_clear_r = max(r["foot_r"]["max_clearance_m"] for r in results)
        max_airborne_l = max(r["foot_l"]["max_airborne_len"] for r in results)
        max_airborne_r = max(r["foot_r"]["max_airborne_len"] for r in results)
        n_promising_l = sum(1 for r in results if r["foot_l"]["max_clearance_m"] > NOISE_FLOOR_M
                             and r["foot_l"]["supported_landing_after_best"])
        n_promising_r = sum(1 for r in results if r["foot_r"]["max_clearance_m"] > NOISE_FLOOR_M
                             and r["foot_r"]["supported_landing_after_best"])

        print(f"\n=== multiplier {mult}x ===")
        print(f"  fall_rate={fall_rate:.1%}  mean_clip_frac={clip_rate:.1%}  mean|delta_action|={jitter:.4f}")
        print(f"  max clearance: L={max_clear_l*1000:.1f}mm R={max_clear_r*1000:.1f}mm "
              f"(noise floor {NOISE_FLOOR_M*1000:.0f}mm)")
        print(f"  max airborne run: L={max_airborne_l} R={max_airborne_r} control-steps "
              f"({max_airborne_l*20}/{max_airborne_r*20}ms)")
        print(f"  promising repeats (clearance>floor AND supported landing): L={n_promising_l}/{N_REPEATS} "
              f"R={n_promising_r}/{N_REPEATS}")

        promising = (n_promising_l + n_promising_r) > 0
        rejected_reason = None
        if fall_rate > 0.5:
            rejected_reason = "mostly immediate falls"
        elif clip_rate > 0.5:
            rejected_reason = "mostly clipped actions"
        elif not promising:
            rejected_reason = "no clearance above noise floor with a supported landing -- looks like jitter, not lift attempts"
        verdict = "REJECT: " + rejected_reason if rejected_reason else "PROMISING"
        print(f"  VERDICT: {verdict}")

        report[str(mult)] = dict(fall_rate=float(fall_rate), clip_rate=float(clip_rate), jitter=float(jitter),
                                  max_clearance_l_m=max_clear_l, max_clearance_r_m=max_clear_r,
                                  max_airborne_l_steps=max_airborne_l, max_airborne_r_steps=max_airborne_r,
                                  n_promising_l=n_promising_l, n_promising_r=n_promising_r,
                                  promising=promising, verdict=verdict)

    print("\n=== choice: smallest PROMISING multiplier ===")
    chosen = None
    for mult in MULTIPLIERS:
        if report[str(mult)]["promising"] and not report[str(mult)]["verdict"].startswith("REJECT"):
            chosen = mult
            break
    print(f"  chosen: {chosen}" if chosen else "  NONE promising -- skip the long run, per instruction")
    report["chosen_multiplier"] = chosen

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"saved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
