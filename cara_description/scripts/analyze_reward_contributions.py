#!/usr/bin/env python3
"""U26 prep -- measure the ACTUAL raw reward-component distribution under
U24, so a new w_action_rate can be chosen from measured contributions
rather than guessed from the coefficient's size alone.

Two rollouts, same env/resets as everywhere else (desired_vx=0.0 stage-0):
  1. Deterministic mean policy (what actually gets evaluated) -- one
     rollout, fully reproducible, runs until it falls.
  2. Stochastic sampled policy using U24's OWN LEARNED std (not a fresh
     init) -- this is what PPO's training-time rollouts actually look like,
     so it's the more representative distribution for picking a training
     reward weight.

Reports mean/median/p75/p90/max of the RAW (unweighted) r_rate and, for
context, r_effort, alongside r_alive/r_vel/r_upright, and proposes a
candidate w_action_rate: the weight at which the P75 raw |r_rate| costs a
stated, deliberately modest fraction of the per-step alive bonus -- stated
explicitly so the choice is auditable, not asserted.
"""

import argparse
import json

import numpy as np
import torch

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from train_ppo import ActorCritic, RunningNorm

TARGET_FRACTION_OF_ALIVE = 0.10  # candidate: P75 jitter should cost ~10% of the alive bonus, not overwhelm it


def rollout(env, agent, norm, max_steps, sample, seed=0):
    obs, _ = env.reset(seed=seed)
    rows = []
    for t in range(max_steps):
        obs_n = norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32)
            mean = agent.actor_mean(obs_t)
            if sample:
                std = torch.exp(agent.actor_logstd.view(-1))
                action = mean + torch.randn(mean.shape) * std
            else:
                action = mean
        action_np = action.numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
        rc = info["reward_components"]
        if rc["alive"] is not None:  # None on the terminal fall step
            rows.append(rc)
        if terminated or truncated:
            break
    return rows


def report(name, rows):
    print(f"\n--- {name} ({len(rows)} non-terminal steps) ---")
    out = {}
    for k in ("alive", "vel", "upright", "effort", "rate", "collision"):
        vals = np.array([r[k] for r in rows])
        stats = dict(mean=float(vals.mean()), median=float(np.median(vals)),
                     p75_abs=float(np.percentile(np.abs(vals), 75)),
                     p90_abs=float(np.percentile(np.abs(vals), 90)),
                     max_abs=float(np.max(np.abs(vals))))
        out[k] = stats
        print(f"  r_{k:10s} mean={stats['mean']:+.5f}  median={stats['median']:+.5f}  "
              f"P75(|.|)={stats['p75_abs']:.5f}  P90(|.|)={stats['p90_abs']:.5f}  max(|.|)={stats['max_abs']:.5f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="../runs/u24_ppo_lowstd/ppo_lowstd_seed0.pt")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env_config = ckpt["env_config"]
    cfg = CaraWalkEnvConfig(**{**env_config, "episode_seconds": 4.0, "desired_vx": 0.0})
    env = CaraWalkEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    agent = ActorCritic(obs_dim, env.n_act)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    norm = RunningNorm(obs_dim)
    norm.load_state_dict(ckpt["obs_norm"])
    max_steps = int(round(4.0 * cfg.control_hz))

    det_rows = rollout(env, agent, norm, max_steps, sample=False, seed=0)
    samp_rows = rollout(env, agent, norm, max_steps, sample=True, seed=0)

    det_stats = report("deterministic mean (U24 as evaluated)", det_rows)
    samp_stats = report("stochastic sampled, U24's OWN learned std (what training rollouts look like)", samp_rows)

    print(f"\n=== deriving w_action_rate from measured contributions ===")
    print(f"target: P75 |raw r_rate| should cost {TARGET_FRACTION_OF_ALIVE:.0%} of the per-step alive "
          f"bonus (w_alive * r_alive = {cfg.w_alive * 1.0:.2f}/step)")
    for name, stats in (("deterministic", det_stats), ("sampled", samp_stats)):
        p75 = stats["rate"]["p75_abs"]
        if p75 > 0:
            w = TARGET_FRACTION_OF_ALIVE * cfg.w_alive / p75
            print(f"  from {name} rollout: P75|raw r_rate|={p75:.5f}  ->  candidate w_action_rate={w:.3f}")
        else:
            print(f"  from {name} rollout: P75|raw r_rate|=0 (degenerate, skip)")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"deterministic": det_stats, "sampled": samp_stats,
                       "target_fraction_of_alive": TARGET_FRACTION_OF_ALIVE}, f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
