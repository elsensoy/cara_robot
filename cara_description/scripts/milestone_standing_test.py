#!/usr/bin/env python3
"""U26/project standing-milestone gate: >=90% full-episode survival over
30 evaluation resets (proposed project gate, not a universal RL threshold).

CaraWalkEnv.reset() is confirmed deterministic (U23 diagnostic -- no noise
added to the stand_nominal keyframe). With a deterministic policy, 30
identical resets would just repeat one trial 30 times and the "survival
rate" would trivially be 0% or 100%. So this test uses the policy's OWN
learned stochastic sampling (its trained actor_logstd, not a re-init)
across 30 independently-seeded repeats -- the only way 30 resets produce
meaningfully different trials in the CURRENT environment, pending the
separately-defined small-perturbation test. This distinction is stated
explicitly in the output, not silently assumed.

Exact-reset standing alone does not demonstrate recovery -- that is a
separate, not-yet-defined test.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from train_ppo import ActorCritic, RunningNorm

MAX_STEPS = 200  # 4s @ 50Hz -- the standard episode length used throughout U20-U26
GATE = 0.9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--n-resets", type=int, default=30)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env_config = ckpt["env_config"]
    cfg = CaraWalkEnvConfig(**{**env_config, "episode_seconds": MAX_STEPS / 50.0, "desired_vx": 0.0})
    env = CaraWalkEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    agent = ActorCritic(obs_dim, env.n_act)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    norm = RunningNorm(obs_dim)
    norm.load_state_dict(ckpt["obs_norm"])

    print("NOTE: CaraWalkEnv.reset() is deterministic (confirmed in U23) -- with a")
    print("deterministic policy, 30 'resets' would be one trial repeated 30 times.")
    print("This test uses the policy's OWN LEARNED stochastic sampling instead, "
          "across independent repeats.\n")

    results = []
    for i in range(args.n_resets):
        gen = torch.Generator().manual_seed(5000 + i)
        obs, _ = env.reset(seed=i)
        steps = 0
        fell = False
        for t in range(MAX_STEPS):
            obs_n = norm.normalize(obs.astype("float64"))
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_n, dtype=torch.float32)
                mean = agent.actor_mean(obs_t)
                std = torch.exp(agent.actor_logstd.view(-1))
                noise = torch.randn(mean.shape, generator=gen)
                action = (mean + noise * std).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated:
                fell = True
                break
            if truncated:
                break
        results.append(dict(steps=steps, fell=fell, survived_full=not fell and steps >= MAX_STEPS))
        print(f"  repeat {i:2d}: steps={steps:3d}/{MAX_STEPS}  fell={fell}")

    survived = sum(r["survived_full"] for r in results)
    rate = survived / args.n_resets
    print(f"\n=== MILESTONE: full-episode survival over {args.n_resets} resets ===")
    print(f"survived: {survived}/{args.n_resets} = {rate:.1%}  (gate: >={GATE:.0%})")
    print(f"mean steps: {np.mean([r['steps'] for r in results]):.1f}  "
          f"fall rate: {np.mean([r['fell'] for r in results]):.1%}")
    print(f"VERDICT: {'MET' if rate >= GATE else 'NOT MET'}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(dict(checkpoint=args.checkpoint, n_resets=args.n_resets, max_steps=MAX_STEPS,
                            results=results, survived_full_count=survived, survival_rate=rate,
                            gate=GATE, met=(rate >= GATE)), f, indent=2)
        print(f"saved -> {args.out_json}")
    return 0 if rate >= GATE else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
