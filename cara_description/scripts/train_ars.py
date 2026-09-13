#!/usr/bin/env python3
"""U20 -- Augmented Random Search (Mania et al. 2018, ARS-V2) training for
CaraWalkEnv. Numpy-only: the project's .venv has no torch/gymnasium (confirmed
via `pip list`), and ARS is a well-established, competitive, gradient-free
alternative for MuJoCo-style locomotion that avoids adding a deep-learning
dependency just for a first nominal-locomotion check.

Policy: a single linear layer, action = clip(W @ normalize(obs), -1, 1).
`normalize` is an online (Welford) running mean/std over observations seen
across all rollouts so far -- ARS needs this to make each observation
dimension comparably scaled before a linear map.

Per-iteration update (ARS-V2, top-b variant):
  1. Sample N random directions delta_i ~ N(0, I), same shape as W.
  2. Roll out W + sigma*delta_i and W - sigma*delta_i for every direction
     (2N episodes total), each starting from the same reset (no randomized
     initial state in this environment yet -- see rl_environment_notes.md).
  3. Keep only the top b directions ranked by max(r_plus, r_minus).
  4. W <- W + (alpha / (b * std(rewards_used))) * sum_i (r_plus_i - r_minus_i) * delta_i

See docs/rl_environment_notes.md U20 for the frozen run conditions (fixed
masses, flat ground, unchanged actuator limits, single fixed forward-velocity
command) and where results get recorded.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from cara_env import CaraWalkEnv, CaraWalkEnvConfig


class Normalizer:
    """Welford online mean/variance, one running estimate per obs dimension."""

    def __init__(self, dim):
        import numpy as np
        self.np = np
        self.n = 0
        self.mean = np.zeros(dim, dtype="float64")
        self.m2 = np.zeros(dim, dtype="float64")

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def normalize(self, x):
        np = self.np
        if self.n < 2:
            return x
        var = self.m2 / (self.n - 1)
        std = np.sqrt(np.maximum(var, 1e-6))
        return (x - self.mean) / std


def rollout(env, W, normalizer, max_steps, update_normalizer=True):
    np_ = __import__("numpy")
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    fell = False
    for _ in range(max_steps):
        if update_normalizer:
            normalizer.update(obs.astype("float64"))
        obs_n = normalizer.normalize(obs.astype("float64"))
        action = np_.clip(W @ obs_n, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated:
            fell = True
            break
        if truncated:
            break
    return total_reward, steps, fell


def train(env_cfg: CaraWalkEnvConfig, n_iters: int, n_directions: int, top_b: int,
          step_size: float, noise_std: float, max_steps: int, seed: int, log_path: str | None):
    import numpy as np

    env = CaraWalkEnv(env_cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    rng = np.random.default_rng(seed)

    W = np.zeros((act_dim, obs_dim), dtype="float64")
    normalizer = Normalizer(obs_dim)

    history = []
    t0 = time.time()
    for it in range(n_iters):
        deltas = rng.standard_normal((n_directions, act_dim, obs_dim))
        r_plus = np.zeros(n_directions)
        r_minus = np.zeros(n_directions)
        for i in range(n_directions):
            r_plus[i], _, _ = rollout(env, W + noise_std * deltas[i], normalizer, max_steps)
            r_minus[i], _, _ = rollout(env, W - noise_std * deltas[i], normalizer, max_steps)

        order = np.argsort(-np.maximum(r_plus, r_minus))[:top_b]
        used_rewards = np.concatenate([r_plus[order], r_minus[order]])
        sigma_r = used_rewards.std() + 1e-8
        step = np.zeros_like(W)
        for i in order:
            step += (r_plus[i] - r_minus[i]) * deltas[i]
        W += (step_size / (top_b * sigma_r)) * step

        eval_reward, eval_steps, eval_fell = rollout(env, W, normalizer, max_steps,
                                                       update_normalizer=False)
        elapsed = time.time() - t0
        row = dict(iter=it, eval_reward=eval_reward, eval_steps=eval_steps, eval_fell=eval_fell,
                   mean_r_plus=float(r_plus.mean()), mean_r_minus=float(r_minus.mean()),
                   elapsed_s=round(elapsed, 1))
        history.append(row)
        print(f"iter {it:4d}  eval_return={eval_reward:8.2f}  eval_steps={eval_steps:4d}  "
              f"fell={eval_fell!s:5}  mean(r+)={r_plus.mean():7.2f}  mean(r-)={r_minus.mean():7.2f}  "
              f"t={elapsed:6.1f}s")

    if log_path:
        np.savez(log_path, W=W, mean=normalizer.mean, m2=normalizer.m2, n=normalizer.n,
                  env_config=json.dumps(asdict(env_cfg)))
        with open(log_path.rsplit(".", 1)[0] + "_history.json", "w") as f:
            json.dump(history, f, indent=2)
    return W, normalizer, history


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--directions", type=int, default=8)
    ap.add_argument("--top-b", type=int, default=4)
    ap.add_argument("--step-size", type=float, default=0.02)
    ap.add_argument("--noise-std", type=float, default=0.03)
    ap.add_argument("--episode-seconds", type=float, default=4.0)
    ap.add_argument("--desired-vx", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ars_policy.npz")
    args = ap.parse_args(argv)

    cfg = CaraWalkEnvConfig(episode_seconds=args.episode_seconds, desired_vx=args.desired_vx, seed=args.seed)
    max_steps = int(round(args.episode_seconds * cfg.control_hz))

    train(cfg, args.iters, args.directions, args.top_b, args.step_size, args.noise_std,
          max_steps, args.seed, args.out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
