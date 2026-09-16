#!/usr/bin/env python3
"""U27 three-evaluation suite. Per the explicit correction: 30 stochastic
rollouts from ONE fixed reset test action-sampling robustness, NOT
robustness to initial-state variation -- these are different questions and
get separate, clearly-labeled results here.

  1. Deterministic policy, nominal reset  -- does learning preserve the
     zero-action standing solution?
  2. Sampled policy, nominal reset (N repeats) -- is exploration tolerable?
     (This is what the earlier "30 resets" milestone test actually measured
     -- relabeled honestly here, not as reset-variation robustness.)
  3. Deterministic policy, VARIED (perturbed) resets -- does feedback
     actually recover from disturbances? Compared directly against ZERO
     ACTION on the SAME perturbed resets -- if both survive equally, the
     perturbations have not demonstrated added feedback capability.

Perturbation family (first and only one tested here, per instruction):
small initial joint-velocity noise, CaraWalkEnvConfig.reset_qvel_noise_std.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from train_ppo import ActorCritic, RunningNorm

MAX_STEPS = 200  # 4s @ 50Hz, the standard episode length used throughout U20-U27


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


def rollout_deterministic(env, agent, norm, seed):
    obs, _ = env.reset(seed=seed)
    steps = 0
    fell = False
    for t in range(MAX_STEPS):
        obs_n = norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            mean = agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
        obs, reward, terminated, truncated, info = env.step(mean.numpy())
        steps += 1
        if terminated:
            fell = True
            break
        if truncated:
            break
    return dict(steps=steps, fell=fell, survived_full=not fell and steps >= MAX_STEPS)


def rollout_zero_action(env, seed):
    obs, _ = env.reset(seed=seed)
    steps = 0
    fell = False
    for t in range(MAX_STEPS):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.n_act))
        steps += 1
        if terminated:
            fell = True
            break
        if truncated:
            break
    return dict(steps=steps, fell=fell, survived_full=not fell and steps >= MAX_STEPS)


def rollout_sampled(env, agent, norm, seed, torch_seed):
    gen = torch.Generator().manual_seed(torch_seed)
    obs, _ = env.reset(seed=seed)
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
    return dict(steps=steps, fell=fell, survived_full=not fell and steps >= MAX_STEPS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--n-sampled-repeats", type=int, default=30)
    ap.add_argument("--n-perturbed-resets", type=int, default=40)
    ap.add_argument("--reset-qvel-noise-std", type=float, default=6.0,
                     help="rad/s -- the one perturbation family tested (initial joint-velocity noise). "
                          "6.0 chosen from a sweep: the passive position-PD baseline (zero action) is "
                          "fully robust (100%% full-episode survival) up to ~3 rad/s and only starts "
                          "degrading past 4 rad/s -- below that, this test cannot discriminate anything.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    agent, norm, env_config = load(args.checkpoint)
    report = {}

    # ---- 1. deterministic, nominal reset ----
    cfg_nominal = CaraWalkEnvConfig(**{**env_config, "episode_seconds": MAX_STEPS / 50.0})
    env_nominal = CaraWalkEnv(cfg_nominal)
    r1 = rollout_deterministic(env_nominal, agent, norm, seed=0)
    print(f"\n=== 1. Deterministic policy, nominal reset ===")
    print(f"  steps={r1['steps']}/{MAX_STEPS}  fell={r1['fell']}  survived_full={r1['survived_full']}")
    print(f"  -> {'Learning PRESERVES the standing solution' if r1['survived_full'] else 'Learning did NOT preserve standing'}")
    report["1_deterministic_nominal"] = r1

    # ---- 2. sampled, nominal reset (N repeats) -- exploration tolerance ----
    print(f"\n=== 2. Sampled policy, nominal reset ({args.n_sampled_repeats} repeats) ===")
    print("    (this tests ACTION-SAMPLING robustness, not reset/initial-state variation)")
    sampled_results = [rollout_sampled(env_nominal, agent, norm, seed=0, torch_seed=6000 + i)
                        for i in range(args.n_sampled_repeats)]
    surv_rate = np.mean([r["survived_full"] for r in sampled_results])
    fall_rate = np.mean([r["fell"] for r in sampled_results])
    mean_steps = np.mean([r["steps"] for r in sampled_results])
    print(f"  full-episode survival: {surv_rate:.1%}  fall rate: {fall_rate:.1%}  mean steps: {mean_steps:.1f}")
    print(f"  -> exploration is {'tolerable' if surv_rate > 0.5 else 'NOT tolerable'} at this std")
    report["2_sampled_nominal"] = dict(repeats=sampled_results, survival_rate=float(surv_rate),
                                        fall_rate=float(fall_rate), mean_steps=float(mean_steps))

    # ---- 3. deterministic, VARIED (perturbed) resets -- vs zero action ----
    cfg_perturbed = CaraWalkEnvConfig(**{**env_config, "episode_seconds": MAX_STEPS / 50.0,
                                          "reset_qvel_noise_std": args.reset_qvel_noise_std})
    env_perturbed = CaraWalkEnv(cfg_perturbed)
    print(f"\n=== 3. Deterministic policy vs. ZERO ACTION, matched perturbed resets "
          f"(reset_qvel_noise_std={args.reset_qvel_noise_std} rad/s, {args.n_perturbed_resets} resets) ===")
    policy_results, zero_results = [], []
    for i in range(args.n_perturbed_resets):
        # Same seed -> same perturbation draw for both, since perturbation
        # sampling happens inside reset() using the seeded self._rng.
        policy_results.append(rollout_deterministic(env_perturbed, agent, norm, seed=100 + i))
        zero_results.append(rollout_zero_action(env_perturbed, seed=100 + i))
    policy_surv = np.mean([r["survived_full"] for r in policy_results])
    zero_surv = np.mean([r["survived_full"] for r in zero_results])
    policy_steps = np.mean([r["steps"] for r in policy_results])
    zero_steps = np.mean([r["steps"] for r in zero_results])
    # Per-reset head-to-head is more sensitive than aggregate survival rate
    # at moderate N (a 2% aggregate move can still hide a consistent,
    # directional per-reset pattern, or vice versa).
    n_policy_better = sum(1 for p, z in zip(policy_results, zero_results) if p["steps"] > z["steps"])
    n_zero_better = sum(1 for p, z in zip(policy_results, zero_results) if p["steps"] < z["steps"])
    n_tied = args.n_perturbed_resets - n_policy_better - n_zero_better
    print(f"  learned policy: full-survival={policy_surv:.1%}  mean steps={policy_steps:.1f}")
    print(f"  zero action:    full-survival={zero_surv:.1%}  mean steps={zero_steps:.1f}")
    print(f"  per-reset head-to-head: policy better={n_policy_better}  zero better={n_zero_better}  tied={n_tied}")
    if abs(policy_surv - zero_surv) < 0.05 and n_policy_better <= n_zero_better * 1.5:
        verdict = "EQUIVALENT to zero action -- these perturbations have NOT demonstrated added feedback capability"
    elif policy_surv >= zero_surv and n_policy_better > n_zero_better:
        verdict = "learned policy shows a real but MODEST recovery edge over zero action"
    else:
        verdict = "learned policy is WORSE than zero action under perturbation"
    print(f"  -> {verdict}")
    report["3_deterministic_perturbed_vs_zero"] = dict(
        reset_qvel_noise_std=args.reset_qvel_noise_std, n_resets=args.n_perturbed_resets,
        policy_results=policy_results, zero_results=zero_results,
        policy_survival_rate=float(policy_surv), zero_survival_rate=float(zero_surv),
        policy_mean_steps=float(policy_steps), zero_mean_steps=float(zero_steps),
        n_policy_better=n_policy_better, n_zero_better=n_zero_better, n_tied=n_tied,
        verdict=verdict)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
