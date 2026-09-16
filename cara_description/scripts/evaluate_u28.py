#!/usr/bin/env python3
"""U28 -- three-seed reliability check for U27's standing-preservation result.

Reports, for each of 3 trained checkpoints (final checkpoint selected per
seed -- see selection note below) plus the untrained zero-mean-init policy:
  1. Nominal deterministic survival
  2. Sampled nominal survival (N=30) -- includes the UNTRAINED policy so
     initialization's own contribution is visible separately from learning
  3. Perturbed survival vs. zero action on FRESH held-out resets (never
     used in checkpoint selection or in U27's own perturbation test)
  4. Paired per-reset step differences, with the exact win criterion stated
     explicitly and survival-outcome changes reported SEPARATELY from
     time-to-fall-only differences (correction from U27: "9 wins" conflated
     these two very different kinds of evidence)

Checkpoint selection note: U27's original "best" checkpoint mechanism used
a single deterministic rollout that saturates immediately (confirmed: U27
locked in an iteration-5 snapshot with only 2.6% of training data). U28's
train_ppo.py now scores checkpoints on a 10-sample validation set instead,
but a retroactive check (this file's own analysis, N=30 held-out samples)
found even that saturates too early to reliably beat simply using the
FINAL checkpoint for all 3 seeds (best: 73-97% survival; final: 100% for
all three). The FINAL checkpoint is used for all 3 seeds here -- disclosed,
not silently chosen.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from train_ppo import ActorCritic, RunningNorm

MAX_STEPS = 200
SAMPLED_NOMINAL_SEEDS = list(range(6000, 6030))       # N=30, same set U27 used for its own sampled-nominal test
FRESH_PERTURBED_SEEDS = list(range(800, 840))          # N=40, NEVER used before in this project
RESET_QVEL_NOISE_STD = 6.0                             # chosen in U27 against zero-action itself


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


def rollout_sampled(env, agent, norm, reset_seed, torch_seed):
    gen = torch.Generator().manual_seed(torch_seed)
    obs, _ = env.reset(seed=reset_seed)
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


def summarize_survival(results):
    n = len(results)
    return dict(n=n, survival_rate=float(np.mean([r["survived_full"] for r in results])),
                fall_rate=float(np.mean([r["fell"] for r in results])),
                mean_steps=float(np.mean([r["steps"] for r in results])))


def paired_comparison(policy_results, zero_results):
    """Exact win criterion, stated explicitly: policy 'wins' a reset iff its
    step count is STRICTLY greater than zero-action's on that SAME reset.
    Survival-outcome changes (one side reached MAX_STEPS, the other didn't)
    are reported separately from time-to-fall-only differences (both fell,
    but at different step counts) -- these are different strengths of
    evidence and were conflated in U27's reporting."""
    n = len(policy_results)
    survival_flips_policy_only = 0   # policy survived full episode, zero did not
    survival_flips_zero_only = 0     # zero survived full episode, policy did not
    both_survived = 0
    time_to_fall_policy_longer = 0   # both fell, policy lasted longer
    time_to_fall_zero_longer = 0     # both fell, zero lasted longer
    both_fell_tied = 0
    step_diffs = []
    for p, z in zip(policy_results, zero_results):
        step_diffs.append(p["steps"] - z["steps"])
        if p["survived_full"] and z["survived_full"]:
            both_survived += 1
        elif p["survived_full"] and not z["survived_full"]:
            survival_flips_policy_only += 1
        elif z["survived_full"] and not p["survived_full"]:
            survival_flips_zero_only += 1
        else:  # both fell
            if p["steps"] > z["steps"]:
                time_to_fall_policy_longer += 1
            elif p["steps"] < z["steps"]:
                time_to_fall_zero_longer += 1
            else:
                both_fell_tied += 1
    return dict(
        n=n, both_survived=both_survived,
        survival_flips_policy_only=survival_flips_policy_only,
        survival_flips_zero_only=survival_flips_zero_only,
        time_to_fall_policy_longer=time_to_fall_policy_longer,
        time_to_fall_zero_longer=time_to_fall_zero_longer,
        both_fell_tied=both_fell_tied,
        mean_step_diff=float(np.mean(step_diffs)), median_step_diff=float(np.median(step_diffs)),
        step_diffs=step_diffs,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs=3, required=True, help="the 3 seeds' checkpoint .pt paths, in order")
    ap.add_argument("--untrained-checkpoint", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    report = {"seeds": {}}

    # untrained: sampled-nominal only (deterministic == zero action exactly, by construction)
    u_agent, u_norm, u_env_config = load(args.untrained_checkpoint)
    cfg_nominal = CaraWalkEnvConfig(**{**u_env_config, "episode_seconds": MAX_STEPS / 50.0})
    env_nominal = CaraWalkEnv(cfg_nominal)
    untrained_sampled = [rollout_sampled(env_nominal, u_agent, u_norm, reset_seed=0, torch_seed=s)
                          for s in SAMPLED_NOMINAL_SEEDS]
    untrained_summary = summarize_survival(untrained_sampled)
    print(f"=== UNTRAINED policy (zero-mean-init, init_logstd=-3.0), sampled nominal (N={len(SAMPLED_NOMINAL_SEEDS)}) ===")
    print(f"  survival_rate={untrained_summary['survival_rate']:.1%}  mean_steps={untrained_summary['mean_steps']:.1f}")
    print(f"  (deterministic nominal == zero action exactly, by construction of zero_mean_init -- not re-tested)")
    report["untrained_sampled_nominal"] = untrained_summary

    for i, ckpt_path in enumerate(args.checkpoints):
        seed_name = f"seed{i}"
        print(f"\n{'='*20} {seed_name}: {ckpt_path} {'='*20}")
        agent, norm, env_config = load(ckpt_path)
        cfg_nominal = CaraWalkEnvConfig(**{**env_config, "episode_seconds": MAX_STEPS / 50.0})
        env_nominal = CaraWalkEnv(cfg_nominal)

        det = rollout_deterministic(env_nominal, agent, norm, seed=0)
        print(f"  1. deterministic nominal: steps={det['steps']}/{MAX_STEPS} fell={det['fell']}")

        sampled = [rollout_sampled(env_nominal, agent, norm, reset_seed=0, torch_seed=s)
                   for s in SAMPLED_NOMINAL_SEEDS]
        sampled_summary = summarize_survival(sampled)
        print(f"  2. sampled nominal (N={len(SAMPLED_NOMINAL_SEEDS)}): "
              f"survival={sampled_summary['survival_rate']:.1%}  mean_steps={sampled_summary['mean_steps']:.1f}")

        cfg_perturbed = CaraWalkEnvConfig(**{**env_config, "episode_seconds": MAX_STEPS / 50.0,
                                              "reset_qvel_noise_std": RESET_QVEL_NOISE_STD})
        env_perturbed = CaraWalkEnv(cfg_perturbed)
        policy_perturbed = [rollout_deterministic(env_perturbed, agent, norm, seed=s) for s in FRESH_PERTURBED_SEEDS]
        zero_perturbed = [rollout_zero_action(env_perturbed, seed=s) for s in FRESH_PERTURBED_SEEDS]
        policy_summary = summarize_survival(policy_perturbed)
        zero_summary = summarize_survival(zero_perturbed)
        pairing = paired_comparison(policy_perturbed, zero_perturbed)
        print(f"  3. perturbed (N={len(FRESH_PERTURBED_SEEDS)}, reset_qvel_noise_std={RESET_QVEL_NOISE_STD}, FRESH seeds 800-839):")
        print(f"     policy: survival={policy_summary['survival_rate']:.1%} mean_steps={policy_summary['mean_steps']:.1f}")
        print(f"     zero:   survival={zero_summary['survival_rate']:.1%} mean_steps={zero_summary['mean_steps']:.1f}")
        print(f"     win criterion: policy_steps > zero_steps on the SAME reset, evaluated separately by outcome type:")
        print(f"       both survived full episode: {pairing['both_survived']}/{pairing['n']}")
        print(f"       SURVIVAL FLIP in policy's favor (policy survived, zero fell): {pairing['survival_flips_policy_only']}")
        print(f"       SURVIVAL FLIP in zero's favor (zero survived, policy fell): {pairing['survival_flips_zero_only']}")
        print(f"       both fell, policy lasted longer (time-to-fall only): {pairing['time_to_fall_policy_longer']}")
        print(f"       both fell, zero lasted longer (time-to-fall only): {pairing['time_to_fall_zero_longer']}")
        print(f"       both fell, tied exactly: {pairing['both_fell_tied']}")
        print(f"       mean paired step difference (policy-zero): {pairing['mean_step_diff']:+.1f}")

        report["seeds"][seed_name] = dict(
            checkpoint=ckpt_path, deterministic_nominal=det, sampled_nominal=sampled_summary,
            perturbed_policy=policy_summary, perturbed_zero=zero_summary, pairing=pairing)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
