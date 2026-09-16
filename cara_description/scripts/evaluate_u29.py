#!/usr/bin/env python3
"""U29 -- compare zero action, the frozen U28 policy, and the U29 policy
(broader-perturbation continuation) on IDENTICAL held-out disturbances,
reported BY MAGNITUDE, not just an overall average. Also checks whether
survivors actually settle (return toward upright, low-velocity standing)
rather than merely surviving to timeout while still oscillating.

Scope note, stated explicitly per instruction: this evaluates 3 POLICIES
(one per seed) against zero action on the SAME 40 reset cases per
magnitude -- these are 3 policies tested on a shared set of disturbance
instances, not 3*40=120 independent disturbance cases.

Held-out seeds (range(2000, 2040)) are FRESH: never used in U28's own
perturbation test (100-139), U28's evaluate_u28.py fresh set (800-839), or
U29's training-time curriculum-gating validation set (9600-9629).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from train_ppo import ActorCritic, RunningNorm

MAX_STEPS = 200
FRESH_EVAL_SEEDS = list(range(2000, 2040))     # N=40, brand new, never used before in this project
MAGNITUDES = [0.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
SETTLE_WINDOW = 20   # last 20 steps (0.4s) of a survived episode
SETTLE_TILT_DEG = 5.0
SETTLE_ANGVEL_RAD_S = 1.0


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


def rollout(env, action_fn, seed):
    obs, _ = env.reset(seed=seed)
    steps = 0
    fell = False
    tilt_hist, angvel_hist = [], []
    for t in range(MAX_STEPS):
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        tilt_hist.append(info["tilt_deg"])
        angvel_hist.append(float(np.linalg.norm(env.data.qvel[3:6])))
        if terminated:
            fell = True
            break
        if truncated:
            break
    survived_full = not fell and steps >= MAX_STEPS
    settled = None
    if survived_full:
        tail_tilt = np.mean(tilt_hist[-SETTLE_WINDOW:])
        tail_angvel = np.mean(angvel_hist[-SETTLE_WINDOW:])
        settled = bool(tail_tilt < SETTLE_TILT_DEG and tail_angvel < SETTLE_ANGVEL_RAD_S)
    return dict(steps=steps, fell=fell, survived_full=survived_full, settled=settled,
                final_tilt_deg=float(tilt_hist[-1]) if tilt_hist else None,
                tail_tilt_deg=float(np.mean(tilt_hist[-SETTLE_WINDOW:])) if len(tilt_hist) >= 1 else None,
                tail_angvel_rad_s=float(np.mean(angvel_hist[-SETTLE_WINDOW:])) if len(angvel_hist) >= 1 else None)


def zero_action_fn(env):
    return lambda obs: np.zeros(env.n_act)


def deterministic_fn(agent, norm):
    def fn(obs):
        obs_n = norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            mean = agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
        return mean.numpy()
    return fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--u28-checkpoint", required=True, help="frozen U28 final checkpoint for this seed")
    ap.add_argument("--u29-checkpoint", required=True, help="U29 continuation checkpoint for this seed")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    u28_agent, u28_norm, u28_cfg = load(args.u28_checkpoint)
    u29_agent, u29_norm, u29_cfg = load(args.u29_checkpoint)

    report = {"magnitudes": {}}
    for mag in MAGNITUDES:
        cfg = CaraWalkEnvConfig(**{**u28_cfg, "episode_seconds": MAX_STEPS / 50.0,
                                    "reset_qvel_noise_bands": None, "reset_qvel_noise_std": mag})
        env = CaraWalkEnv(cfg)

        zero_results = [rollout(env, zero_action_fn(env), s) for s in FRESH_EVAL_SEEDS]
        u28_results = [rollout(env, deterministic_fn(u28_agent, u28_norm), s) for s in FRESH_EVAL_SEEDS]
        u29_results = [rollout(env, deterministic_fn(u29_agent, u29_norm), s) for s in FRESH_EVAL_SEEDS]

        def summarize(results):
            n = len(results)
            survived = [r for r in results if r["survived_full"]]
            n_settled = sum(1 for r in survived if r["settled"])
            return dict(
                n=n, survival_rate=float(np.mean([r["survived_full"] for r in results])),
                mean_steps=float(np.mean([r["steps"] for r in results])),
                n_survived=len(survived), n_settled_of_survived=n_settled,
                settled_frac_of_survived=(n_settled / len(survived)) if survived else None,
                mean_tail_tilt_of_survived=float(np.mean([r["tail_tilt_deg"] for r in survived])) if survived else None,
                mean_tail_angvel_of_survived=float(np.mean([r["tail_angvel_rad_s"] for r in survived])) if survived else None,
            )

        zs, u28s, u29s = summarize(zero_results), summarize(u28_results), summarize(u29_results)
        print(f"\n=== magnitude {mag} rad/s (N={len(FRESH_EVAL_SEEDS)}) ===")
        print(f"  zero action: survival={zs['survival_rate']:.1%}  mean_steps={zs['mean_steps']:.1f}  "
              f"settled/survived={zs['n_settled_of_survived']}/{zs['n_survived']}")
        print(f"  frozen U28:  survival={u28s['survival_rate']:.1%}  mean_steps={u28s['mean_steps']:.1f}  "
              f"settled/survived={u28s['n_settled_of_survived']}/{u28s['n_survived']}")
        print(f"  U29:         survival={u29s['survival_rate']:.1%}  mean_steps={u29s['mean_steps']:.1f}  "
              f"settled/survived={u29s['n_settled_of_survived']}/{u29s['n_survived']}")
        delta_u29_vs_u28 = u29s["survival_rate"] - u28s["survival_rate"]
        print(f"  U29 vs frozen U28: {delta_u29_vs_u28:+.1%} survival")

        report["magnitudes"][str(mag)] = dict(
            zero_action=zs, frozen_u28=u28s, u29=u29s,
            u29_minus_u28_survival=delta_u29_vs_u28,
            raw=dict(zero=zero_results, u28=u28_results, u29=u29_results))

    # Success criterion readout, per instruction
    print("\n=== success criterion: U29 improves on frozen U28 in the harder band, "
          "while preserving nominal standing and the existing (U28-tested) recovery range ===")
    nominal_ok = (report["magnitudes"]["0.0"]["u29"]["survival_rate"] >=
                  report["magnitudes"]["0.0"]["frozen_u28"]["survival_rate"] - 0.01)
    existing_range_ok = (report["magnitudes"]["6.0"]["u29"]["survival_rate"] >=
                          report["magnitudes"]["6.0"]["frozen_u28"]["survival_rate"] - 0.05)
    harder_improved = any(report["magnitudes"][str(m)]["u29_minus_u28_survival"] > 0.05
                           for m in (7.0, 8.0, 9.0, 10.0))
    print(f"  nominal (0.0) preserved: {nominal_ok}")
    print(f"  existing recovery range (6.0, U28's own test point) preserved: {existing_range_ok}")
    print(f"  harder band (7-10) improved by >5pp somewhere: {harder_improved}")
    success = nominal_ok and existing_range_ok and harder_improved
    print(f"  OVERALL: {'SUCCESS' if success else 'NOT MET -- report where the improvement ends'}")
    report["success_criterion"] = dict(nominal_ok=nominal_ok, existing_range_ok=existing_range_ok,
                                        harder_improved=harder_improved, overall=success)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
