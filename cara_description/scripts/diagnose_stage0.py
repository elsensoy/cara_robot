#!/usr/bin/env python3
"""U23 -- bounded diagnostic, NOT a training run. Isolates why PPO training
moves away from the zero-action standing baseline even at stage 0
(desired_vx=0.0, zero forward-velocity demand). Run with .venv-rl (needs
torch for the policy controllers).

Four controllers, identical resets / stage-0 reward / episode length /
model config:

  1. zero action                      -- reconfirms standing under the
                                          actual training conditions
  2. untrained policy, deterministic  -- does initialization start near
                                          nominal stance?
  3. untrained policy, sampled        -- how disruptive is initial
                                          exploration?
  4. trained policy, deterministic    -- what did learning change?
     (the U22 curriculum checkpoint, stuck at stage 0)

Note on "identical resets": CaraWalkEnv.reset() is currently fully
deterministic (mj_resetDataKeyframe to the stand_nominal keyframe, no noise
added) -- confirmed by reading cara_env.py, not assumed. So conditions 1, 2,
and 4 are each exactly reproducible in a SINGLE rollout (no seed sweep adds
information); only condition 3 (sampling) has genuine randomness, so it gets
repeated across N_REPEATS independent samples. Observation normalizer state
is held FROZEN during every rollout here (no online .update() calls) --
untrained conditions use a fresh, never-updated RunningNorm (mean=0, var=1,
matching what a real training run's normalizer looks like at step 0);
condition 4 uses the actual trained normalizer loaded from the checkpoint.
This isolates policy behavior cleanly from a shifting normalizer, which is a
separate question from what's being asked here.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from train_ppo import ActorCritic, RunningNorm

N_REPEATS_SAMPLED = 10
EPISODE_SECONDS = 4.0  # matches U20-U22's stage-0 / training episode length


def print_action_range_table(env):
    """Translate action_range_frac into actual per-joint radians/degrees --
    the fraction alone doesn't say whether the range is sensible."""
    print("\n=== action_range_frac=0.30 translated to per-joint radians ===")
    print(f"{'joint':<16} {'lower':>8} {'upper':>8} {'offset_scale(rad)':>18} {'offset_scale(deg)':>18}")
    for i, name in enumerate(env.jn):
        print(f"{name:<16} {env.lo[i]:>8.3f} {env.hi[i]:>8.3f} "
              f"{env.offset_scale[i]:>18.4f} {np.degrees(env.offset_scale[i]):>18.2f}")
    # Initial exploration std (actor_logstd = -0.5 at init -> std=exp(-0.5))
    action_std = float(np.exp(-0.5))
    print(f"\ninitial exploration std (action-space units, all joints): {action_std:.4f}")
    print(f"{'joint':<16} {'1-sigma swing (rad)':>20} {'1-sigma swing (deg)':>20}")
    for i, name in enumerate(env.jn):
        swing = action_std * env.offset_scale[i]
        print(f"{name:<16} {swing:>20.4f} {np.degrees(swing):>20.2f}")
    # theoretical clip probability for a single joint, single step, at init
    from math import erf, sqrt
    z = 1.0 / action_std
    p_within = erf(z / sqrt(2))
    p_clip_one_joint = 1.0 - p_within
    p_clip_any_of_12 = 1.0 - (1.0 - p_clip_one_joint) ** 12
    print(f"\ntheoretical P(|raw action| > 1.0) per joint per step at init: {p_clip_one_joint:.1%}")
    print(f"theoretical P(at least one of 12 joints clipped) per step at init: {p_clip_any_of_12:.1%}")


def rollout(env, act_fn, obs_norm, max_steps, seed=0):
    """act_fn(obs_raw, obs_norm) -> (raw_action, action_used) both length-12
    numpy arrays; raw_action is what the policy actually produced BEFORE the
    environment's internal [-1,1] clip, action_used is what was passed to
    env.step (== raw_action for these controllers; env.step re-clips
    internally, so action_used here is only used for our own clip-fraction
    bookkeeping, matching exactly what the env will do)."""
    obs, _ = env.reset(seed=seed)
    prev_raw = np.zeros(env.n_act)
    log = dict(reward=[], r_alive=[], r_vel=[], r_upright=[], r_effort=[], r_rate=[], r_collision=[],
               raw_action_absmean=[], action_delta_absmean=[], action_clip_frac=[], target_clip_frac=[],
               qpos_err=[], torque_frac=[], tilt_deg=[])
    fell = False
    steps = 0
    for t in range(max_steps):
        raw_action = act_fn(obs, obs_norm)
        action_used = np.clip(raw_action, -1.0, 1.0)
        target_preclip = env.nominal + action_used * env.offset_scale
        target_clip_frac = float(np.mean((target_preclip < env.lo) | (target_preclip > env.hi)))
        action_clip_frac = float(np.mean(np.abs(raw_action) > 1.0))
        delta = raw_action - prev_raw
        prev_raw = raw_action

        obs, reward, terminated, truncated, info = env.step(raw_action)
        steps += 1
        rc = info["reward_components"]
        log["reward"].append(reward)
        for k in ("alive", "vel", "upright", "effort", "rate", "collision"):
            log[f"r_{k}"].append(rc[k])  # None on the terminal fall step -- filtered at aggregation
        log["raw_action_absmean"].append(float(np.mean(np.abs(raw_action))))
        log["action_delta_absmean"].append(float(np.mean(np.abs(delta))))
        log["action_clip_frac"].append(action_clip_frac)
        log["target_clip_frac"].append(target_clip_frac)
        log["qpos_err"].append(info["qpos_err"])
        log["torque_frac"].append(info["torque_frac"])
        log["tilt_deg"].append(info["tilt_deg"])
        if terminated:
            fell = True
            break
        if truncated:
            break
    return dict(fell=fell, steps=steps, log=log)


def summarize(name, result):
    log = result["log"]
    n = result["steps"]
    total_return = float(np.sum(log["reward"]))
    comp_sums = {}
    for k in ("alive", "vel", "upright", "effort", "rate", "collision"):
        vals = [v for v in log[f"r_{k}"] if v is not None]
        comp_sums[k] = float(np.sum(vals)) if vals else 0.0
    fall_penalty = -10.0 if result["fell"] else 0.0
    print(f"\n--- {name} ---")
    print(f"  steps: {n}  fell: {result['fell']}")
    print(f"  TOTAL RETURN: {total_return:.2f}  "
          f"(components: alive={comp_sums['alive']:.2f} vel={comp_sums['vel']:.2f} "
          f"upright={comp_sums['upright']:.2f} effort={comp_sums['effort']:.2f} "
          f"rate={comp_sums['rate']:.2f} collision={comp_sums['collision']:.2f} "
          f"fall_penalty={fall_penalty:.2f})")
    print(f"  action magnitude (mean |raw action|): {np.mean(log['raw_action_absmean']):.4f}")
    print(f"  action change (mean |Δraw action| step-to-step): {np.mean(log['action_delta_absmean']):.4f}")
    print(f"  action clip rate (|raw action|>1.0, fraction of joint-steps): "
          f"{np.mean(log['action_clip_frac']):.1%}")
    print(f"  target clip rate (joint target outside its limits, fraction of joint-steps): "
          f"{np.mean(log['target_clip_frac']):.1%}")
    print(f"  actuator saturation: mean={np.mean(log['torque_frac']):.1%} max={np.max(log['torque_frac']):.1%}")
    print(f"  joint tracking error: mean={np.mean(log['qpos_err']):.4f} max={np.max(log['qpos_err']):.4f} rad")
    print(f"  peak tilt: {np.max(log['tilt_deg']):.1f} deg")
    return dict(steps=n, fell=result["fell"], total_return=total_return, components=comp_sums,
                fall_penalty=fall_penalty,
                action_absmean=float(np.mean(log["raw_action_absmean"])),
                action_delta_absmean=float(np.mean(log["action_delta_absmean"])),
                action_clip_rate=float(np.mean(log["action_clip_frac"])),
                target_clip_rate=float(np.mean(log["target_clip_frac"])),
                torque_frac_mean=float(np.mean(log["torque_frac"])),
                torque_frac_max=float(np.max(log["torque_frac"])),
                qpos_err_mean=float(np.mean(log["qpos_err"])),
                qpos_err_max=float(np.max(log["qpos_err"])),
                tilt_max=float(np.max(log["tilt_deg"])))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trained-checkpoint",
                     default="../runs/u22_ppo_curriculum/ppo_curr_seed0.pt")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    cfg = CaraWalkEnvConfig(episode_seconds=EPISODE_SECONDS, desired_vx=0.0)  # stage 0
    env = CaraWalkEnv(cfg)
    max_steps = int(round(EPISODE_SECONDS * cfg.control_hz))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.n_act

    print_action_range_table(env)

    report = {}

    # 1. zero action
    zero_fn = lambda obs, norm: np.zeros(act_dim)
    r = rollout(env, zero_fn, None, max_steps, seed=0)
    report["1_zero_action"] = summarize("1. zero action", r)

    # 2 & 3. untrained policy (fresh init, matching train_ppo.py's ActorCritic exactly)
    torch.manual_seed(0)
    untrained_agent = ActorCritic(obs_dim, act_dim)
    untrained_agent.eval()
    fresh_norm = RunningNorm(obs_dim)  # never updated -- "as seen at step 0 of training"

    def untrained_mean_fn(obs, norm):
        obs_n = norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            mean = untrained_agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
        return mean.numpy()

    r = rollout(env, untrained_mean_fn, fresh_norm, max_steps, seed=0)
    report["2_untrained_mean"] = summarize("2. untrained policy, deterministic mean", r)

    def make_untrained_sampled_fn(torch_seed):
        gen = torch.Generator().manual_seed(torch_seed)

        def fn(obs, norm):
            obs_n = norm.normalize(obs.astype("float64"))
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_n, dtype=torch.float32)
                mean = untrained_agent.actor_mean(obs_t)
                std = torch.exp(untrained_agent.actor_logstd.view(-1))  # state-independent, same shape as unbatched mean
                noise = torch.randn(mean.shape, generator=gen)
                action = mean + noise * std
            return action.numpy()
        return fn

    sampled_returns, sampled_steps, sampled_fell = [], [], []
    sampled_summaries = []
    for rep in range(N_REPEATS_SAMPLED):
        fn = make_untrained_sampled_fn(1000 + rep)
        r = rollout(env, fn, fresh_norm, max_steps, seed=0)
        s = summarize(f"3. untrained policy, sampled (repeat {rep})", r)
        sampled_summaries.append(s)
        sampled_returns.append(s["total_return"])
        sampled_steps.append(s["steps"])
        sampled_fell.append(s["fell"])
    print(f"\n--- 3. untrained policy, sampled: aggregate over {N_REPEATS_SAMPLED} repeats ---")
    print(f"  mean return: {np.mean(sampled_returns):.2f}  mean steps: {np.mean(sampled_steps):.1f}  "
          f"fall rate: {np.mean(sampled_fell):.1%}")
    report["3_untrained_sampled_repeats"] = sampled_summaries
    report["3_untrained_sampled_aggregate"] = dict(
        mean_return=float(np.mean(sampled_returns)), mean_steps=float(np.mean(sampled_steps)),
        fall_rate=float(np.mean(sampled_fell)))

    # 4. trained policy, deterministic mean (the U22 curriculum checkpoint)
    ckpt = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=False)
    trained_agent = ActorCritic(obs_dim, act_dim)
    trained_agent.load_state_dict(ckpt["model_state"])
    trained_agent.eval()
    trained_norm = RunningNorm(obs_dim)
    trained_norm.load_state_dict(ckpt["obs_norm"])

    def trained_mean_fn(obs, norm):
        obs_n = norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            mean = trained_agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
        return mean.numpy()

    r = rollout(env, trained_mean_fn, trained_norm, max_steps, seed=0)
    report["4_trained_mean"] = summarize("4. trained policy, deterministic mean", r)

    # ---- decision-sequence readout ----
    print("\n=== decision sequence ===")
    z = report["1_zero_action"]
    print(f"1. zero action under training resets: steps={z['steps']}/{max_steps} fell={z['fell']}  "
          f"return={z['total_return']:.2f}  "
          f"-> {'FAILS to stand -- inspect reset differences and baseline stability' if z['fell'] or z['steps'] < max_steps else 'holds stance as expected'}")
    t4 = report["4_trained_mean"]
    print(f"   zero-action return ({z['total_return']:.2f}) vs learned-fall return ({t4['total_return']:.2f}): "
          f"{'objective favors the WRONG behavior -- fix the objective' if t4['total_return'] > z['total_return'] else 'objective correctly favors standing -- the problem is exploration/optimization, not reward'}")
    m2 = report["2_untrained_mean"]
    s3 = report["3_untrained_sampled_aggregate"]
    print(f"2. untrained mean: steps={m2['steps']}/{max_steps} fell={m2['fell']}  "
          f"3. untrained sampled: mean_steps={s3['mean_steps']:.1f} fall_rate={s3['fall_rate']:.1%}  "
          f"action_clip_rate(sampled, repeat0)={report['3_untrained_sampled_repeats'][0]['action_clip_rate']:.1%}")
    if not m2["fell"] and s3["fall_rate"] > 0.5:
        print("   -> initial MEAN policy stands, but SAMPLING destroys it: reduce initial exploration std "
              "(keep action bounds unchanged)")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
