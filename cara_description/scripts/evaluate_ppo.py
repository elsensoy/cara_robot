#!/usr/bin/env python3
"""U21 evaluation -- mirrors evaluate_policy.py's metrics/verdict logic
(imported directly, not reimplemented) but for a PPO checkpoint saved by
train_ppo.py. Needs torch -- run with .venv-rl, not the main .venv.

Same milestone bar as U20: repeatable forward stepping, not reward alone,
and a fall disqualifies a "STEPPING" verdict regardless of what its
contact-pattern numbers look like (see evaluate_policy.summarize).
"""

from __future__ import annotations

import argparse
import json

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from evaluate_policy import summarize  # reused as-is, no torch dependency in that module

try:
    import torch
    from train_ppo import ActorCritic, RunningNorm
except ImportError as e:
    raise SystemExit(f"evaluate_ppo.py needs torch (run with .venv-rl). Original error: {e}")


def load_ppo(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    env_config = ckpt["env_config"]
    obs_dim = len(ckpt["obs_norm"]["mean"])
    act_dim = ckpt["model_state"]["actor_mean.4.bias"].shape[0]
    agent = ActorCritic(obs_dim, act_dim)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    obs_norm = RunningNorm(obs_dim)
    obs_norm.load_state_dict(ckpt["obs_norm"])
    return agent, obs_norm, env_config, ckpt.get("args", {})


def run_episode(env, agent, obs_norm, max_steps, record_gif_every=None):
    import numpy as np
    obs, _ = env.reset()
    frames = []
    renderer = None
    if record_gif_every is not None:
        import mujoco
        renderer = mujoco.Renderer(env.model, height=160, width=240)

    log = {"vx": [], "tilt_deg": [], "qpos_err": [], "torque_frac": [],
           "foot_touch_l": [], "foot_touch_r": [], "foot_z_l": [], "foot_z_r": [],
           "pelvis_x": []}
    fell = False
    steps = 0
    total_reward = 0.0
    for t in range(max_steps):
        obs_n = obs_norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            mean = agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
        action = mean.numpy()  # deterministic (mean) action for evaluation, not a sampled one
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        log["vx"].append(info["vx"])
        log["tilt_deg"].append(info["tilt_deg"])
        log["qpos_err"].append(info["qpos_err"])
        log["torque_frac"].append(info["torque_frac"])
        log["foot_touch_l"].append(info["foot_touch"]["l_foot_collision"])
        log["foot_touch_r"].append(info["foot_touch"]["r_foot_collision"])
        log["foot_z_l"].append(info["foot_z"]["l_foot_collision"])
        log["foot_z_r"].append(info["foot_z"]["r_foot_collision"])
        log["pelvis_x"].append(float(env.data.qpos[0]))
        if renderer is not None and t % record_gif_every == 0:
            renderer.update_scene(env.data)
            frames.append(renderer.render().copy())
        if terminated:
            fell = True
            break
        if truncated:
            break
    if renderer is not None:
        renderer.close()
    return dict(fell=fell, steps=steps, total_reward=total_reward, frames=frames, **log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="path to a .pt saved by train_ppo.py")
    ap.add_argument("--episode-seconds", type=float, default=6.0)
    ap.add_argument("--gif-out", default=None)
    ap.add_argument("--gif-fps", type=float, default=10.0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    agent, obs_norm, env_config, train_args = load_ppo(args.checkpoint)
    print(f"loaded PPO checkpoint: obs_norm.count={obs_norm.count:.0f}")
    print(f"training env config: {env_config}")

    fell_count = 0
    n_trials = 0
    report = {}
    for name, desired_vx in (("standing (vx=0)", 0.0), ("forward walk (vx=trained target)",
                                                          env_config["desired_vx"])):
        cfg = CaraWalkEnvConfig(**{**env_config, "episode_seconds": args.episode_seconds,
                                    "desired_vx": desired_vx})
        env = CaraWalkEnv(cfg)
        max_steps = int(round(args.episode_seconds * cfg.control_hz))
        record_every = None
        if args.gif_out and desired_vx != 0.0:
            record_every = max(1, round(cfg.control_hz / args.gif_fps))
        result = run_episode(env, agent, obs_norm, max_steps, record_gif_every=record_every)
        n_trials += 1
        fell_count += int(result["fell"])
        report[name] = summarize(name, result, cfg.control_hz, max_steps)
        if record_every is not None and result["frames"]:
            import gif_writer
            gif_writer.write_gif(args.gif_out, result["frames"], fps=args.gif_fps)
            print(f"\n  saved {len(result['frames'])} frames -> {args.gif_out}")
        elif args.gif_out and desired_vx != 0.0:
            print("\n  GIF not written: episode produced no frames")

    print(f"\n=== fall rate across {n_trials} eval episodes: {fell_count}/{n_trials} ===")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "env_config": env_config, "train_args": train_args,
                       "episode_seconds": args.episode_seconds,
                       "fall_rate": f"{fell_count}/{n_trials}", "scenarios": report}, f, indent=2)
        print(f"eval report saved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
