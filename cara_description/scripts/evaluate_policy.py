#!/usr/bin/env python3
"""U20 evaluation -- separate from training, per the review guidance: load a
saved ARS policy (from train_ars.py) and run two fixed suites, zero-speed
standing and forward walking, reporting the metrics the roadmap asked for.
Also (best-effort) renders one rollout to an animated GIF, using the
dependency-free writer in gif_writer.py -- the venv has no imageio/opencv/
Pillow and the host has no ffmpeg, so this is a from-scratch stdlib+numpy
encoder, not a wrapped library call.

The milestone this checks for is repeatable forward stepping, not reward:
a policy that slides a foot while always in contact, or leans/falls forward
past the start line, must NOT be reported as "walking" just because it moved.
"""

from __future__ import annotations

import argparse
import json

from cara_env import CaraWalkEnv, CaraWalkEnvConfig
from train_ars import Normalizer


def load_policy(path):
    import numpy as np
    data = np.load(path, allow_pickle=True)
    W = data["W"]
    env_config = json.loads(str(data["env_config"]))
    normalizer = Normalizer(W.shape[1])
    normalizer.mean = data["mean"]
    normalizer.m2 = data["m2"]
    normalizer.n = int(data["n"])
    return W, normalizer, env_config


def run_episode(env, W, normalizer, max_steps, record_gif_every=None):
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
        obs_n = normalizer.normalize(obs.astype("float64"))
        action = np.clip(W @ obs_n, -1.0, 1.0)
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


def summarize(name, result, control_hz, requested_steps):
    import numpy as np
    n = result["steps"]
    duration_s = n / control_hz
    vx = np.array(result["vx"])
    touch_l = np.array(result["foot_touch_l"])
    touch_r = np.array(result["foot_touch_r"])
    # alternation count: how many times support switches between "left only",
    # "right only", "double", "none" -- a real gait alternates repeatedly.
    support = touch_l.astype(int) * 2 + touch_r.astype(int)
    switches = int(np.sum(support[1:] != support[:-1])) if n > 1 else 0
    both_off = float(np.mean((~touch_l) & (~touch_r))) if n else 0.0
    fwd_dist = result["pelvis_x"][-1] - result["pelvis_x"][0] if n else 0.0
    print(f"\n--- {name} ---")
    print(f"  duration: {duration_s:.2f}s ({n} steps)  fell: {result['fell']}")
    print(f"  forward distance (pelvis x): {fwd_dist:+.3f} m")
    print(f"  commanded vx: (see env config)   achieved vx: mean={vx.mean():+.3f} "
          f"std={vx.std():.3f} m/s")
    print(f"  support switches (contact-pattern changes): {switches}  "
          f"both-feet-airborne fraction: {both_off:.1%}")
    print(f"  joint tracking error: mean={np.mean(result['qpos_err']):.4f} "
          f"max={np.max(result['qpos_err']):.4f} rad")
    print(f"  actuator saturation (max |force|/forcerange): "
          f"mean={np.mean(result['torque_frac']):.2%} max={np.max(result['torque_frac']):.2%}")
    # This is the actual "is it stepping" check, not a reward number. A fall
    # (or a rollout that ends well short of the requested duration) is
    # disqualifying on its own: contact-pattern flicker and forward CoM
    # translation both happen freely while a robot topples, and reporting
    # that as "stepping" would be exactly the false positive the milestone
    # definition warns against (sliding, shuffling, or falling forward).
    survived = (not result["fell"]) and n >= 0.8 * requested_steps
    is_stepping = survived and switches >= 4 and fwd_dist > 0.02 and not (both_off > 0.5)
    if not survived:
        verdict = "FELL" if result["fell"] else "TIMED OUT EARLY"
    elif is_stepping:
        verdict = "STEPPING"
    elif fwd_dist > 0.02:
        verdict = "SLIDING/NO-CLEARANCE"
    else:
        verdict = "NOT WALKING"
    reason = {
        "STEPPING": "survived the episode with an alternating contact pattern and forward progress",
        "SLIDING/NO-CLEARANCE": "survived and moved forward, but without alternating single-foot support",
        "NOT WALKING": "survived but made no forward progress",
        "FELL": "episode ended in a fall -- any contact/forward-progress numbers above are from the fall itself, not gait",
        "TIMED OUT EARLY": "episode ended early without a recorded fall (invalid state) -- do not credit this as walking",
    }[verdict]
    print(f"  verdict: {verdict} ({reason})")
    return dict(duration_s=duration_s, fell=result["fell"], fwd_dist=fwd_dist,
                vx_mean=float(vx.mean()) if n else 0.0, switches=switches,
                qpos_err_mean=float(np.mean(result["qpos_err"])) if n else float("nan"),
                torque_frac_max=float(np.max(result["torque_frac"])) if n else float("nan"),
                verdict=verdict)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", help="path to a .npz saved by train_ars.py")
    ap.add_argument("--episode-seconds", type=float, default=6.0)
    ap.add_argument("--gif-out", default=None, help="if given, save a GIF of the forward-walk eval here")
    ap.add_argument("--gif-fps", type=float, default=10.0)
    args = ap.parse_args(argv)

    W, normalizer, env_config = load_policy(args.policy)
    print(f"loaded policy: W shape={W.shape}, normalizer n={normalizer.n}")
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
        result = run_episode(env, W, normalizer, max_steps, record_gif_every=record_every)
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
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
