#!/usr/bin/env python3
"""U32 -- demonstration-validation experiment. NOT a training run.

Replays the U10/U11 quasi-static stepping controller (scripts/gait.py,
unmodified) through CaraWalkEnv's actual RL interface: the same full-body
model, position actuators, 50Hz action rate, and action bounds every PPO
run in this project used. gait.py drives data.ctrl directly, every physics
substep (500Hz) -- captured via a non-invasive monkeypatch of mujoco.mj_step
(gait.py's own source is untouched, preserving it as a generator/validator)
and downsampled to CaraWalkEnv's 50Hz control rate by taking the ctrl value
active at the start of every 10th physics step.

Commanded targets are converted to normalized actions via
    a_j = (q_target,j - q_nominal,j) / offset_scale_j
using CaraWalkEnv's OWN nominal pose and offset_scale (not gait.py's
values) -- out-of-range |a_j| > 1 are counted and reported explicitly, not
silently clipped and hidden.

The converted action sequence is then replayed through a REAL CaraWalkEnv
instance -- its own termination check applies (a "successful" gait.py run
does not guarantee survival through this different interface/dynamics/
gain path) -- and scored under the CURRENT (v4) reward, compared against
zero action and an existing trained ('sliding') policy over the SAME
command and episode duration.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cara_env import CaraWalkEnv, CaraWalkEnvConfig

DEFAULT_CONFIG = "../config/cara_full_body.yaml"


def record_gait_ctrl(config_path, n_steps):
    """Runs gait.py's own walk() unmodified, recording data.ctrl right
    before every mujoco.mj_step() call via a monkeypatch -- gait.py's
    source is never touched."""
    import mujoco
    original_mj_step = mujoco.mj_step
    recorded = []

    def recording_mj_step(model, data):
        recorded.append(np.array(data.ctrl, dtype=np.float64).copy())
        original_mj_step(model, data)

    mujoco.mj_step = recording_mj_step
    try:
        import gait
        exit_code = gait.run(config_path, n_steps, view=False, json_path=None, baseline_path=None)
    finally:
        mujoco.mj_step = original_mj_step
    return recorded, exit_code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--n-steps", type=int, default=2, help="gait.py steps to record (default 2: minimum to show alternation)")
    ap.add_argument("--desired-vx", type=float, default=0.03)
    ap.add_argument("--compare-checkpoint", default=None, help="an existing trained ('sliding') policy .pt to compare against")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    print(f"=== recording gait.py's own controller (unmodified), n_steps={args.n_steps} ===")
    recorded, gait_exit = record_gait_ctrl(args.config, args.n_steps)
    n_physics_steps = len(recorded)
    print(f"gait.py's own exit code: {gait_exit} (0 = its own milestone MET)")
    print(f"recorded {n_physics_steps} physics substeps")

    # ---- build a CaraWalkEnv to get nominal/offset_scale/dt/substeps ----
    probe_cfg = CaraWalkEnvConfig(episode_seconds=1.0, desired_vx=args.desired_vx)
    probe_env = CaraWalkEnv(probe_cfg)
    substeps = probe_env.substeps
    n_control_steps = n_physics_steps // substeps
    demo_duration_s = n_control_steps / probe_env.cfg.control_hz
    print(f"CaraWalkEnv substeps/control-step: {substeps}  ->  {n_control_steps} control steps "
          f"({demo_duration_s:.1f}s at {probe_env.cfg.control_hz:.0f}Hz)")

    # ---- downsample to 50Hz and convert to normalized actions ----
    targets_50hz = [recorded[i * substeps] for i in range(n_control_steps)]
    jn = probe_env.jn
    nominal = probe_env.nominal
    offset_scale = probe_env.offset_scale
    actions_raw = []
    out_of_range = []
    for k, target in enumerate(targets_50hz):
        # target is in MJCF actuator order; CaraWalkEnv's jn is the SAME
        # order (both derived from leg_model.actuated_joint_names(spec)).
        a = (target - nominal) / offset_scale
        actions_raw.append(a)
        for j, name in enumerate(jn):
            if abs(a[j]) > 1.0:
                out_of_range.append(dict(control_step=k, joint=name, normalized_action=float(a[j]),
                                          raw_target=float(target[j])))
    actions_raw = np.array(actions_raw)
    actions_clipped = np.clip(actions_raw, -1.0, 1.0)

    print(f"\n=== action-range check (a_j = (q_target-q_nominal)/offset_scale) ===")
    print(f"out-of-range (|a_j|>1) entries: {len(out_of_range)} of {actions_raw.size} "
          f"({len(out_of_range)/actions_raw.size:.2%})")
    if out_of_range:
        by_joint = {}
        for r in out_of_range:
            by_joint.setdefault(r["joint"], []).append(r["normalized_action"])
        for j, vals in by_joint.items():
            print(f"  {j}: {len(vals)} entries, range [{min(vals):.2f}, {max(vals):.2f}]")
    else:
        print("  none -- every commanded target fell within the actor's own [-1,1] action bounds.")

    # ---- replay through CaraWalkEnv (its own termination applies) ----
    def replay(env, action_fn, seed=0, max_steps=None):
        obs, _ = env.reset(seed=seed)
        log = dict(reward=[], components=[], vx=[], tilt=[], touch_l=[], touch_r=[],
                   z_l=[], z_r=[], pelvis_x=[], qpos_err=[], torque_frac=[])
        fell = False
        steps = 0
        limit = n_control_steps if max_steps is None else max_steps
        for k in range(limit):
            action = action_fn(k)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            log["reward"].append(reward)
            log["components"].append(info["reward_components"])
            log["vx"].append(info["vx"])
            log["tilt"].append(info["tilt_deg"])
            log["touch_l"].append(info["foot_touch"]["l_foot_collision"])
            log["touch_r"].append(info["foot_touch"]["r_foot_collision"])
            log["z_l"].append(info["foot_z"]["l_foot_collision"])
            log["z_r"].append(info["foot_z"]["r_foot_collision"])
            log["pelvis_x"].append(float(env.data.qpos[0]))
            log["qpos_err"].append(info["qpos_err"])
            log["torque_frac"].append(info["torque_frac"])
            if terminated:
                fell = True
                break
            if truncated:
                break
        return dict(steps=steps, fell=fell, log=log)

    # w_action_rate=0.3 matches U26 onward (every trained checkpoint compared
    # against here was trained under that weight, not CaraWalkEnvConfig's
    # own bare default of 0.01) -- an earlier pass here used the stale
    # default, a real (if numerically small) inconsistency, fixed here.
    cfg = CaraWalkEnvConfig(episode_seconds=demo_duration_s + 1.0, desired_vx=args.desired_vx,
                             w_action_rate=0.3)

    print(f"\n=== replaying the demonstration through CaraWalkEnv (desired_vx={args.desired_vx}) ===")
    env_demo = CaraWalkEnv(cfg)
    demo_result = replay(env_demo, lambda k: actions_clipped[k])
    print(f"  steps survived: {demo_result['steps']}/{n_control_steps}  fell={demo_result['fell']}")

    def genuine_swings(touch):
        run = 0
        n = 0
        for v in touch:
            if not v:
                run += 1
            else:
                if run >= 3:
                    n += 1
                run = 0
        if run >= 3:
            n += 1
        return n

    log = demo_result["log"]
    n = demo_result["steps"]
    swings_l = genuine_swings(log["touch_l"])
    swings_r = genuine_swings(log["touch_r"])
    fwd_dist = log["pelvis_x"][-1] - log["pelvis_x"][0] if n else 0.0
    achieved_vx = float(np.mean(log["vx"])) if n else 0.0
    total_return = float(np.sum(log["reward"]))

    print(f"  genuine swing phases (>=60ms airborne): L={swings_l} R={swings_r}")
    print(f"  forward distance: {fwd_dist:+.3f}m over {n/50.0:.1f}s  achieved_vx(mean)={achieved_vx:+.4f} m/s "
          f"(commanded {args.desired_vx} m/s)")
    print(f"  DISCLOSURE: demonstrated speed is {achieved_vx/args.desired_vx*100 if args.desired_vx else float('nan'):.1f}% "
          f"of the commanded {args.desired_vx} m/s -- NOT assumed to match.")
    print(f"  joint tracking error: mean={np.mean(log['qpos_err']):.4f} rad  "
          f"actuator saturation: mean={np.mean(log['torque_frac']):.1%} max={np.max(log['torque_frac']):.1%}")
    print(f"  peak tilt: {np.max(log['tilt']):.1f} deg")
    print(f"  TOTAL RETURN (current reward v4, desired_vx={args.desired_vx}): {total_return:.2f} over {n} steps")

    valid_demo = (not demo_result["fell"]) and swings_l >= 1 and swings_r >= 1 and fwd_dist > 0.0
    print(f"\n  VALID DEMONSTRATION: {valid_demo} "
          f"(survived={not demo_result['fell']}, both feet swung genuinely, net forward progress)")

    # ---- comparison baselines, SAME command + SAME duration ----
    # max_steps=n truncates to the DEMO's own survived length -- without
    # this, zero action and the sliding policy (which don't fall) run the
    # full n_control_steps (2365) instead of matching the demo's 764, which
    # is exactly the accounting bug a reviewer caught: it produced a return
    # (1654.66) impossible for a 764-step episode under this reward.
    print(f"\n=== comparison: zero action, over the SAME {n} control steps / {n/50.0:.1f}s, desired_vx={args.desired_vx} ===")
    env_zero = CaraWalkEnv(cfg)
    zero_result = replay(env_zero, lambda k: np.zeros(probe_env.n_act), max_steps=n)
    zero_return = float(np.sum(zero_result["log"]["reward"]))
    print(f"  zero action: steps={zero_result['steps']}/{n}  fell={zero_result['fell']}  return={zero_return:.2f}")

    sliding_return = None
    if args.compare_checkpoint:
        import torch
        from train_ppo import ActorCritic, RunningNorm
        ckpt = torch.load(args.compare_checkpoint, map_location="cpu", weights_only=False)
        obs_dim = len(ckpt["obs_norm"]["mean"])
        act_dim = ckpt["model_state"]["actor_mean.4.bias"].shape[0]
        agent = ActorCritic(obs_dim, act_dim)
        agent.load_state_dict(ckpt["model_state"])
        agent.eval()
        norm = RunningNorm(obs_dim)
        norm.load_state_dict(ckpt["obs_norm"])

        env_slide = CaraWalkEnv(cfg)
        obs_holder = {}

        def sliding_fn(k):
            obs_n = norm.normalize(obs_holder["obs"].astype("float64"))
            with torch.no_grad():
                mean = agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
            return mean.numpy()

        obs0, _ = env_slide.reset(seed=0)
        obs_holder["obs"] = obs0
        sl_log = dict(reward=[])
        fell = False
        steps = 0
        for k in range(n):
            a = sliding_fn(k)
            obs, r, term, trunc, info = env_slide.step(a)
            obs_holder["obs"] = obs
            sl_log["reward"].append(r)
            steps += 1
            if term:
                fell = True
                break
            if trunc:
                break
        sliding_return = float(np.sum(sl_log["reward"]))
        print(f"  sliding policy ({args.compare_checkpoint}): steps={steps}/{n}  fell={fell}  return={sliding_return:.2f}")

    # Weighted, not raw: info["reward_components"] holds the RAW (pre-weight)
    # per-term values, e.g. r_effort = -mean(action^2) directly, NOT
    # w_effort*r_effort. Printing those raw sums under a bare "breakdown"
    # header is exactly the kind of thing that reads as "effort dominates
    # the return" when its actual (weighted) contribution is two orders of
    # magnitude smaller -- shown both ways here so the total is reconstructible.
    weights = dict(alive=cfg.w_alive, vel=cfg.w_vel, upright=cfg.w_upright,
                    effort=cfg.w_effort, rate=cfg.w_action_rate, collision=cfg.w_collision)
    print(f"\n=== reward component breakdown (demo, summed over {n} non-terminal steps) ===")
    print(f"  {'term':10s}  {'raw sum':>10s}  {'weight':>7s}  {'weighted':>10s}")
    comp_sums = {}
    weighted_total = 0.0
    for key in ("alive", "vel", "upright", "effort", "rate", "collision"):
        vals = [c[key] for c in log["components"] if c[key] is not None]
        comp_sums[key] = float(np.sum(vals)) if vals else 0.0
        weighted = comp_sums[key] * weights[key]
        weighted_total += weighted
        print(f"  {key:10s}  {comp_sums[key]:>10.2f}  {weights[key]:>7.3f}  {weighted:>10.3f}")
    fall_pen = -10.0 if demo_result["fell"] else 0.0
    weighted_total += fall_pen
    print(f"  {'fall_penalty':10s}  {'--':>10s}  {'--':>7s}  {fall_pen:>10.3f}")
    print(f"  reconstructed total (sum of weighted terms + fall penalty): {weighted_total:.2f}  "
          f"(reported TOTAL RETURN above: {total_return:.2f} -- must match)")

    print(f"\n=== decision readout ===")
    print(f"  demo return: {total_return:.2f}   zero-action return: {zero_return:.2f}"
          + (f"   sliding-policy return: {sliding_return:.2f}" if sliding_return is not None else ""))
    if not valid_demo:
        print("  -> INVALID DEMONSTRATION through this interface. Do not train on it; "
              "identify the specific interface/controller dependency that broke.")
    elif total_return > zero_return and (sliding_return is None or total_return > sliding_return):
        print("  -> Stepping succeeds AND earns a better return. "
              "Next: initialize a policy from the demonstration, then fine-tune with PPO.")
    else:
        print("  -> Stepping succeeds but earns LESS than standing/sliding. "
              "Next: fix the demonstrated objective mismatch (likely the speed-tracking penalty "
              "given the disclosed speed mismatch above) before more PPO.")

    if args.out_json:
        out = dict(gait_exit_code=gait_exit, n_physics_steps=n_physics_steps, n_control_steps=n_control_steps,
                   demo_duration_s=demo_duration_s, out_of_range_count=len(out_of_range),
                   out_of_range_entries=out_of_range[:200],
                   valid_demonstration=valid_demo, demo_return=total_return, zero_action_return=zero_return,
                   sliding_policy_return=sliding_return, achieved_vx=achieved_vx, commanded_vx=args.desired_vx,
                   fwd_dist_m=fwd_dist, genuine_swings_l=swings_l, genuine_swings_r=swings_r,
                   demo_fell=demo_result["fell"], reward_components=comp_sums,
                   demo_steps_survived=n, qpos_err_mean=float(np.mean(log["qpos_err"])),
                   torque_frac_mean=float(np.mean(log["torque_frac"])), peak_tilt_deg=float(np.max(log["tilt"])))
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
