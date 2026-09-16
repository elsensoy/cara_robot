#!/usr/bin/env python3
"""U25 -- small, smooth actuator-response probes. NOT a training run.

The question this answers: does Cara respond predictably to reasonable
commands? Per instruction, this must be settled BEFORE justifying more
training budget -- U24 improved survival/smoothness but the underlying
question (actuator/controller path OK, or not) was still open.

Four test classes, same model / resets / control timing as every other
script in this project (CaraWalkEnv, control_hz=50, the default
cara_full_body.yaml dynamic MJCF):

  1. Zero-action hold      -- per-joint baseline saturation & tracking error
  2. Fast smooth excursion -- ramp to a target offset and back, ~100ms ramps
  3. Slow smooth excursion -- same shape, ~400ms ramps (isolates command
                               SPEED sensitivity from pose/loading)
  4. U24 comparison        -- replay the actual trained (U24) policy and log
                               the SAME per-joint metrics, to see whether its
                               commands exceed the range/speed tested above

Tested one joint at a time (all other 11 actuators held at raw action=0,
i.e. nominal) for l_ankle_pitch, l_knee_pitch, l_hip_pitch -- picked per
instruction ("start with individual ankle, knee, and hip pitch joints").
Excursion amplitude 0.4 (normalized action units) was chosen to bracket
U24's own mean action magnitude (0.3811, from the U23/U24 diagnostics) --
conservative and comparable, not arbitrary.

Needs torch only for test 4 (replaying the U24 policy); tests 1-3 are
plain env + numpy and would run under the main .venv too, but this file is
run under .venv-rl throughout for consistency with the rest of the U21+
scripts.
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from cara_env import CaraWalkEnv, CaraWalkEnvConfig

TEST_JOINTS = ["l_ankle_pitch", "l_knee_pitch", "l_hip_pitch"]
AMPLITUDE = 0.4  # normalized action units -- brackets U24's mean |action|=0.3811
FAST_RAMP_STEPS = 5     # 100ms
SLOW_RAMP_STEPS = 20    # 400ms
HOLD_STEPS = 10         # 200ms at peak
SETTLE_STEPS = 30       # 600ms after returning to nominal, to check settling
SAT_THRESHOLD = 0.95    # fraction of forcerange counted as "saturated"


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 3 * u ** 2 - 2 * u ** 3  # cubic, zero endpoint velocity -- same shape used in dcm_walk.py


def build_trajectory(amplitude, t_ramp, t_hold, t_settle):
    total = 2 * t_ramp + t_hold + t_settle
    traj = np.zeros(total)
    for t in range(total):
        if t < t_ramp:
            traj[t] = amplitude * smoothstep(t / t_ramp)
        elif t < t_ramp + t_hold:
            traj[t] = amplitude
        elif t < 2 * t_ramp + t_hold:
            u = (t - t_ramp - t_hold) / t_ramp
            traj[t] = amplitude * (1.0 - smoothstep(u))
        else:
            traj[t] = 0.0
    return traj


def per_joint_actuator_ids(env, joint_names):
    return {n: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in joint_names}


def longest_run(bool_array):
    best = cur = 0
    for v in bool_array:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def run_probe(env, action_sequence_fn, total_steps, probe_joints, aids, seed=0):
    """action_sequence_fn(t) -> length-12 raw action array for step t."""
    obs, _ = env.reset(seed=seed)
    per_joint = {n: dict(target=[], actual_pos=[], actual_vel=[], force=[], saturated=[])
                 for n in probe_joints}
    tilt = []
    fell = False
    steps = 0
    for t in range(total_steps):
        raw_action = action_sequence_fn(t)
        obs, reward, terminated, truncated, info = env.step(raw_action)
        steps += 1
        for n in probe_joints:
            ji = env.jidx[n]
            aid = aids[n]
            action_used = float(np.clip(raw_action[ji], -1.0, 1.0))
            target = float(env.nominal[ji] + action_used * env.offset_scale[ji])
            target = float(np.clip(target, env.lo[ji], env.hi[ji]))
            force = float(env.data.actuator_force[aid])
            forcerange = float(env.model.actuator_forcerange[aid][1])
            per_joint[n]["target"].append(target)
            per_joint[n]["actual_pos"].append(float(env.data.qpos[7 + ji]))
            per_joint[n]["actual_vel"].append(float(env.data.qvel[6 + ji]))
            per_joint[n]["force"].append(force)
            per_joint[n]["saturated"].append(abs(force) >= SAT_THRESHOLD * forcerange)
        tilt.append(info["tilt_deg"])
        if terminated:
            fell = True
            break
        if truncated:
            break
    return dict(fell=fell, steps=steps, per_joint=per_joint, tilt=tilt)


def summarize_joint(pj, settle_last_n=10):
    target = np.array(pj["target"])
    actual = np.array(pj["actual_pos"])
    vel = np.array(pj["actual_vel"])
    force = np.array(pj["force"])
    sat = np.array(pj["saturated"])
    err = np.abs(actual - target)
    n = len(target)
    settle_err = float(np.mean(err[-settle_last_n:])) if n >= settle_last_n else float(np.mean(err))
    return dict(
        mean_tracking_error_rad=float(np.mean(err)), max_tracking_error_rad=float(np.max(err)),
        return_to_target_error_rad=settle_err,
        saturation_fraction=float(np.mean(sat)), saturation_longest_run_steps=int(longest_run(sat)),
        saturation_longest_run_ms=int(longest_run(sat)) * 20,
        max_abs_force=float(np.max(np.abs(force))), mean_abs_vel=float(np.mean(np.abs(vel))),
        max_abs_vel=float(np.max(np.abs(vel))),
    )


def print_joint_summary(label, name, s):
    print(f"  [{name}] tracking err: mean={s['mean_tracking_error_rad']:.4f} "
          f"max={s['max_tracking_error_rad']:.4f} rad  "
          f"return-to-target err: {s['return_to_target_error_rad']:.4f} rad  "
          f"saturation: {s['saturation_fraction']:.1%} of steps, "
          f"longest run {s['saturation_longest_run_ms']}ms  "
          f"peak|vel|={s['max_abs_vel']:.2f} rad/s")


def test_zero_hold(env, aids, total_steps=100, seed=0):
    print("\n=== Test 1: zero-action hold (baseline) ===")
    fn = lambda t: np.zeros(env.n_act)
    r = run_probe(env, fn, total_steps, TEST_JOINTS, aids, seed=seed)
    print(f"  steps={r['steps']} fell={r['fell']} peak_tilt={max(r['tilt']):.2f}deg")
    out = {}
    for n in TEST_JOINTS:
        s = summarize_joint(r["per_joint"][n])
        print_joint_summary("zero_hold", n, s)
        out[n] = s
    return out


def test_excursion(env, joint_name, aids, t_ramp, label, seed=0):
    total_steps = 2 * t_ramp + HOLD_STEPS + SETTLE_STEPS
    traj = build_trajectory(AMPLITUDE, t_ramp, HOLD_STEPS, SETTLE_STEPS)
    ji_all = env.jidx

    def fn(t):
        a = np.zeros(env.n_act)
        a[ji_all[joint_name]] = traj[t]
        return a

    r = run_probe(env, fn, total_steps, [joint_name], aids, seed=seed)
    print(f"\n=== Test [{label}] {joint_name}: amplitude={AMPLITUDE}, ramp={t_ramp*20}ms, "
          f"hold={HOLD_STEPS*20}ms ===")
    print(f"  steps={r['steps']}/{total_steps} fell={r['fell']} peak_tilt={max(r['tilt']):.2f}deg")
    s = summarize_joint(r["per_joint"][joint_name])
    print_joint_summary(label, joint_name, s)
    if r["fell"]:
        print(f"  NOTE: fell during this single-joint excursion -- examine whether {joint_name} "
              f"tracked smoothly before balance was lost (see mean/max tracking error above) before "
              f"calling this an actuator defect.")
    return s


def test_u24_comparison(env, aids, checkpoint_path, seed=0):
    import torch
    from train_ppo import ActorCritic, RunningNorm

    print("\n=== Test 4: U24 trained policy (deterministic mean) -- same per-joint metrics ===")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    obs_dim = len(ckpt["obs_norm"]["mean"])
    act_dim = env.n_act
    agent = ActorCritic(obs_dim, act_dim)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    norm = RunningNorm(obs_dim)
    norm.load_state_dict(ckpt["obs_norm"])

    obs_holder = {}

    def fn(t):
        obs = obs_holder["obs"]
        obs_n = norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            mean = agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
        return mean.numpy()

    # run_probe calls env.step with the action from fn(t), but fn needs the
    # CURRENT obs, which run_probe owns internally -- so re-implement the
    # loop here directly rather than forcing run_probe's fn signature to
    # carry state it wasn't designed for.
    obs, _ = env.reset(seed=seed)
    obs_holder["obs"] = obs
    per_joint = {n: dict(target=[], actual_pos=[], actual_vel=[], force=[], saturated=[]) for n in TEST_JOINTS}
    action_absmean, action_delta_absmean = [], []
    prev_action = np.zeros(act_dim)
    tilt = []
    fell = False
    steps = 0
    for t in range(200):
        raw_action = fn(t)
        action_absmean.append(float(np.mean(np.abs(raw_action))))
        action_delta_absmean.append(float(np.mean(np.abs(raw_action - prev_action))))
        prev_action = raw_action
        obs, reward, terminated, truncated, info = env.step(raw_action)
        obs_holder["obs"] = obs
        steps += 1
        for n in TEST_JOINTS:
            ji = env.jidx[n]
            aid = aids[n]
            action_used = float(np.clip(raw_action[ji], -1.0, 1.0))
            target = float(np.clip(env.nominal[ji] + action_used * env.offset_scale[ji], env.lo[ji], env.hi[ji]))
            force = float(env.data.actuator_force[aid])
            forcerange = float(env.model.actuator_forcerange[aid][1])
            per_joint[n]["target"].append(target)
            per_joint[n]["actual_pos"].append(float(env.data.qpos[7 + ji]))
            per_joint[n]["actual_vel"].append(float(env.data.qvel[6 + ji]))
            per_joint[n]["force"].append(force)
            per_joint[n]["saturated"].append(abs(force) >= SAT_THRESHOLD * forcerange)
        tilt.append(info["tilt_deg"])
        if terminated:
            fell = True
            break
        if truncated:
            break

    print(f"  steps={steps}/200 fell={fell} peak_tilt={max(tilt):.2f}deg  "
          f"whole-body action |mean|={np.mean(action_absmean):.4f}  "
          f"action delta step-to-step={np.mean(action_delta_absmean):.4f}")
    out = {}
    for n in TEST_JOINTS:
        s = summarize_joint(per_joint[n])
        print_joint_summary("u24_replay", n, s)
        out[n] = s
    out["_whole_body"] = dict(steps=steps, fell=fell, peak_tilt=float(max(tilt)),
                               action_absmean=float(np.mean(action_absmean)),
                               action_delta_absmean=float(np.mean(action_delta_absmean)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trained-checkpoint", default="../runs/u24_ppo_lowstd/ppo_lowstd_seed0.pt")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    cfg = CaraWalkEnvConfig(episode_seconds=4.0, desired_vx=0.0)  # same stage-0 config as U22-U24
    env = CaraWalkEnv(cfg)
    aids = per_joint_actuator_ids(env, TEST_JOINTS)

    report = {}
    report["1_zero_hold"] = test_zero_hold(env, aids)

    report["2_fast_excursion"] = {}
    for j in TEST_JOINTS:
        report["2_fast_excursion"][j] = test_excursion(env, j, aids, FAST_RAMP_STEPS, "fast")

    report["3_slow_excursion"] = {}
    for j in TEST_JOINTS:
        report["3_slow_excursion"][j] = test_excursion(env, j, aids, SLOW_RAMP_STEPS, "slow")

    report["4_u24_replay"] = test_u24_comparison(env, aids, args.trained_checkpoint)

    print("\n=== decision readout ===")
    for j in TEST_JOINTS:
        zero = report["1_zero_hold"][j]
        fast = report["2_fast_excursion"][j]
        slow = report["3_slow_excursion"][j]
        u24 = report["4_u24_replay"][j]
        print(f"\n{j}:")
        print(f"  zero-hold saturation: {zero['saturation_fraction']:.1%}  "
              f"(does zero action ALREADY saturate this joint? {'YES' if zero['saturation_fraction'] > 0.05 else 'no'})")
        print(f"  fast excursion:  tracking err mean={fast['mean_tracking_error_rad']:.4f} rad  "
              f"return-to-target={fast['return_to_target_error_rad']:.4f} rad  "
              f"saturation={fast['saturation_fraction']:.1%}")
        print(f"  slow excursion:  tracking err mean={slow['mean_tracking_error_rad']:.4f} rad  "
              f"return-to-target={slow['return_to_target_error_rad']:.4f} rad  "
              f"saturation={slow['saturation_fraction']:.1%}")
        print(f"  U24 replay:      tracking err mean={u24['mean_tracking_error_rad']:.4f} rad  "
              f"peak|vel|={u24['max_abs_vel']:.2f} rad/s  saturation={u24['saturation_fraction']:.1%}  "
              f"(fast test peak|vel|={fast['max_abs_vel']:.2f}, slow={slow['max_abs_vel']:.2f} rad/s -- "
              f"{'U24 EXCEEDS the tested speed range' if u24['max_abs_vel'] > 1.3*fast['max_abs_vel'] else 'U24 is within the tested speed range'})")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
