#!/usr/bin/env python3
"""U20 reward audit -- run BEFORE any new training experiment, per review
guidance: "adding an alive bonus and increasing velocity weight are
reward-design changes, not automatically complete fixes." This computes the
actual discounted return CaraWalkEnv's current reward function assigns to
three canonical outcomes, over the real episode horizon:

  A. standing        -- zero action, held for the full episode
  B. brief-then-fall  -- real simulated rollout that makes some forward
                         progress and then falls (the hand-crafted open-loop
                         swing from the U20 actuator-tracking check)
  C. sustained target-speed -- the INTENDED behavior. No working controller
                         produces this yet (that is the unsolved problem),
                         so this is computed as an idealized upper bound:
                         plug vx=desired_vx exactly, perfectly upright,
                         and a modest realistic action magnitude directly
                         into the reward formula for every step of the same
                         horizon. It is not a simulated rollout, and is
                         labeled as such in the output -- but it gives the
                         reward function every benefit of the doubt, so if
                         it still doesn't clearly dominate, that is a real
                         problem with the reward, not a simulation artifact.

Pass/fail: the intended behavior (C) must rank strictly above both A and B,
under both discounted (gamma=0.99, the value planned for PPO) and
undiscounted (gamma=1.0, what ARS effectively used) returns.
"""

from __future__ import annotations

import argparse

from cara_env import CaraWalkEnv, CaraWalkEnvConfig


def discounted_return(rewards, gamma):
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


def scenario_standing(env, max_steps):
    import numpy as np
    env.reset()
    rewards = []
    for _ in range(max_steps):
        _, r, terminated, truncated, _ = env.step(np.zeros(env.n_act))
        rewards.append(r)
        if terminated or truncated:
            break
    return rewards


def scenario_brief_then_fall(env, max_steps):
    """The same hand-crafted, non-balancing alternating swing used in the
    U20 actuator-tracking check (see docs/rl_environment_notes.md) -- a real
    simulated rollout that makes real forward progress before toppling, not
    a hypothetical."""
    import numpy as np
    env.reset()
    amp = np.array([0, 0, 0.5, 0.4, -0.3, 0] * 2)
    freq = 0.5
    rewards = []
    x0 = float(env.data.qpos[0])
    for t in range(max_steps):
        phase = 2 * np.pi * freq * (t / env.cfg.control_hz)
        a = np.zeros(env.n_act)
        a[:6] = amp[:6] * np.sin(phase)
        a[6:] = amp[6:] * np.sin(phase + np.pi)
        a = np.clip(a, -1, 1)
        _, r, terminated, truncated, info = env.step(a)
        rewards.append(r)
        if terminated or truncated:
            break
    fwd = float(env.data.qpos[0]) - x0
    return rewards, fwd


def scenario_sustained_ideal(env, max_steps):
    """Idealized, NOT simulated: the reward formula evaluated at
    vx=desired_vx exactly, projected_gravity_z=-1 exactly (perfectly
    upright), and a modest realistic action (rms 0.3, smoothly varying so
    the action-rate penalty is small too). This is the best case the current
    reward function could ever pay a real gait -- deliberately generous."""
    cfg = env.cfg
    r_alive = 1.0
    r_vel = 0.0  # zero error, by construction
    r_upright = 0.0  # projected_gravity_z + 1 = -1 + 1 = 0
    action_rms = 0.3
    r_effort = -(action_rms ** 2)
    r_rate = -((action_rms * 0.3) ** 2)  # smooth gait: small step-to-step change
    r_collision = 0.0
    per_step = (cfg.w_alive * r_alive + cfg.w_vel * r_vel + cfg.w_upright * r_upright
                + cfg.w_effort * r_effort + cfg.w_action_rate * r_rate
                + cfg.w_collision * r_collision)
    return [per_step] * max_steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-seconds", type=float, default=10.0,
                     help="the actual episode horizon to audit under (default: CaraWalkEnvConfig's own default)")
    ap.add_argument("--gamma", type=float, default=0.99, help="discount factor planned for PPO")
    ap.add_argument("--w-vel", type=float, default=None, help="override w_vel (for before/after comparisons)")
    args = ap.parse_args(argv)

    overrides = {} if args.w_vel is None else {"w_vel": args.w_vel}
    cfg = CaraWalkEnvConfig(episode_seconds=args.episode_seconds, **overrides)
    env = CaraWalkEnv(cfg)
    max_steps = int(round(args.episode_seconds * cfg.control_hz))
    print(f"reward weights: w_alive={cfg.w_alive} w_vel={cfg.w_vel} w_upright={cfg.w_upright} "
          f"w_effort={cfg.w_effort} w_action_rate={cfg.w_action_rate} w_collision={cfg.w_collision}")
    print(f"desired_vx={cfg.desired_vx}  horizon={max_steps} steps ({args.episode_seconds}s)  "
          f"gamma={args.gamma}\n")

    rewards_a = scenario_standing(env, max_steps)
    rewards_b, fwd_b = scenario_brief_then_fall(env, max_steps)
    rewards_c = scenario_sustained_ideal(env, max_steps)

    rows = [
        ("A. standing", rewards_a, None),
        ("B. brief-forward-then-fall", rewards_b, fwd_b),
        ("C. sustained target-speed (IDEALIZED, not simulated)", rewards_c, None),
    ]

    results = {}
    print(f"{'scenario':<55} {'steps':>6} {'undisc.(g=1.0)':>15} {'disc.(g=%.2f)' % args.gamma:>15}")
    for name, rewards, fwd in rows:
        g1 = discounted_return(rewards, 1.0)
        gd = discounted_return(rewards, args.gamma)
        results[name] = (len(rewards), g1, gd)
        extra = f"  fwd={fwd:+.3f}m" if fwd is not None else ""
        print(f"{name:<55} {len(rewards):>6} {g1:>15.2f} {gd:>15.2f}{extra}")

    order_g1 = sorted(results, key=lambda k: -results[k][1])
    order_gd = sorted(results, key=lambda k: -results[k][2])
    intended = rows[2][0]
    standing = rows[0][0]
    print(f"\nranking (undiscounted): {' > '.join(order_g1)}")
    print(f"ranking (discounted):   {' > '.join(order_gd)}")

    ranked_top = order_g1[0] == intended and order_gd[0] == intended

    # Ranking alone is not enough: a technically-correct order with a thin
    # margin over the safe "standing" local optimum is still an attractor
    # for a noisy gradient-free search (this is exactly why w_vel=1.0 got
    # stuck on standing in the first ARS run, despite C nominally
    # outranking A there too -- see the printed comparison below).
    c_gd = results[intended][2]
    a_gd = results[standing][2]
    margin = (c_gd - a_gd) / abs(c_gd) if c_gd else float("nan")
    print(f"\nmargin of intended over standing (discounted): "
          f"{c_gd:.2f} vs {a_gd:.2f}  ->  {margin:.1%} of C's return")
    thin_margin = margin < 0.5
    if thin_margin:
        print("  WARNING: margin under 50% -- standing is close enough to the intended "
              "return that a noisy gradient-free search can plausibly get stuck there "
              "even though the ranking is technically correct. Treat this as a soft risk "
              "signal, not just the pass/fail below.")

    ok = ranked_top and not thin_margin
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} -- intended behavior "
          f"{'ranks highest with an adequate margin' if ok else 'does not both rank highest AND clear standing by a safe margin'}.")
    if not ok:
        print("Do not start a new training run against this reward until this passes.")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
