#!/usr/bin/env python3
"""U19 -- a minimal RL environment around the existing MuJoCo full-body model.

Per the 2026-09-13 strategic redirect: dynamic (DCM) walking is PARKED as of
U18 (real ankle-instability found, contact-gated fix identified but not
wired in -- see docs/single_support_notes.md U18). The project's actual next
goal is RL-readiness, and completing dcm_walk is explicitly optional. This
module is step 3 of that roadmap: expose reset/step around the model with
bounded actions, observations, rewards, and termination -- nothing here
depends on dcm_walk, its footstep schedule, or its controllers. The policy
is meant to learn its own coordination.

Design (matches the roadmap given):
  * Actions: bounded joint-position OFFSETS around the nominal standing pose,
    executed by the model's existing <position> PD actuators (NOT raw
    torques, NOT the torque-ankle scheme from U15/dcm_walk -- this uses the
    plain, byte-identical, already-validated default dynamic MJCF).
  * Observations: joint pos/vel, projected gravity, base angular velocity,
    desired walking velocity, previous action. Anything not available on the
    eventual real hardware is marked below.
  * Reward: velocity tracking + upright, minus effort / action-rate /
    unwanted-collision penalties.
  * Termination: fall (tilt/height), invalid sim state (NaN/Inf). Truncation:
    episode length.
  * No gymnasium/gym dependency required -- this implements the same
    reset()/step() contract (Gymnasium's 5-tuple step return) with a small
    built-in `Box` space so it runs with only mujoco+numpy installed (already
    project dependencies). `pip install gymnasium` and swapping `Box` for
    `gymnasium.spaces.Box` is a drop-in change if a training library wants
    real Gymnasium space objects.

Usage:
    python3 cara_env.py                 # self-test: resets + short rollouts
    python3 cara_env.py --episodes 5 --steps 200 --render
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import leg_model as lm

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_full_body.yaml"))


class Box:
    """Minimal stand-in for gymnasium.spaces.Box -- no gymnasium dependency."""

    def __init__(self, low, high, shape, dtype="float32"):
        import numpy as np
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy()
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng=None):
        import numpy as np
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x):
        import numpy as np
        x = np.asarray(x)
        return bool(np.all(x >= self.low) and np.all(x <= self.high))


@dataclass
class CaraWalkEnvConfig:
    # Bump whenever the reward FORMULA or weights change, and re-run
    # reward_audit.py before trusting a new training run against it. "v1"
    # (pre-alive-bonus) and "v2" (alive bonus, w_vel=1.0) both failed the
    # audit after the fact; "v3" is the first version that passed it. "v4"
    # (U30) fixes a real sign bug in r_upright: the v3 formula rewarded
    # tipping/inversion (verified at synthetic tilts: +0.015 at 10deg,
    # +2.0 fully inverted) opposite of its own "0 upright, -2 tipped"
    # comment. All U20-U29 checkpoints were trained under v3 -- disclosed,
    # not retroactively changed.
    reward_version: str = "v4_upright_sign_fix"
    config_path: str = DEFAULT_CONFIG
    base_pose: str = "stand_nominal"
    control_hz: float = 50.0            # matches the project's stated sim/hardware control rate
    episode_seconds: float = 10.0
    # Provisional -- TODO: replace with a measured/CAD-derived per-joint speed
    # and reach limit once real servos are chosen. Fraction of each joint's
    # OWN [lower, upper] half-range used as the bounded action offset.
    action_range_frac: float = 0.30
    desired_vx: float = 0.10            # m/s, forward -- fixed for now (randomized in step 5)
    # U30: per-episode desired_vx MIXTURE, same pattern as reset_qvel_noise_bands
    # -- a tuple of (probability, vx) pairs, one drawn per reset() call and
    # held CONSTANT for that whole episode (not changed mid-episode). None
    # (default) falls back to the single desired_vx value above, reproducing
    # every prior result exactly -- opt-in, like the perturbation bands.
    desired_vx_bands: tuple | None = None
    fall_tilt_deg: float = 40.0
    fall_height_m: float = 0.15         # pelvis world-Z below this = fallen
    w_alive: float = 1.0
    # NOTE: 1.0 gave standing-still ~0.9/step and perfect tracking ~1.0/step
    # at desired_vx=0.10 (verified via train_ars.py's first run: reward
    # plateaued at "never move" almost immediately, ~180/200-step episode,
    # because the marginal benefit of walking over standing was only
    # 0.1/step against the risk of forfeiting the rest of the episode's
    # alive bonus by falling). 10.0 makes standing-still ~0.0/step (neutral)
    # and perfect tracking ~1.0/step -- a real per-step incentive to move,
    # while a fall (-10, a fixed one-time cost) stays clearly worse than
    # riding out the episode at the neutral baseline. Still provisional --
    # TODO: revisit once a real training run either does or does not learn
    # to step under this weighting.
    w_vel: float = 10.0
    w_upright: float = 0.5
    w_effort: float = 0.01
    w_action_rate: float = 0.01
    w_collision: float = 1.0
    seed: int | None = None
    # U27: the first reset-perturbation family, per instruction -- "begin
    # with one perturbation family, such as small initial joint-velocity
    # disturbances. Expand only after recovery improves." 0.0 (default)
    # reproduces every prior deterministic-reset result in this project
    # exactly -- U23 confirmed reset() was fully deterministic and multiple
    # scripts rely on that; this stays opt-in, never on by default.
    reset_qvel_noise_std: float = 0.0   # rad/s, applied to each of the 12 actuated joints' initial qvel
    # U29: broaden the SAME perturbation family (still initial joint-velocity
    # noise, nothing else) into a per-episode MIXTURE instead of one fixed
    # magnitude -- "mix nominal resets, previously manageable disturbances,
    # and a modestly harder band. Preserve some easy episodes so recovery
    # training does not erase standing." Each tuple is (probability, std).
    # None (default) falls back to the single reset_qvel_noise_std value
    # above, reproducing every prior result exactly -- opt-in, like U27's
    # field was.
    reset_qvel_noise_bands: tuple | None = None


class CaraWalkEnv:
    """Bare reset()/step() MuJoCo environment. No training loop, no policy --
    just the interface the roadmap asks for, built on the model that's
    already validated (standing MET, weight-shift MET, single-support
    balance MET, quasi-static stepping MET)."""

    def __init__(self, cfg: CaraWalkEnvConfig | None = None):
        import mujoco
        import numpy as np
        import generate_mjcf

        self.mujoco = mujoco
        self.np = np
        self.cfg = cfg or CaraWalkEnvConfig()

        self.spec = lm.load_spec(self.cfg.config_path)
        xml = generate_mjcf.build_mjcf(self.spec, dynamic=True)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.substeps = max(1, round(1.0 / self.cfg.control_hz / self.dt))
        self.max_steps = int(round(self.cfg.episode_seconds * self.cfg.control_hz))

        self.jn = lm.actuated_joint_names(self.spec)
        self.n_act = len(self.jn)
        base_cfg = lm.reference_poses(self.spec)[self.cfg.base_pose]
        self.nominal = np.array([float(base_cfg.get(n, 0.0)) for n in self.jn], dtype="float64")
        self.jidx = {n: i for i, n in enumerate(self.jn)}
        limits = lm.joint_limits(self.spec)
        self.lo = np.array([limits[n][0] for n in self.jn], dtype="float64")
        self.hi = np.array([limits[n][1] for n in self.jn], dtype="float64")
        self.offset_scale = self.cfg.action_range_frac * (self.hi - self.lo) / 2.0

        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.cfg.base_pose)
        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        foot_names = ("l_foot_collision", "r_foot_collision")
        self.foot_gid = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in foot_names}
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # non-foot geoms that would count as an "unwanted collision" if they
        # touch the floor (everything else that's collidable).
        self.other_gids = {g for g in range(self.model.ngeom)
                            if g not in self.foot_gid.values() and g != self.floor_gid
                            and self.model.geom_contype[g] != 0}

        self.action_space = Box(-1.0, 1.0, (self.n_act,))
        obs0 = self._dummy_obs()
        self.observation_space = Box(-np.inf, np.inf, (obs0.shape[0],))

        self._rng = np.random.default_rng(self.cfg.seed)
        self._prev_action = np.zeros(self.n_act, dtype="float64")
        self._step_count = 0
        self._prev_com_x = 0.0

    # ------------------------------------------------------------------ #
    def _dummy_obs(self):
        import numpy as np
        n_grav, n_ang, n_vel, n_prev = 3, 3, 1, self.n_act
        return np.zeros(2 * self.n_act + n_grav + n_ang + n_vel + n_prev, dtype="float32")

    def _projected_gravity(self):
        """World gravity direction expressed in the pelvis (base) frame --
        the standard IMU-equivalent orientation observation. Available on
        real hardware via an IMU; this is not a sim-only quantity."""
        np = self.np
        R = self.data.xmat[self.pelvis_bid].reshape(3, 3)
        g_world = np.array([0.0, 0.0, -1.0])
        return R.T @ g_world

    def _base_ang_vel(self):
        """Base angular velocity, body frame -- an IMU gyro observation, also
        available on real hardware."""
        return self.np.array(self.data.qvel[3:6], dtype="float64")

    @property
    def desired_vx_obs_index(self):
        """Index of the desired_vx scalar within the observation vector
        returned by _obs() -- computed from the actual concatenation order
        below, not a hardcoded magic number a future edit could silently
        invalidate."""
        return 2 * self.n_act + 3 + 3

    def _obs(self):
        np = self.np
        qpos = np.array([self.data.qpos[7 + self.jidx[n]] for n in self.jn])
        qvel = np.array([self.data.qvel[6 + self.jidx[n]] for n in self.jn])
        grav = self._projected_gravity()
        ang_vel = self._base_ang_vel()
        desired = np.array([self.cfg.desired_vx])
        obs = np.concatenate([qpos, qvel, grav, ang_vel, desired, self._prev_action]).astype("float32")
        return obs

    def _foot_contacts_and_collisions(self):
        """Returns (contact geoms touching the floor, any NON-foot geom
        touching the floor -- an 'unwanted collision')."""
        touching = set()
        collided = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.floor_gid not in pair:
                continue
            other = (pair - {self.floor_gid}).pop()
            if other in self.foot_gid.values():
                touching.add(other)
            elif other in self.other_gids:
                collided = True
        return touching, collided

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> tuple[Any, dict]:
        mujoco = self.mujoco
        np = self.np
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        if self.cfg.desired_vx_bands is not None:
            probs = [b[0] for b in self.cfg.desired_vx_bands]
            vxs = [b[1] for b in self.cfg.desired_vx_bands]
            self.cfg.desired_vx = vxs[self._rng.choice(len(vxs), p=probs)]
        perturbed = False
        if self.cfg.reset_qvel_noise_bands is not None:
            probs = [b[0] for b in self.cfg.reset_qvel_noise_bands]
            stds = [b[1] for b in self.cfg.reset_qvel_noise_bands]
            chosen_std = stds[self._rng.choice(len(stds), p=probs)]
            self._last_reset_band_std = chosen_std
            if chosen_std > 0.0:
                noise = self._rng.normal(0.0, chosen_std, size=self.n_act)
                for n, dv in zip(self.jn, noise):
                    self.data.qvel[6 + self.jidx[n]] += dv
                perturbed = True
        elif self.cfg.reset_qvel_noise_std > 0.0:
            self._last_reset_band_std = self.cfg.reset_qvel_noise_std
            noise = self._rng.normal(0.0, self.cfg.reset_qvel_noise_std, size=self.n_act)
            for n, dv in zip(self.jn, noise):
                self.data.qvel[6 + self.jidx[n]] += dv
            perturbed = True
        else:
            self._last_reset_band_std = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._prev_action = np.zeros(self.n_act, dtype="float64")
        self._step_count = 0
        self._prev_com_x = float(self.data.subtree_com[0][0])
        info = {"nominal_pose": self.nominal.copy(), "reset_perturbed": perturbed,
                "reset_band_std": self._last_reset_band_std}
        return self._obs(), info

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        np = self.np
        action = np.clip(np.asarray(action, dtype="float64"), -1.0, 1.0)
        target = np.clip(self.nominal + action * self.offset_scale, self.lo, self.hi)
        self.data.ctrl[:] = target  # <position> actuators: ctrl IS the joint target

        for _ in range(self.substeps):
            self.mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        valid = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))

        pelvis_pos = self.data.xpos[self.pelvis_bid]
        roll, pitch, _ = _quat_rpy(self.data.qpos[3:7])
        tilt_deg = math.degrees(max(abs(roll), abs(pitch)))
        fallen = (not valid) or tilt_deg > self.cfg.fall_tilt_deg or float(pelvis_pos[2]) < self.cfg.fall_height_m

        com_x = float(self.data.subtree_com[0][0]) if valid else self._prev_com_x
        vx = (com_x - self._prev_com_x) / (self.substeps * self.dt)
        self._prev_com_x = com_x

        touching, collided = self._foot_contacts_and_collisions() if valid else (set(), False)
        qpos_err = float(np.mean(np.abs(
            np.array([self.data.qpos[7 + self.jidx[n]] for n in self.jn]) - target))) if valid else float("nan")
        foot_z = {n: float(self.data.geom_xpos[gid][2]) for n, gid in self.foot_gid.items()}
        foot_touch = {n: (gid in touching) for n, gid in self.foot_gid.items()}

        if fallen:
            # The CoM can lurch forward as the robot collapses (a pitching
            # fall drags the pelvis/torso mass ahead of the feet) -- that is
            # not stepping and must not earn velocity-tracking credit. On the
            # terminating step the only signal is the flat fall penalty; no
            # component here is a function of vx.
            reward = -10.0
            r_alive = r_vel = r_upright = r_effort = r_rate = r_collision = None
        else:
            # A per-step "alive" bonus is required, not cosmetic: without it,
            # every non-terminal step costs (-w_vel*|vx-desired| + small
            # penalties) < 0, so accumulated reward strictly decreases the
            # longer an episode survives, while a fall pays a single fixed
            # -10 and then STOPS accumulating further negative reward. A
            # random or partially-trained policy that cannot track vx is
            # then rewarded for falling as early as possible (verified: the
            # first ARS training run drove episode length from 200 steps
            # down to ~20-30 across 20 iterations before this bonus was
            # added -- see docs/rl_environment_notes.md U20). w_alive must
            # exceed the typical per-step tracking penalty so survival is
            # always preferred to an early, deliberate fall.
            r_alive = 1.0
            r_vel = -abs(vx - self.cfg.desired_vx)
            # U30 sign fix: upright (R=I) gives projected_gravity_z=-1, so
            # (z+1)=0 there -- correct so far. But fully inverted flips the
            # body z-axis, giving z=+1, so (z+1)=+2 -- the ORIGINAL formula
            # (without the leading minus) REWARDED tipping/inversion,
            # opposite of its own comment ("0 upright, -2 fully tipped").
            # Verified at synthetic tilts before fixing, not assumed: 10deg
            # gave +0.0152, 20deg gave +0.0603, 90deg gave +1.0, 180deg gave
            # +2.0 under the buggy sign -- monotonically REWARDED leaning
            # further over. Negating restores 0 upright / -2 inverted, i.e.
            # a real penalty that grows with tilt, matching the comment.
            r_upright = -float(self._projected_gravity()[2] + 1.0)  # 0 upright, -2 fully tipped
            r_effort = -float(np.mean(action ** 2))
            r_rate = -float(np.mean((action - self._prev_action) ** 2))
            r_collision = -1.0 if collided else 0.0
            reward = (self.cfg.w_alive * r_alive
                      + self.cfg.w_vel * r_vel + self.cfg.w_upright * r_upright
                      + self.cfg.w_effort * r_effort + self.cfg.w_action_rate * r_rate
                      + self.cfg.w_collision * r_collision)

        self._prev_action = action
        terminated = fallen
        truncated = self._step_count >= self.max_steps
        info = {
            "vx": vx, "tilt_deg": tilt_deg, "pelvis_z": float(pelvis_pos[2]),
            "valid": valid, "collided": collided, "fallen": fallen,
            "torque_frac": float(np.max(np.abs(self.data.actuator_force) / self.model.actuator_forcerange[:, 1])),
            "qpos_err": qpos_err, "foot_z": foot_z, "foot_touch": foot_touch,
            "reward_components": {"alive": r_alive, "vel": r_vel, "upright": r_upright,
                                   "effort": r_effort, "rate": r_rate, "collision": r_collision},
        }
        obs = self._obs() if valid else self._dummy_obs()
        return obs, float(reward), terminated, truncated, info

    def close(self):
        pass


def _quat_rpy(q):
    """q = [w,x,y,z] -> (roll, pitch, yaw), radians. Local copy so this file
    has no dependency on weight_shift.py -- keeps the RL env import-light."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--policy", choices=["zero", "random"], default="zero",
                     help="zero: hold nominal pose (sanity check); random: bounded random actions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    cfg = CaraWalkEnvConfig(config_path=args.config, episode_seconds=args.steps / 50.0, seed=args.seed)
    env = CaraWalkEnv(cfg)
    print(f"CaraWalkEnv: {env.n_act} actuated joints, obs_dim={env.observation_space.shape[0]}, "
          f"action_dim={env.action_space.shape[0]}, control_hz={cfg.control_hz:.0f}, "
          f"substeps/ctrl={env.substeps}, max_steps={env.max_steps}")

    import numpy as np
    rng = np.random.default_rng(args.seed)
    all_ok = True
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if not np.all(np.isfinite(obs)):
            print(f"episode {ep}: FAIL -- non-finite obs on reset")
            all_ok = False
            continue
        ep_return = 0.0
        ep_len = 0
        fell = False
        t0 = env.data.time
        for t in range(args.steps):
            action = np.zeros(env.n_act) if args.policy == "zero" else env.action_space.sample(rng)
            obs, reward, terminated, truncated, step_info = env.step(action)
            if not np.all(np.isfinite(obs)) or not math.isfinite(reward):
                print(f"episode {ep} step {t}: FAIL -- non-finite obs/reward")
                all_ok = False
                break
            ep_return += reward
            ep_len += 1
            if terminated:
                fell = True
                break
            if truncated:
                break
        elapsed = env.data.time - t0
        expected = ep_len / cfg.control_hz
        timing_ok = abs(elapsed - expected) < 1e-6
        all_ok = all_ok and timing_ok
        print(f"episode {ep} [{args.policy}]: len={ep_len}/{args.steps}  return={ep_return:.2f}  "
              f"fell={fell}  final_tilt={step_info.get('tilt_deg', float('nan')):.1f}deg  "
              f"sim_time={elapsed:.3f}s (expected {expected:.3f}s) timing_ok={timing_ok}")

    print("\nRESULT:", "PASS" if all_ok else "FAIL",
          "-- resets + rollouts produced finite, correctly-timed results" if all_ok
          else "-- see failures above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
