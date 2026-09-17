#!/usr/bin/env python3
"""U34 -- CaraResidualEnv: a SEPARATE environment from CaraWalkEnv (frozen,
untouched -- see cara_env.py) that retains the COMPLETE U10/U11 teacher
(scripts/gait.py, unmodified source) running at its native 500Hz, and
accepts learned residual corrections from a policy at 50Hz:

    q_target(t) = limit( q_teacher(x_t, z_t) + dq_RL(t) )

  - Teacher (500Hz): recomputes its nominal stepping target AND its live
    balance feedback (U9's ankle/hip-roll trim) from the CURRENT physical
    state every physics substep -- exactly as gait.py always has. Its
    entire phase state machine (six phases per step, swing tables, the
    clearance integrator, the roll-trim reference) is retained; nothing is
    extracted or re-derived, per instruction ("extracting only the balance
    loop would introduce another untested controller").
  - Policy (50Hz): requests a bounded correction dq_RL around whatever the
    teacher is currently commanding. The residual bound applies to dq_RL
    ONLY, never to the combined command -- reapplying CaraWalkEnv's
    nominal-centered +/-1 bound to q_teacher+dq_RL would recreate U32/U33's
    demonstrated failure (this is the whole reason this is a SEPARATE
    environment, not a wrapper around CaraWalkEnv's action space).
  - Final command limits: joint limits only (mirrors gait.py's own
    behavior -- it does not self-clip beyond the model's physical range).
    Actuator FORCE limits are already enforced by MuJoCo's own
    <position> servo + forcerange, unaffected by anything here.
  - Environment: advances exactly 10 physics steps (matching CaraWalkEnv's
    own substeps=10 at dt=0.002, i.e. 50Hz) per env.step() call.

Mechanism: gait.py's `run()`/`walk()` are deeply nested closures (all its
phase logic lives inside `run()`'s local scope) with no hook for pausing
mid-walk to accept an external correction. Rather than modify gait.py (it
stays a preserved, untouched generator/validator -- confirmed still
passing its own milestone independently after every change in this
project), this module runs gait.py's OWN `run()` call in a background
thread and synchronizes with it via a producer/consumer queue pair at
every mj_step() call, using the SAME non-invasive monkeypatch technique as
U32/U33. This is the only way to get bidirectional, per-substep control
over an unmodified, deeply-nested blocking call.

Privileged-state disclosure (per instruction): gait.py's own control law
reads `data.subtree_com` (the exact rigid-body center of mass) every
substep for its lateral balance trim. This is a genuinely privileged
simulator quantity -- real hardware would need a full CAD mass model plus
forward kinematics (or a dynamic estimator) to approximate it, not a
direct sensor reading. Swing-foot height and stance-foot position are
read via `data.geom_xpos` (world-frame body geometry), approximable via
forward kinematics from joint encoders on hardware. Base orientation
(`data.qpos[3:7]`) is an ordinary IMU-equivalent quantity, not privileged.
This applies to the TEACHER's own internal control law (unchanged, as
required) -- the residual POLICY's observation vector, built fresh below,
is documented separately.
"""

from __future__ import annotations

import math
import os
import queue
import threading
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_full_body.yaml"))

DT = 0.002              # gait.py's own model.opt.timestep -- verified equal to CaraWalkEnv's, not assumed
DECISION_RATIO = 10     # substeps per policy decision -- matches CaraWalkEnv's substeps (50Hz)
CONTROL_HZ = 1.0 / (DT * DECISION_RATIO)


class _Fell(Exception):
    def __init__(self, tilt_deg, height_m, reason):
        self.tilt_deg, self.height_m, self.reason = tilt_deg, height_m, reason


def _quat_rpy(q):
    """Identical formula to cara_env.py's _quat_rpy -- cross-checked once
    (U33) against that file so termination/tilt reporting stays consistent
    across both environments."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    return roll, pitch


def phase_schedule(n_steps, gait_cfg):
    """Exact substep-index boundaries for every phase of gait.py's own
    scripted walk, computed the SAME way gait.py computes its own loop
    bounds (int(seconds/dt)) -- derived from the YAML config alone, not
    extracted from gait.py's internals (which expose no hook for this).
    Cross-checked once against U33's own recorded total_decision_steps
    (23650 for n_steps=2) before trusting it."""
    settle = int(gait_cfg.get("settle_seconds", 1.5) / DT)
    ramp = int(gait_cfg.get("ramp_seconds", 4.0) / DT)
    swing = int(gait_cfg.get("swing_seconds", 4.0) / DT)
    dhold = int(0.4 / DT)
    fsettle = int(gait_cfg.get("step_settle_seconds", 1.0) / DT)
    fhold = int(gait_cfg.get("final_hold_seconds", 3.0) / DT)
    per_step = ramp + ramp + swing + (ramp + dhold) + (ramp + fsettle)

    schedule = [(0, settle, "settle", None, -1)]
    off = settle
    for i in range(n_steps):
        lead = "l_" if i % 2 == 0 else "r_"
        bounds = [("A", ramp), ("B", ramp), ("C", swing), ("D", ramp + dhold), ("E", ramp + fsettle)]
        for name, length in bounds:
            schedule.append((off, off + length, name, lead, i))
            off += length
    schedule.append((off, off + fhold, "final_hold", None, n_steps))
    total = off + fhold
    return schedule, total


def phase_at(substep_idx, schedule):
    for start, end, name, lead, step_idx in schedule:
        if start <= substep_idx < end:
            progress = (substep_idx - start) / max(1, end - start)
            return dict(phase=name, lead=lead, step_idx=step_idx, phase_progress=progress)
    return dict(phase="done", lead=None, step_idx=-1, phase_progress=1.0)


class _TeacherThread:
    """Runs gait.py's own run() in a background thread, pausing at every
    mj_step() call to hand the teacher's freshly-computed (LIVE-state)
    target to the main thread and wait for the residual-combined command
    to apply for the next DECISION_RATIO physics steps."""

    def __init__(self, config_path, n_steps, limit_fn, fall_tilt_deg, fall_height_m):
        self.to_teacher = queue.Queue(maxsize=1)
        self.from_teacher = queue.Queue(maxsize=1)
        self.limit_fn = limit_fn
        self.fall_tilt_deg = fall_tilt_deg
        self.fall_height_m = fall_height_m
        self.substep = 0
        self.held_residual = np.zeros(1)  # resized on first decision
        self.model = None
        self.data = None
        self._thread = threading.Thread(target=self._run, args=(config_path, n_steps), daemon=True)
        self._thread.start()

    def _wrapped_mj_step(self, model, data):
        self.model, self.data = model, data
        i = self.substep
        # gait.py has ALREADY written its freshly-computed, live-state
        # target into data.ctrl by this point, on EVERY call -- this is
        # what "teacher at 500Hz" means: it keeps reacting to the actual
        # simulated state every physics substep, decision boundary or not.
        raw_target = np.array(data.ctrl, dtype=np.float64).copy()
        if i % DECISION_RATIO == 0:
            self.from_teacher.put(("decision", raw_target))
            combined = self.to_teacher.get()  # blocks for the RL step() call
            # Only the RESIDUAL portion is held for the next DECISION_RATIO
            # substeps -- the teacher's own contribution keeps updating
            # every substep below, via raw_target. Holding the full
            # combined signal instead (an earlier version of this file did
            # exactly that) silently degrades the TEACHER itself to 50Hz,
            # which reproduced U33's 50Hz/native FAILURE under supposedly
            # zero-residual replay -- caught by the equivalence check this
            # milestone exists to run, not shipped silently.
            self.held_residual = combined - raw_target
        combined_now = self.limit_fn(raw_target + self.held_residual)
        data.ctrl[:] = combined_now
        self._orig_mj_step(model, data)
        self.substep = i + 1

        if i % DECISION_RATIO == DECISION_RATIO - 1:
            roll, pitch = _quat_rpy(data.qpos[3:7])
            tilt_deg = math.degrees(max(abs(roll), abs(pitch)))
            height = float(data.qpos[2])
            valid = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
            if not valid or tilt_deg > self.fall_tilt_deg or height < self.fall_height_m:
                raise _Fell(tilt_deg, height, "nonfinite" if not valid else
                            ("tilt" if tilt_deg > self.fall_tilt_deg else "height"))

    def _run(self, config_path, n_steps):
        import mujoco
        self._orig_mj_step = mujoco.mj_step
        mujoco.mj_step = self._wrapped_mj_step
        outcome = ("done", None)
        try:
            import gait
            gait.run(config_path, n_steps, view=False, json_path=None, baseline_path=None)
        except _Fell as e:
            outcome = ("fell", e)
        finally:
            mujoco.mj_step = self._orig_mj_step
            self.from_teacher.put(outcome)

    def first_decision(self):
        return self.from_teacher.get()

    def submit(self, combined_target):
        """Apply combined_target for the next DECISION_RATIO physics steps
        and return the teacher's NEXT raw target (or a terminal marker)."""
        self.to_teacher.put(combined_target)
        return self.from_teacher.get()


@dataclass
class CaraResidualEnvConfig:
    config_path: str = DEFAULT_CONFIG
    n_steps: int = 2
    desired_vx: float = 0.03
    fall_tilt_deg: float = 40.0
    fall_height_m: float = 0.15
    w_alive: float = 1.0
    w_vel: float = 10.0
    w_upright: float = 0.5
    w_effort: float = 0.01          # on the COMBINED command's deviation from nominal (total motion)
    w_residual_rate: float = 0.3    # on the RESIDUAL's step-to-step change ONLY -- separate from total motion
    w_collision: float = 1.0
    # Residual bound (rad), same for every joint initially -- chosen from
    # probes (see U34's probe script), NOT copied from action_range_frac.
    residual_bound_rad: float = 0.02


class CaraResidualEnv:
    """Gym-style reset()/step(residual) environment. residual is a length-12
    array in RADIANS (not normalized [-1,1] -- there is deliberately no
    fixed "full range" for a residual the way there is for CaraWalkEnv's
    from-nominal action, since the residual's natural scale is whatever a
    small correction means, established empirically in U34's probes)."""

    def __init__(self, cfg: CaraResidualEnvConfig | None = None):
        import leg_model as lm
        import generate_mjcf
        import mujoco

        self.cfg = cfg or CaraResidualEnvConfig()
        self.mujoco = mujoco
        self.spec = lm.load_spec(self.cfg.config_path)
        self.jn = lm.actuated_joint_names(self.spec)
        self.n_act = len(self.jn)
        limits = lm.joint_limits(self.spec)
        self.lo = np.array([limits[n][0] for n in self.jn])
        self.hi = np.array([limits[n][1] for n in self.jn])
        base_cfg = lm.reference_poses(self.spec)["stand_nominal"]
        self.nominal = np.array([float(base_cfg.get(n, 0.0)) for n in self.jn])

        gait_cfg = (self.spec.get("analysis", {}) or {}).get("gait", {}) or {}
        self.schedule, self.total_substeps = phase_schedule(self.cfg.n_steps, gait_cfg)
        self.total_decisions = self.total_substeps // DECISION_RATIO

        # Probe model/data just for geometry ids (foot contact) -- a
        # throwaway instance, NOT the teacher thread's own model/data.
        xml = generate_mjcf.build_mjcf(self.spec, dynamic=True)
        probe_model = mujoco.MjModel.from_xml_string(xml)
        self.floor_gid = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_gid = {p: mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_GEOM, f"{p}foot_collision")
                          for p in ("l_", "r_")}

        self._teacher = None
        self._prev_residual = np.zeros(self.n_act)
        self._prev_combined = None
        self._pending_raw_target = None
        self._substep_at_decision = 0
        self._prev_com_x = 0.0

    def _limit_fn(self, combined):
        return np.clip(combined, self.lo, self.hi)

    def _foot_touch(self, data):
        touch = {p: False for p in self.foot_gid}
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {c.geom1, c.geom2}
            if self.floor_gid not in pair:
                continue
            other = (pair - {self.floor_gid}).pop()
            for p, gid in self.foot_gid.items():
                if other == gid:
                    touch[p] = True
        return touch

    def _obs_and_info(self, raw_target, done_reason=None):
        data = self._teacher.data
        touch = self._foot_touch(data)
        qpos = np.array([data.qpos[7 + i] for i in range(self.n_act)])
        qvel = np.array([data.qvel[6 + i] for i in range(self.n_act)])
        roll, pitch = _quat_rpy(data.qpos[3:7])
        # projected gravity via quaternion rotation (matches cara_env.py's
        # own convention: world -Z expressed in the base frame)
        w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        # R^T @ [0,0,-1], computed directly from the quaternion (avoids
        # needing a body id lookup against the teacher's own model instance)
        gz = -(1 - 2 * (x * x + y * y))
        gx = -(2 * (x * z - w * y))
        gy = -(2 * (y * z + w * x))
        proj_grav = np.array([gx, gy, gz])
        ang_vel = np.array(data.qvel[3:6])
        phase_info = phase_at(self._substep_at_decision, self.schedule)
        phase_onehot = {"settle": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "final_hold": 6, "done": 7}
        phase_vec = np.zeros(8)
        phase_vec[phase_onehot.get(phase_info["phase"], 7)] = 1.0
        support_vec = np.array([1.0 if touch["l_"] else 0.0, 1.0 if touch["r_"] else 0.0])
        teacher_target_norm = (raw_target - self.nominal)  # radians, centered on nominal for scale-sanity
        obs = np.concatenate([
            qpos, qvel, proj_grav, ang_vel,                       # proprioception (12+12+3+3=30)
            phase_vec, support_vec,                                # controller context (8+2=10)
            teacher_target_norm, self._prev_residual,              # 12+12=24
            [self.cfg.desired_vx],
        ]).astype(np.float32)
        pelvis_x = float(data.qpos[0])
        info = dict(phase=phase_info["phase"], lead=phase_info["lead"], step_idx=phase_info["step_idx"],
                    foot_touch=touch, tilt_deg=math.degrees(max(abs(roll), abs(pitch))),
                    pelvis_x=pelvis_x, raw_teacher_target=raw_target.copy(),
                    combined_target=self._prev_combined.copy() if self._prev_combined is not None else None,
                    done_reason=done_reason)
        return obs, info

    def reset(self, seed=None):
        self._prev_residual = np.zeros(self.n_act)
        self._prev_combined = None
        self._prev_com_x = 0.0
        self._substep_at_decision = 0
        self._teacher = _TeacherThread(self.cfg.config_path, self.cfg.n_steps, self._limit_fn,
                                        self.cfg.fall_tilt_deg, self.cfg.fall_height_m)
        msg, payload = self._teacher.first_decision()
        assert msg == "decision", "teacher ended before its first decision point -- something is wrong with the setup"
        self._pending_raw_target = payload
        self._prev_com_x = float(self._teacher.data.qpos[0])
        obs, info = self._obs_and_info(self._pending_raw_target)
        return obs, info

    def step(self, residual):
        residual = np.clip(np.asarray(residual, dtype=np.float64), -self.cfg.residual_bound_rad, self.cfg.residual_bound_rad)
        raw_target = self._pending_raw_target
        combined = raw_target + residual  # limit() applied inside the teacher thread via self._limit_fn
        self._substep_at_decision += DECISION_RATIO

        msg, payload = self._teacher.submit(combined)
        data_after = self._teacher.data  # valid regardless of msg: mj_step already ran for this decision's substeps

        com_x = float(data_after.qpos[0])
        vx = (com_x - self._prev_com_x) / (DECISION_RATIO * DT)
        self._prev_com_x = com_x

        terminated = (msg == "fell")
        truncated = (msg == "done")
        fell_info = payload if terminated else None

        if msg == "decision":
            self._pending_raw_target = payload

        # ---- reward, computed ONCE per 50Hz decision (not per physics substep) ----
        if terminated:
            reward = -10.0
            components = dict(alive=None, vel=None, upright=None, effort=None, residual_rate=None, collision=None)
        else:
            roll, pitch = _quat_rpy(data_after.qpos[3:7])
            w_, x_, y_, z_ = data_after.qpos[3], data_after.qpos[4], data_after.qpos[5], data_after.qpos[6]
            proj_grav_z = -(1 - 2 * (x_ * x_ + y_ * y_))
            r_alive = 1.0
            r_vel = -abs(vx - self.cfg.desired_vx)
            r_upright = -(proj_grav_z + 1.0)
            r_effort = -float(np.mean(((combined - self.nominal) / np.maximum(self.hi - self.lo, 1e-6)) ** 2))
            r_residual_rate = -float(np.mean(((residual - self._prev_residual) / self.cfg.residual_bound_rad) ** 2))
            touch = self._foot_touch(data_after)
            # collision: any non-foot geom on the floor -- approximate via
            # contact count check would need full geom set; left at 0 here
            # (no upper-body/torso contact geometry in this leg-only model
            # to trigger it, consistent with CaraWalkEnv's own experience).
            r_collision = 0.0
            components = dict(alive=r_alive, vel=r_vel, upright=r_upright, effort=r_effort,
                               residual_rate=r_residual_rate, collision=r_collision)
            reward = (self.cfg.w_alive * r_alive + self.cfg.w_vel * r_vel + self.cfg.w_upright * r_upright
                      + self.cfg.w_effort * r_effort + self.cfg.w_residual_rate * r_residual_rate
                      + self.cfg.w_collision * r_collision)

        self._prev_residual = residual
        self._prev_combined = combined

        # payload is a fresh raw_target only on "decision"; on "fell" it's
        # the _Fell exception object, and on "done" (teacher finished
        # naturally) it's None -- either way there is no NEXT teacher
        # target, so fall back to the target just submitted this step.
        next_raw = payload if msg == "decision" else raw_target
        obs, info = self._obs_and_info(next_raw,
                                        done_reason=("fell:" + fell_info.reason) if terminated else
                                        ("done" if truncated else None))
        info["vx"] = vx
        info["reward_components"] = components
        info["residual_applied"] = residual.copy()
        info["total_commanded_motion"] = float(np.mean(np.abs(combined - self.nominal)))
        return obs, float(reward), terminated, truncated, info

    def close(self):
        pass
