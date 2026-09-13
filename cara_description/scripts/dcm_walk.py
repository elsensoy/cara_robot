#!/usr/bin/env python3
"""U14 / U15 / U16 -- a DCM-tracking walk on torque-controlled ankles.

U13 (`walk_model.py`) showed a dynamically-consistent walk is within Cara's
morphology.  U14 built the DCM-tracking controller (plan + capture-point
feedback + step adjustment) but hit two realisation walls on the position-PD
model: the from-rest lateral state exceeds the double-support envelope, and the
position servos cannot place the center of pressure.

**U15 unblocked the actuation:**

  * the four ankle joints become direct-torque `<motor>` actuators
    (`dynamics.actuators.torque_joints`, set here at runtime) -- the DCM
    controller's CoP command is realised as an ankle torque `tau = Fz * (p_cmd -
    p_ankle)` on top of a software attitude PD that keeps the foot behaving;
  * the walk is entered with a **limit-cycle warm-start** -- a few rocking
    half-steps in place that build the lateral momentum a LIPM gait needs,
    before any forward progress.

The warm-start itself didn't work yet: it fell during the very first rock.

**U16 fixes gait initiation.**  Two root causes, both in the warm-up rock:

  1. *Wrong-foot CoP realisation.*  Each rocking half-step alternated which
     foot's ankle received the CoP torque (by step index), but both feet stay
     planted the whole warm-up -- as the rock's amplitude grows, weight can
     (and does) transfer fully onto one foot *before* its half-step officially
     ends, handing control to a foot carrying ~0 N right when it's needed
     most.  Fixed: drive each ankle from its OWN measured contact force and
     its OWN local CoP clamp, so whichever foot is actually loaded is the one
     doing the work.
  2. *An ill-conditioned excitation.*  The DCM equation is exponential in
     omega0*T; a CoP nudge held for a full forward-step duration (t_step,
     ~0.5s, ~2.9 time constants at omega0~5.7) amplifies any tracking error
     by ~18x by the end of the step -- reaching a target from rest needs a
     near-mm-precise CoP placement, which neither the feedback law nor the
     foot-sized CoP clamp can deliver.  Fixed: give the warm-up its own, much
     shorter step duration (`warmup_t_step`), keeping that amplification factor
     small (~2-4x) and the controller in its well-behaved linear regime; also
     cap the warm-up's peak CoP excursion below the full foot half-width
     (`warmup_amp_hi`) so the last, biggest rock doesn't run out of margin.

Result: the warm-up rock now survives cleanly regardless of how many steps it
runs (DCM error single-digit-mm growing to ~40 mm), and carries into the first
real forward step -- which itself then completed, but handing off into the
SECOND real forward step still failed (same exponential-amplification bug as
U16 fixed for the warm-up, just now between forward steps).

**U17 extends both U16 fixes to real forward steps.**  The forward-step
duration (`t_step`) was still 0.5s (~18x DCM-error amplification per step);
shortened toward U13's own T_min (0.22s, ~3.5x amplification) it tracks far
tighter.  And a real step's own opening/closing double-support windows
(`double_support_frac`) were still realising the CoP through a single
index-alternated "stance" ankle, the same bug U16 fixed for the warm-up;
realising it through whichever sole is actually loaded, every substep where
`sp` is at 0 or 1 (not just during `warm`), removes that inconsistency too. A
longer, gentler warm-up (`warmup_steps` 6 -> 12, `warmup_amp_hi` capped at
0.030 m, `cop_torque_gain` softened to 0.5) brings the entry error down
further. Together: peak DCM error 143 mm -> 44 mm, and Cara now clears the
*entire* warm-up plus the first real forward step, every run.

The walk still doesn't complete -- the SECOND real forward step is the new
narrower failure; see the printed report for the current diagnosis.

Then: plan the CoP + footholds + DCM reference from the LIPM, track the DCM
(`p_cmd = p_ref + (1 + k/omega0)(xi_meas - xi_ref)`), adjust each foothold to the
capture point.  No fixed pose cycle, no RL.  Failures are reported, not hidden.

Requires `mujoco` (brings numpy).  Prints SKIPPED / exits 0 without it.

Usage:
    python3 dcm_walk.py                     # full body: warm-up + N forward steps
    python3 dcm_walk.py --steps 10
    python3 dcm_walk.py --t-step 0.40
    python3 dcm_walk.py --warmup-t-step 0.15
    python3 dcm_walk.py --no-torque-ankles  # U14 mode (position servos -- fails)
    python3 dcm_walk.py --view
    python3 dcm_walk.py --json baselines/full_body_dcm_walk.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import leg_model as lm
import weight_shift as wsh
from walk_model import LIPM

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf"))
DEFAULT_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_full_body.yaml"))

SIDE = {"l_": +1.0, "r_": -1.0}
OTHER = {"l_": "r_", "r_": "l_"}
C_TOP = 0.035
ANKLES = [p + a for p in ("l_", "r_") for a in ("ankle_roll", "ankle_pitch")]


def run(config, n_steps_arg, t_step_arg, t_warm_arg, torque_ankles, view, json_path, baseline_path,
        live_swing_test=False, ankle_kd_sweep=False):
    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0
    import generate_mjcf

    spec = lm.load_spec(config)
    model_name = spec["meta"]["name"]
    if torque_ankles:
        tj = spec.setdefault("dynamics", {}).setdefault("actuators", {}).setdefault("torque_joints", [])
        for j in ANKLES:
            if j not in tj:
                tj.append(j)
    xml = generate_mjcf.build_mjcf(spec, dynamic=True)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    g = lm.analysis_gravity(spec)

    dw = (spec.get("analysis", {}) or {}).get("dcm_walk", {}) or {}
    if not dw:
        print("this config has no analysis.dcm_walk block")
        return 2
    base_pose = dw.get("base_pose", "stand_nominal")
    stride = float(dw.get("stride", 0.024))
    t_step = float(t_step_arg if t_step_arg is not None else dw.get("t_step", 0.40))
    t_warm = float(t_warm_arg if t_warm_arg is not None else dw.get("warmup_t_step", 0.18))
    n_steps = int(n_steps_arg if n_steps_arg is not None else dw.get("n_steps", 8))
    n_warm = int(dw.get("warmup_steps", 4))
    ds_frac = float(dw.get("double_support_frac", 0.20))
    lift_h = float(dw.get("lift_height", 0.008))
    lift_warm = float(dw.get("warmup_lift", 0.004))
    CG = float(dw.get("clearance_gain", 0.02))
    k_dcm = float(dw.get("k_dcm", 3.0))
    kp_att = float(dw.get("ankle_kp", 22.0))
    kd_att = float(dw.get("ankle_kd", 1.2))
    cop_gain = float(dw.get("cop_torque_gain", 1.0))
    step_adj_gain = float(dw.get("step_adjust_gain", 0.7))
    lead_in = float(dw.get("lead_in_seconds", 1.5))
    tail = float(dw.get("tail_seconds", 3.0))
    acc = dw.get("accept", {}) or {}
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 10.0)))
    MAX_DCM_ERR = float(acc.get("max_dcm_error", 0.025))
    MIN_SPEED_FRAC = float(acc.get("min_speed_frac", 0.5))
    MAX_TQ = float(acc.get("max_torque_frac", 1.0))
    TAIL_DRIFT = float(acc.get("tail_hold_drift", 0.012))

    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)[base_pose]
    FULL0 = {n: float(base_cfg.get(n, 0.0)) for n in jn}
    nominal_ctrl = [FULL0[n] for n in jn]
    is_torque = [jn[i] in ANKLES and torque_ankles for i in range(len(jn))]

    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose)
    foot_gid = {"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")}
    floor_gid = gid("floor")
    forcerng = np.array([model.actuator_forcerange[aid(n)][1] for n in jn])
    m_total = float(sum(model.body_mass))
    total_weight = m_total * g

    tf0 = lm.forward_kinematics(spec, base_cfg)
    SOLE0 = {p: lm.frame_world_position(spec, tf0, p + "foot_sole_center") for p in ("l_", "r_")}
    ROT0 = {p: tf0[p + "foot"][0] for p in ("l_", "r_")}
    SAG = {p: [p + j for j in ("hip_pitch", "knee_pitch", "ankle_pitch")] for p in ("l_", "r_")}
    ROLLJ = ["l_hip_roll", "l_ankle_roll", "r_hip_roll", "r_ankle_roll"]
    JIDX = {n: i for i, n in enumerate(jn)}
    JOINT_LIMITS = lm.joint_limits(spec)
    Z0 = SOLE0["l_"][2]
    X0 = SOLE0["l_"][0]

    _, com0, _ = lm.center_of_mass(spec, base_cfg)
    z_com = com0[2] - Z0
    lip = LIPM(z_com, g)
    w0 = lip.w
    sym = lm.resolve_symbols(spec)
    a_x, a_y = 0.5 * float(sym["foot_len"]), 0.5 * float(sym["foot_width"])
    s_half = float(sym["w_hip_half"])
    com_bias = (float(com0[0]) - X0, 0.0)

    shift_table = wsh.build_ik_table(spec, base_cfg, 0.06, 61)

    def smooth01(u):
        u = min(1.0, max(0.0, u))
        return u * u * (3.0 - 2.0 * u)

    # ---------------------------------------------------------------- #
    # PLAN: n_warm rocking half-steps in place, then n_steps forward.
    # ---------------------------------------------------------------- #
    total_steps = n_warm + n_steps
    foot_nom = {"l_": [X0, +s_half], "r_": [X0, -s_half]}
    amp_lo = float(dw.get("warmup_amp_lo", 0.012))    # first rock's lateral CoP amplitude
    amp_hi = float(dw.get("warmup_amp_hi", s_half))   # last rock's lateral CoP amplitude (<= s_half)
    plan_p, plan_lead = [], []
    fp = {"l_": list(foot_nom["l_"]), "r_": list(foot_nom["r_"])}
    for i in range(total_steps):
        lead = "l_" if i % 2 == 0 else "r_"
        stance = OTHER[lead]
        plan_lead.append(lead)
        if i < n_warm:
            # gait initiation: both feet planted, the CoP LEADS the COM -- it sits
            # toward the *swing* side so the pendulum accelerates the COM toward
            # the stance foot.  Amplitude ramps amp_lo -> full foot over the warm-up.
            frac = smooth01((i + 1) / max(1, n_warm))
            amp = amp_lo + (amp_hi - amp_lo) * frac
            plan_p.append([X0 + com_bias[0], -SIDE[stance] * amp])
        else:
            plan_p.append([fp[stance][0] + com_bias[0], fp[stance][1] + com_bias[1]])
        adv = 0.0 if i < n_warm else stride
        fp[lead] = [fp[stance][0] + adv, foot_nom[lead][1]]
    # Each warm-up half-step gets its own (short) duration `t_warm`, not the
    # forward-step duration `t_step`.  This matters: the DCM equation is
    # exponential in omega0*T, so a nudge held for the full t_step (~0.5 s,
    # ~2.9 time constants at omega0~5.7) amplifies any tracking error ~18x by
    # the end of the step -- a razor's-edge CoP placement is needed to land
    # exactly on target, and the single-stance-foot CoP clamp can't correct
    # a miss that large.  A short t_warm (a fraction of a time constant) keeps
    # that amplification factor small (~2-4x) so the DCM feedback law and the
    # foot-sized CoP clamp stay inside their linear, non-saturating regime.
    step_T = [t_warm] * n_warm + [t_step] * n_steps
    eT_i = [math.exp(w0 * T) for T in step_T]
    xi_ini = [None] * (total_steps + 1)
    xi_ini[total_steps] = list(plan_p[-1])
    for i in range(total_steps - 1, -1, -1):
        p = plan_p[i]
        xi_ini[i] = [p[k] + (xi_ini[i + 1][k] - p[k]) / eT_i[i] for k in (0, 1)]

    def dcm_ref(i, tau):
        p = plan_p[i]
        e = math.exp(w0 * tau)
        return [p[k] + (xi_ini[i][k] - p[k]) * e for k in (0, 1)]

    # ---------------------------------------------------------------- #
    # swing-leg table (per step): progress s x clearance c -> sagittal joints
    #
    # U18 note: a live-pelvis-frame per-substep IK version of this was tried
    # (re-solving the sagittal target every substep against the pelvis's
    # ACTUAL pose, to fix the stale-nominal-frame touchdown-height bug this
    # table has -- see docs/single_support_notes.md U18) applied everywhere
    # (warm-up rocking included) and caused a regression: the previously
    # rock-solid 12-step warm-up fell at step 2.  An offline, physics-free
    # replay of the recorded (pelvis pose, commanded roll) trajectory through
    # the same IK -- reseeded from the true measured joint state at each step
    # boundary -- then showed the constrained task (x, z, pitch) IS reachable
    # and continuous along that trajectory (0.00mm residual almost
    # everywhere; one exception: l_knee_pitch pinned at full extension for
    # ~90/246 substeps near the hip-roll zero-crossing, residual 1-5mm
    # there).  So the live regression wasn't IK infeasibility -- most likely
    # activating it during the *warm-up rocking* (never intended; the
    # instruction was real single support only) or a dynamics-loop effect
    # the offline check can't see.  `--live-swing-test` below re-tries it,
    # SCOPED to only the first real swing, everything else on this table.
    # ---------------------------------------------------------------- #
    def build_swing_table(lead, x0_pf, xt_pf, y_pf, top, ns=9, nc=5):
        free = SAG[lead]
        q0 = dict(base_cfg)
        z0 = lm.frame_world_position(spec, lm.forward_kinematics(spec, q0),
                                     lead + "foot_sole_center")[2]
        rot = ROT0[lead]
        grid = []
        for i in range(ns):
            s = i / (ns - 1)
            row, q = [], dict(q0)
            for kk in range(nc):
                c = top * kk / (nc - 1)
                sol, _r = lm.leg_ik(spec, lead, lead + "foot_sole_center",
                                    (x0_pf + s * (xt_pf - x0_pf), y_pf, z0 + c), rot, q,
                                    free_joints=free, task_rows=[0, 2, 4], iters=150)
                q = {**q, **sol}
                row.append((c, [sol[j] for j in free]))
            grid.append((s, row))
        return grid, free

    def swing_lookup(grid, s, c):
        s = min(1.0, max(0.0, s))
        for a in range(len(grid) - 1):
            if grid[a][0] <= s <= grid[a + 1][0]:
                break
        else:
            a = len(grid) - 2
        fa = (s - grid[a][0]) / (grid[a + 1][0] - grid[a][0] or 1.0)

        def at(rl):
            cs = [cc for cc, _ in rl]
            cc = min(cs[-1], max(cs[0], c))
            for b in range(len(cs) - 1):
                if cs[b] <= cc <= cs[b + 1]:
                    fb = (cc - cs[b]) / (cs[b + 1] - cs[b] or 1.0)
                    return [rl[b][1][k] * (1 - fb) + rl[b + 1][1][k] * fb for k in range(len(rl[b][1]))]
            return rl[-1][1]

        va, vb = at(grid[a][1]), at(grid[a + 1][1])
        return [va[k] * (1 - fa) + vb[k] * fa for k in range(len(va))]

    # ---------------------------------------------------------------- #
    # --live-swing-test only: world-frame swing target, activated at the
    # first real swing's start (see run() gate below), left ACTIVE through
    # descent and the touchdown-gate extension (not reverted at sp=1).
    # Orientation target is left nominal (isolated per the earlier finding
    # that counter-rotating it was destabilising and untested).
    # ---------------------------------------------------------------- #
    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def pelvis_pose():
        p = tuple(float(v) for v in data.xpos[pelvis_bid])
        R = tuple(tuple(float(v) for v in data.xmat[pelvis_bid][3 * r:3 * r + 3]) for r in range(3))
        return R, p

    def swing_target_world(sp, start, end_x, lift):
        s = smooth01(sp)
        x_w = start[0] + s * (end_x - start[0])
        z_w = start[2] + lift * math.sin(math.pi * sp) ** 2
        return (x_w, start[1], z_w)

    def solve_swing(lead, target_world, q_seed, task_rows=(0, 2, 4)):
        R_wp, p_wp = pelvis_pose()
        p_pelvis = lm.mat_vec(lm.mat_transpose(R_wp), lm.vec_sub(target_world, p_wp))
        rot_pelvis = ROT0[lead]
        sol, resid = lm.leg_ik(spec, lead, lead + "foot_sole_center", p_pelvis, rot_pelvis,
                               q_seed, free_joints=SAG[lead], task_rows=list(task_rows), iters=150)
        return sol, resid

    def foot_normal_force(fg):
        fz = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if floor_gid in (c.geom1, c.geom2) and fg in (c.geom1, c.geom2):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f6)
                fr = c.frame
                fz += fr[2] * f6[0] + fr[5] * f6[1] + fr[8] * f6[2]
        return fz

    def foot_cop(fg):
        """Actual, contact-weighted CoP (x, y, Fz) under one foot -- the
        force-weighted mean of each contact POINT (data.contact[i].pos), not
        the foot geom's own body-fixed center.  As the ankle tips the sole,
        the load concentrates toward one edge; the geom's own position barely
        moves.  Returns (None, None, 0.0) if the foot isn't touching."""
        fz_sum = 0.0
        px = py = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if floor_gid in (c.geom1, c.geom2) and fg in (c.geom1, c.geom2):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f6)
                fr = c.frame
                fz_c = fr[2] * f6[0] + fr[5] * f6[1] + fr[8] * f6[2]
                fz_c = max(0.0, fz_c)
                fz_sum += fz_c
                px += fz_c * float(c.pos[0])
                py += fz_c * float(c.pos[1])
        if fz_sum < 1e-6:
            return None, None, 0.0
        return px / fz_sum, py / fz_sum, fz_sum

    def total_cop():
        """Whole-body actual CoP: the Fz-weighted mean of both feet's own
        (contact-weighted) CoPs -- what the plant is actually doing with the
        ground, to compare directly against p_cmd."""
        lx, ly, lz = foot_cop(foot_gid["l_"])
        rx, ry, rz = foot_cop(foot_gid["r_"])
        fz_sum = lz + rz
        if fz_sum < 1e-6:
            return None, None, 0.0, 0.0
        px = ((lx or 0.0) * lz + (rx or 0.0) * rz) / fz_sum
        py = ((ly or 0.0) * lz + (ry or 0.0) * rz) / fz_sum
        return px, py, lz, rz

    def foot_corners(fg):
        return sum(1 for i in range(data.ncon)
                   if {data.contact[i].geom1, data.contact[i].geom2} == {fg, floor_gid})

    # ---------------------------------------------------------------- #
    # U18 step 1: a CONTACT OBSERVER -- classifies actual support state from
    # measured Fz (Schmitt-trigger thresholds + a time persistence window, so
    # normal contact chatter during a fast rock doesn't flip the classification
    # every substep), run in SHADOW MODE for now: it only classifies and logs,
    # it does not yet gate anything.  Compare it against the scheduled phase
    # (`warm`/`sp`) to see whether the timer's assumed support ever disagrees
    # with what's actually on the ground -- per code review, that disagreement,
    # not CoP allocation, looks like the proximate cause of the U17 falls.
    # ---------------------------------------------------------------- #
    LOAD_ON = 0.15 * total_weight
    LOAD_OFF = 0.05 * total_weight
    PERSIST_S = 0.02
    TOUCHDOWN_WAIT_S = 0.15   # extra hold time to confirm touchdown before failing outright

    def make_support_observer():
        loaded = {"l_": None, "r_": None}   # None = not yet seeded
        since = {"l_": 0.0, "r_": 0.0}

        def update(fzl, fzr):
            for p, fz in (("l_", fzl), ("r_", fzr)):
                if loaded[p] is None:
                    # seed from the first real reading -- don't report a false
                    # NO_SUPPORT for the first PERSIST_S while a genuinely
                    # planted foot is just waiting out the debounce window.
                    loaded[p] = fz > LOAD_ON
                    continue
                raw = (fz > LOAD_ON) if not loaded[p] else not (fz < LOAD_OFF)
                if raw == loaded[p]:
                    since[p] = 0.0
                else:
                    since[p] += dt
                    if since[p] >= PERSIST_S:
                        loaded[p] = raw
                        since[p] = 0.0
            if loaded["l_"] and loaded["r_"]:
                return "DOUBLE_SUPPORT"
            if loaded["l_"]:
                return "LEFT_SUPPORT"
            if loaded["r_"]:
                return "RIGHT_SUPPORT"
            return "NO_SUPPORT"
        return update

    KD_OVERRIDE = {}   # {joint_name: kd} -- U18 ankle-loop investigation only;
                       # empty in every normal run, so apply_ctrl is unchanged.

    def apply_ctrl(cmd, cop_axis_tau):
        """cmd: {joint: position target}.  cop_axis_tau: {ankle_joint: extra torque}.
        Position joints get the target; torque ankles get a software attitude PD
        + any CoP torque, clamped to the effort limit."""
        out = np.zeros(len(jn))
        for i, n in enumerate(jn):
            if is_torque[i]:
                th = float(data.qpos[7 + i])
                thd = float(data.qvel[6 + i])
                kd_use = KD_OVERRIDE.get(n, kd_att)
                tau = kp_att * (cmd[n] - th) - kd_use * thd + cop_axis_tau.get(n, 0.0)
                out[i] = min(forcerng[i], max(-forcerng[i], tau))
            else:
                out[i] = cmd[n]
        data.ctrl[:] = out

    # ---------------------------------------------------------------- #
    # the walk
    # ---------------------------------------------------------------- #
    def walk_sim(record=True, viewer=None):
        mujoco.mj_resetDataKeyframe(model, data, kid)
        cmd0 = dict(FULL0)
        for _ in range(int((lead_in + 0.5) / dt)):
            apply_ctrl(cmd0, {})
            mujoco.mj_step(model, data)
        foot_z0 = {p: float(data.geom_xpos[foot_gid[p]][2]) for p in ("l_", "r_")}
        fw = {p: [float(data.geom_xpos[foot_gid[p]][0]), float(data.geom_xpos[foot_gid[p]][1])]
              for p in ("l_", "r_")}
        com_x0 = float(data.subtree_com[0][0])
        prev_com = [float(data.subtree_com[0][0]), float(data.subtree_com[0][1])]

        log = {"tilt": 0.0, "tq": 0.0, "dcm_err": 0.0, "fell": False,
               "step_x": [com_x0], "step_t": [0.0], "dcm_err_step": [], "cop_sat": 0,
               "cop_track_err": 0.0, "n_sub": 0, "n_sat": 0,
               "support_counts": {"DOUBLE_SUPPORT": 0, "LEFT_SUPPORT": 0,
                                   "RIGHT_SUPPORT": 0, "NO_SUPPORT": 0},
               "no_support_first": None, "stance_unsupported_n": 0}
        t_global = 0.0
        cur = dict(FULL0)
        support_state = make_support_observer()

        for i in range(total_steps):
            lead = plan_lead[i]
            stance = OTHER[lead]
            warm = i < n_warm
            p_i = plan_p[i]
            top = (lift_warm if warm else lift_h) + 0.006
            lh = lift_warm if warm else lift_h

            # --- foothold: nominal + capture-point adjustment -------------- #
            cx, cy = float(data.subtree_com[0][0]), float(data.subtree_com[0][1])
            vx = (cx - prev_com[0]) / dt
            vy = (cy - prev_com[1]) / dt
            xi_now = [cx + vx / w0, cy + vy / w0]
            xi_eos_pred = [p_i[k] + (xi_now[k] - p_i[k]) * eT_i[i] for k in (0, 1)]
            nom_next = (plan_p[i + 1] if i + 1 < total_steps else list(p_i))
            adj_cop = [nom_next[k] + (0.0 if warm else step_adj_gain) * (xi_eos_pred[k] - xi_ini[i + 1][k])
                       for k in (0, 1)]
            adj_fh = [adj_cop[0] - com_bias[0], adj_cop[1] - com_bias[1]]
            nom_fh = [fw[stance][0] + (0.0 if warm else stride), foot_nom[lead][1]]
            adj_fh[0] = min(nom_fh[0] + 0.02, max(nom_fh[0] - 0.02, adj_fh[0]))
            adj_fh[1] = min(foot_nom[lead][1] + 0.015, max(foot_nom[lead][1] - 0.015, adj_fh[1]))

            px = float(data.qpos[0])
            y_pf = lm.frame_world_position(spec, tf0, lead + "foot_sole_center")[1]
            grid, free = build_swing_table(lead, fw[lead][0] - px, adj_fh[0] - px, y_pf, top)

            # --- per-step LIPM roll-out from the measured COM state -------- #
            cx0, cy0, vx0, vy0 = cx, cy, vx, vy

            def com_ref_y(tau):
                c, sh = math.cosh(w0 * tau), math.sinh(w0 * tau)
                return p_i[1] + (cy0 - p_i[1]) * c + (vy0 / w0) * sh

            t_step_i = step_T[i]
            nsub = int(t_step_i / dt)
            ss_lo, ss_hi = ds_frac, 1.0 - ds_frac
            a_roll = stance + "ankle_roll"
            a_pit = stance + "ankle_pitch"
            step_err = 0.0
            last_obs = [None]
            SWEEP_LOG = None   # U18 ankle-kd-sweep only; see driver below
            # U18 --live-swing-test: only the FIRST real step (i == n_warm) is
            # eligible; warm-up and every later real step stay on the table
            # unchanged.  `live` activates the moment genuine single support
            # begins (sp>0) and, once active, stays active through sp=1 and
            # the touchdown-gate extension -- it is never switched back to
            # the table mid-step.
            live = {"active": False, "start": None, "seed": None, "prev": None,
                    "blend_delta": None, "blend_T": 0.0, "t_activate": 0.0}
            scoped_step = live_swing_test and (not warm) and (i == n_warm)
            # Provisional joint-speed bound used ONLY to size the activation
            # blend below -- not an actuator/servo spec, a conservative cap
            # on how fast we're willing to *ask* a joint to move to erase a
            # geometric discrepancy.  TODO: replace with a measured/CAD-derived
            # servo speed limit once real hardware is chosen.
            V_MAX_BLEND = 4.0  # rad/s

            def run_substep(p, sync, k_label):
                nonlocal step_err, t_global, prev_com
                tau = p * t_step_i
                sp = 0.0 if p <= ss_lo else (1.0 if p >= ss_hi else (p - ss_lo) / (ss_hi - ss_lo))
                clr = lh * math.sin(math.pi * sp) if 0.0 < sp < 1.0 else 0.0

                cmd = dict(cur)
                # swing sagittal joints
                wc = float(data.geom_xpos[foot_gid[lead]][2]) - foot_z0[lead]
                cur.setdefault("_cc", 0.0)
                cc = min(C_TOP, max(0.0, cur["_cc"] + CG * (clr - wc)))
                cur["_cc"] = cc
                table_targets = dict(zip(free, swing_lookup(grid, sp, cc)))

                if (not warm) and (i == n_warm) and os.environ.get("DCM_DBG_PRE"):
                    # Pre-activation trace: SCHEDULED phase (sp, from the step
                    # timer) and OBSERVED contact (raw Fz, no hysteresis) kept
                    # explicitly separate -- sp>0 means the timer's *schedule*
                    # has left double support, not that contact has.  Also
                    # break down the SWING leg's own ankle_pitch PD (it's one
                    # of the 4 torque-controlled ANKLES too, purely tracking
                    # the table's own kinematic target here -- checking
                    # whether it already chatters under the UNMODIFIED table).
                    roll_dbg, pitch_dbg, _ = wsh.quat_rpy(data.qpos[3:7])
                    a_pit_l = lead + "ankle_pitch"
                    th_ = float(data.qpos[7 + JIDX[a_pit_l]])
                    thd_ = float(data.qvel[6 + JIDX[a_pit_l]])
                    kp_ = kp_att * (table_targets[a_pit_l] - th_)
                    kd_ = -kd_att * thd_
                    unclip_ = kp_ + kd_
                    clip_ = min(forcerng[JIDX[a_pit_l]], max(-forcerng[JIDX[a_pit_l]], unclip_))
                    print(f"[pre] i={i} k={k_label:3d} p={p:.4f} sp_scheduled={sp:.4f} "
                          f"Fzl_observed={foot_normal_force(foot_gid['l_']):.1f} "
                          f"Fzr_observed={foot_normal_force(foot_gid['r_']):.1f} "
                          f"roll={math.degrees(roll_dbg):.2f} pitch={math.degrees(pitch_dbg):.2f} "
                          f"table_targets={ {n: round(table_targets[n],4) for n in free} } "
                          f"swing_ankle_pitch(thd={thd_:.2f}rad/s,kp={kp_:.2f},kd={kd_:.2f},"
                          f"unclipped={unclip_:.2f},clipped={clip_:.2f})",
                          file=sys.stderr)

                use_live = scoped_step and (live["active"] or sp > 0.0)
                if use_live and not live["active"]:
                    live["active"] = True
                    live["t_activate"] = t_global
                    gp = data.geom_xpos[foot_gid[lead]]
                    live["start"] = (float(gp[0]), float(gp[1]), float(gp[2]))
                    live["seed"] = {n: float(cur[n]) for n in free}
                    # One-shot IK solve at t0 to size the blend -- compare its
                    # result against the outgoing table command AT THE SAME
                    # instant.  delta_q is what the activation transition must
                    # remove; T_blend is sized from |delta_q| and a provisional
                    # speed cap, NOT an arbitrary short window (per review).
                    target_w0 = swing_target_world(sp, live["start"], adj_fh[0], lh)
                    seed0 = dict(cmd)
                    seed0.update(live["seed"])
                    sol0, _ = solve_swing(lead, target_w0, seed0, task_rows=(0, 2, 4))
                    delta_q = {n: table_targets[n] - sol0[n] for n in free}
                    t_blend = max(abs(delta_q[n]) for n in free) / V_MAX_BLEND
                    t_available = (ss_hi - p) * t_step_i
                    live["blend_delta"] = delta_q
                    live["blend_T"] = t_blend
                    # outgoing (table) velocity just before activation, for an
                    # explicit check -- the blend does NOT enforce this match.
                    p_prev = max(0.0, p - 1.0 / nsub)
                    sp_prev = 0.0 if p_prev <= ss_lo else (1.0 if p_prev >= ss_hi else (p_prev - ss_lo) / (ss_hi - ss_lo))
                    prev_table = dict(zip(free, swing_lookup(grid, sp_prev, cc)))
                    outgoing_vel = {n: (table_targets[n] - prev_table[n]) / dt for n in free}
                    if os.environ.get("DCM_DBG"):
                        fit = "FITS" if t_blend <= t_available else "DOES NOT FIT"
                        print(f"[live-swing] ACTIVATE i={i} k={k_label} sp={sp:.4f}  "
                              f"start_world={tuple(round(v,4) for v in live['start'])}  "
                              f"OUTGOING(table)={ {n: round(table_targets[n],4) for n in free} }  "
                              f"IK(t0)={ {n: round(sol0[n],4) for n in free} }  "
                              f"delta_q_deg={ {n: round(math.degrees(v),1) for n,v in delta_q.items()} }  "
                              f"T_blend={t_blend*1e3:.1f}ms T_available={t_available*1e3:.1f}ms [{fit}]  "
                              f"outgoing_vel_deg_s={ {n: round(math.degrees(v),1) for n,v in outgoing_vel.items()} }",
                              file=sys.stderr)

                if use_live:
                    target_w = swing_target_world(sp, live["start"], adj_fh[0], lh)
                    seed = dict(cmd)
                    seed.update(live["seed"])
                    sol_task, resid_task = solve_swing(lead, target_w, seed, task_rows=(0, 2, 4))
                    # Activation transition: command sol_task PLUS a decaying
                    # joint-space offset (quintic, C2 at both ends) that starts
                    # EXACTLY at the outgoing table value and decays to zero --
                    # this preserves the outgoing position command at t0 and
                    # removes its geometric error gradually, instead of jumping
                    # straight to the new target.  It does not match outgoing
                    # VELOCITY (see the ACTIVATE-line check above).
                    elapsed = t_global - live["t_activate"]
                    u = min(1.0, elapsed / live["blend_T"]) if live["blend_T"] > 0 else 1.0
                    s_u = 10 * u**3 - 15 * u**4 + 6 * u**5
                    for n in free:
                        blended = sol_task[n] + (1.0 - s_u) * live["blend_delta"][n]
                        lo, hi = JOINT_LIMITS[n]
                        cmd[n] = min(hi, max(lo, blended))
                        live["seed"][n] = sol_task[n]
                    if os.environ.get("DCM_DBG"):
                        _, resid_full = solve_swing(lead, target_w, seed, task_rows=(0, 1, 2, 3, 4, 5))
                        blended_at_lim = [n for n in free if cmd[n] <= JOINT_LIMITS[n][0] + 1e-4
                                          or cmd[n] >= JOINT_LIMITS[n][1] - 1e-4]
                        prev = live["prev"] or {n: cmd[n] for n in free}
                        jdeg_s = {n: round(math.degrees(cmd[n] - prev[n]) / dt, 1) for n in free}
                        # FK of the BLENDED command (what's actually issued),
                        # not the pure IK target -- the blend temporarily
                        # sacrifices exact Cartesian tracking, so check its
                        # actual world foot height (clearance) explicitly.
                        tf_ach = lm.forward_kinematics(spec, {**base_cfg, **seed, **{n: cmd[n] for n in free}})
                        ach_pf = lm.frame_world_position(spec, tf_ach, lead + "foot_sole_center")
                        R_wp, p_wp = pelvis_pose()
                        ach_w = lm.vec_add(lm.mat_vec(R_wp, ach_pf), p_wp)
                        fz_l = foot_normal_force(foot_gid["l_"])
                        fz_r = foot_normal_force(foot_gid["r_"])
                        roll_dbg, pitch_dbg, _ = wsh.quat_rpy(data.qpos[3:7])
                        a_pit_l = lead + "ankle_pitch"
                        th = float(data.qpos[7 + JIDX[a_pit_l]])
                        thd = float(data.qvel[6 + JIDX[a_pit_l]])
                        kp_term = kp_att * (cmd[a_pit_l] - th)
                        kd_term = -kd_att * thd
                        unclipped = kp_term + kd_term
                        clipped = min(forcerng[JIDX[a_pit_l]], max(-forcerng[JIDX[a_pit_l]], unclipped))
                        clearance_flag = "GROUND-PENETRATION" if float(ach_w[2]) < -0.001 else "ok"
                        print(f"[live-swing] i={i} k={k_label:3d} sp={sp:.3f} tau={tau:.3f} u={u:.3f}  "
                              f"target_w={tuple(round(v,4) for v in target_w)} ach_w(blended)={tuple(round(float(v),4) for v in ach_w)} clearance={clearance_flag}  "
                              f"resid[task]={resid_task*1e3:.2f}mm resid[full]={resid_full*1e3:.2f}mm blended_at_limit={blended_at_lim}  "
                              f"ik={ {n: round(sol_task[n],4) for n in free} } cmd(blended)={ {n: round(cmd[n],4) for n in free} } "
                              f"deg/s={jdeg_s}  "
                              f"ankle_pitch(kp={kp_term:.2f},kd={kd_term:.2f},unclipped={unclipped:.2f},clipped={clipped:.2f})  "
                              f"Fzl={fz_l:.1f} Fzr={fz_r:.1f} roll={math.degrees(roll_dbg):.2f} pitch={math.degrees(pitch_dbg):.2f}",
                              file=sys.stderr)
                    live["prev"] = {n: cmd[n] for n in free}
                else:
                    for n, v in table_targets.items():
                        cmd[n] = v
                # roll joints track the LIPM COM-y arc (feed-forward, via weight_shift)
                _, qr = wsh.table_lookup(shift_table, max(-0.055, min(0.055, com_ref_y(tau))))
                for j in ROLLJ:
                    cmd[j] = qr[JIDX[j]]

                # --- DCM feedback -> CoP command --------------------------- #
                cxk, cyk = float(data.subtree_com[0][0]), float(data.subtree_com[0][1])
                vxk = (cxk - prev_com[0]) / dt
                vyk = (cyk - prev_com[1]) / dt
                prev_com = [cxk, cyk]
                xi_m = [cxk + vxk / w0, cyk + vyk / w0]
                xi_r = dcm_ref(i, tau)
                err = [xi_m[k2] - xi_r[k2] for k2 in (0, 1)]
                step_err = max(step_err, math.hypot(*err))
                log["dcm_err"] = max(log["dcm_err"], math.hypot(*err))
                p_cmd = [p_i[k2] + (1.0 + k_dcm / w0) * err[k2] for k2 in (0, 1)]
                cop_tau = {}
                in_double_support = warm or sp <= 0.0 or sp >= 1.0
                if in_double_support:
                    # Both feet are grounded here -- either the whole warm-up rock
                    # (no foot ever lifts), or the opening/closing double-support
                    # windows of a real step (sp<=0 before the swing foot lifts,
                    # sp>=1 once it has landed -- U17: the CODE used to switch to
                    # single-"stance"-foot realisation the instant i >= n_warm, even
                    # though the first ~20% of a real step is still measurably
                    # double support (both Fz substantial) -- so the CoP is
                    # realisable across BOTH soles, not just the nominal "stance"
                    # foot.  As the rock builds amplitude, weight can (and does)
                    # transfer fully onto one foot *before* its half-step is
                    # officially over -- realising the CoP only through the
                    # index-alternated "stance" ankle then hands control to a foot
                    # carrying ~0 N, i.e. zero authority, right when it's needed
                    # most.  Fix: drive each ankle from its OWN measured contact
                    # force and its OWN local CoP clamp (a global target beyond one
                    # sole's +-a_y just saturates that ankle at its own edge, it
                    # doesn't hand it an unrealisable target), so whichever foot is
                    # actually loaded is the one doing the work, within what its
                    # sole allows.
                    cop_sat_any = False
                    for fp_ in ("l_", "r_"):
                        sf_ = data.geom_xpos[foot_gid[fp_]]
                        Fz_ = max(0.0, foot_normal_force(foot_gid[fp_]))
                        px_l = min(float(sf_[0]) + a_x, max(float(sf_[0]) - a_x, p_cmd[0]))
                        py_l = min(float(sf_[1]) + a_y, max(float(sf_[1]) - a_y, p_cmd[1]))
                        if abs(px_l - p_cmd[0]) > 1e-4 or abs(py_l - p_cmd[1]) > 1e-4:
                            cop_sat_any = True
                        r_j, p_j = fp_ + "ankle_roll", fp_ + "ankle_pitch"
                        if is_torque[JIDX[r_j]]:
                            cop_tau[r_j] = cop_gain * (-SIDE[fp_]) * Fz_ * (py_l - float(sf_[1]))
                            cop_tau[p_j] = cop_gain * Fz_ * (px_l - float(sf_[0]))
                        else:
                            cmd[r_j] = cmd.get(r_j, 0.0) + (-SIDE[fp_]) * 1.8 * (py_l - float(sf_[1]))
                            cmd[p_j] = cmd.get(p_j, 0.0) + 1.8 * (px_l - float(sf_[0]))
                    if cop_sat_any:
                        log["cop_sat"] += 1
                else:
                    # genuine single support: the swing foot is off the ground,
                    # only the stance ankle has any authority.
                    sf = data.geom_xpos[foot_gid[stance]]
                    px_c = min(float(sf[0]) + a_x, max(float(sf[0]) - a_x, p_cmd[0]))
                    py_c = min(float(sf[1]) + a_y, max(float(sf[1]) - a_y, p_cmd[1]))
                    if abs(px_c - p_cmd[0]) > 1e-4 or abs(py_c - p_cmd[1]) > 1e-4:
                        log["cop_sat"] += 1

                    # --- realise the CoP as stance ankle torque ------------ #
                    Fz = max(0.0, foot_normal_force(foot_gid[stance]))
                    if is_torque[JIDX[a_roll]]:
                        cop_tau[a_roll] = cop_gain * (-SIDE[stance]) * Fz * (py_c - float(sf[1]))
                        cop_tau[a_pit] = cop_gain * Fz * (px_c - float(sf[0]))
                    else:
                        # position-servo fallback (U14 mode): trim the target angle
                        cmd[a_roll] += (-SIDE[stance]) * 1.8 * (py_c - float(sf[1]))
                        cmd[a_pit] += 1.8 * (px_c - float(sf[0]))

                if SWEEP_LOG is not None:
                    # pre-step state: what apply_ctrl actually saw to compute
                    # tau_P/tau_raw for THIS substep (post-step th/thd would
                    # be one substep stale for that purpose).
                    a_pit_l = lead + "ankle_pitch"
                    ai = JIDX[a_pit_l]
                    th_pre = float(data.qpos[7 + ai])
                    thd_pre = float(data.qvel[6 + ai])
                    tau_p_pre = kp_att * (cmd[a_pit_l] - th_pre)

                apply_ctrl(cmd, cop_tau)
                mujoco.mj_step(model, data)
                t_global += dt

                if SWEEP_LOG is not None:
                    SWEEP_LOG.append(dict(
                        thd=float(data.qvel[6 + ai]),
                        th=float(data.qpos[7 + ai]),
                        target=float(cmd[a_pit_l]),
                        torque_applied=float(data.ctrl[aid(a_pit_l)]),
                        fz=foot_normal_force(foot_gid[lead]),
                        thd_pre=thd_pre, tau_p_pre=tau_p_pre,
                    ))

                # --- diagnostic: does the realised CoP actually land on p_cmd? --- #
                # (per code review: independently clamping p_cmd into each foot's
                # own range does not guarantee the Fz-weighted RESULTANT lands on
                # p_cmd -- measure it directly from contact points, don't assume it.
                # Only meaningful under real load: CoP is undefined / ill-conditioned
                # near zero total contact force, so gate on a real fraction of body
                # weight actually being carried.)
                log["n_sub"] += 1
                acx, acy, fzl_, fzr_ = total_cop()
                if acx is not None and (fzl_ + fzr_) > 0.2 * total_weight:
                    log["cop_track_err"] = max(log["cop_track_err"],
                                                math.hypot(acx - p_cmd[0], acy - p_cmd[1]))

                # --- diagnostic: contact-observer classification (shadow mode) --- #
                obs = support_state(fzl_, fzr_)
                last_obs[0] = obs
                log["support_counts"][obs] += 1
                # No scheduled phase of a walk ever expects NO_SUPPORT -- that's
                # never "correct," warm-up or forward, single or double support.
                if obs == "NO_SUPPORT":
                    log["stance_unsupported_n"] += 1
                    if log["no_support_first"] is None:
                        log["no_support_first"] = (i, k_label, round(t_global, 4))
                else:
                    expect_side = None if (warm or sp <= 0.0 or sp >= 1.0) else stance
                    if expect_side is not None:
                        want = "LEFT_SUPPORT" if expect_side == "l_" else "RIGHT_SUPPORT"
                        if obs != want and obs != "DOUBLE_SUPPORT":
                            log["stance_unsupported_n"] += 1

                if record:
                    tj_ = np.array([data.actuator_force[aid(n)] for n in jn])
                    if np.any(np.abs(tj_[is_torque]) / forcerng[is_torque] > 0.999):
                        log["n_sat"] += 1

                roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
                log["tilt"] = max(log["tilt"], abs(roll), abs(pitch))
                if record:
                    tj_ = np.array([data.actuator_force[aid(n)] for n in jn])
                    log["tq"] = max(log["tq"], float(np.max(np.abs(tj_) / forcerng)))
                if max(abs(roll), abs(pitch)) > math.radians(45):
                    log["fell"] = True
                    log["fell_t"] = t_global
                    log["fell_step"] = i
                    return True
                if viewer is not None and sync:
                    viewer.sync()
                return False

            if ankle_kd_sweep and (not warm) and (i == n_warm):
                # U18 ankle-loop investigation: swing ankle is already
                # unloaded here (Fz~0) with a STATIC table target (sp=0 the
                # whole window) -- live swing IK is NOT engaged (run with
                # --ankle-kd-sweep alone).  Snapshot state once, then replay
                # a short window from that IDENTICAL state under each kd,
                # everything else (kp, model, other joints' gains, target,
                # +-2N*m limit) unchanged.  Reports and exits; does not walk.
                a_pit_l = lead + "ankle_pitch"
                qpos0 = np.copy(data.qpos)
                qvel0 = np.copy(data.qvel)
                warm0 = np.copy(data.qacc_warmstart)
                N_SWEEP = 40  # 40 * 2ms = 80ms
                print(f"\n[ankle-kd-sweep] joint={a_pit_l}  kp={kp_att}  "
                      f"N={N_SWEEP} substeps ({N_SWEEP*dt*1e3:.0f}ms)  "
                      f"static target={cur.get(a_pit_l, float('nan')):.4f}rad "
                      f"(table target this window: see below)", file=sys.stderr)
                # Torque-headroom bound (instantaneous, not a stability
                # guarantee): kd <= (2 - |tau_P|)/|qvel| when |tau_P|<2.
                # Computed from the kd=1.2 (baseline) trial's own tau_P/qvel
                # below, since tau_P barely depends on kd (it doesn't use kd
                # at all) -- report it once candidates are chosen.
                headroom_bounds = []
                for kd_trial in (1.2, 0.1, 0.05, 0.025, 0.0):
                    data.qpos[:] = qpos0
                    data.qvel[:] = qvel0
                    data.qacc_warmstart[:] = warm0
                    mujoco.mj_forward(model, data)
                    KD_OVERRIDE[a_pit_l] = kd_trial
                    SWEEP_LOG = []
                    fell_trial = False
                    for k_sw in range(N_SWEEP):
                        if run_substep(0.0, False, 0):
                            fell_trial = True
                            break
                    KD_OVERRIDE.pop(a_pit_l, None)
                    if not SWEEP_LOG:
                        print(f"  kd={kd_trial:.3f}: no data recorded (fell immediately)", file=sys.stderr)
                        continue
                    thds = [r["thd"] for r in SWEEP_LOG]
                    tors = [r["torque_applied"] for r in SWEEP_LOG]
                    errs = [r["target"] - r["th"] for r in SWEEP_LOG]
                    fzs = [r["fz"] for r in SWEEP_LOG]
                    n_sat = sum(1 for t in tors if abs(t) >= forcerng[JIDX[a_pit_l]] - 1e-6)
                    n_alt = sum(1 for a, b in zip(tors, tors[1:]) if a * b < 0)
                    if kd_trial == 1.2:
                        for r in SWEEP_LOG:
                            if abs(r["tau_p_pre"]) < 2.0 and abs(r["thd_pre"]) > 1e-6:
                                headroom_bounds.append((2.0 - abs(r["tau_p_pre"])) / abs(r["thd_pre"]))
                    print(f"  kd={kd_trial:.3f}: |thd| max={max(abs(v) for v in thds):.2f} "
                          f"mean={sum(abs(v) for v in thds)/len(thds):.2f} rad/s   "
                          f"sat_frac={n_sat/len(tors)*100:.0f}%  sign_flips={n_alt}/{len(tors)-1}  "
                          f"pos_err: peak={max(abs(v) for v in errs):.4f} final={abs(errs[-1]):.4f} rad  "
                          f"Fz range=[{min(fzs):.1f},{max(fzs):.1f}]  "
                          f"{'(FELL during window)' if fell_trial else ''}", file=sys.stderr)
                    print(f"    thd series (rad/s, every 2 substeps): "
                          f"{[round(v,2) for v in thds[::2]]}", file=sys.stderr)
                if headroom_bounds:
                    print(f"  [headroom] instantaneous kd bound (2-|tau_P|)/|qvel| over the "
                          f"kd=1.2 trial: min={min(headroom_bounds):.4f} "
                          f"median={sorted(headroom_bounds)[len(headroom_bounds)//2]:.4f}  "
                          f"(torque-headroom check, not a stability guarantee)", file=sys.stderr)
                # restore the true state so nothing downstream is corrupted,
                # then stop -- this mode reports, it does not keep walking.
                data.qpos[:] = qpos0
                data.qvel[:] = qvel0
                data.qacc_warmstart[:] = warm0
                mujoco.mj_forward(model, data)
                print("\n[ankle-kd-sweep] done -- exiting (no walk).", file=sys.stderr)
                sys.exit(0)

            for k in range(nsub):
                if run_substep(k / nsub, k % 5 == 0, k):
                    break
            else:
                # U18: confirm touchdown before exchanging leg roles -- per
                # code review, the timer used to declare a real step "done"
                # (and hand stance to the just-swung foot) whether or not that
                # foot had actually landed.  If the observer doesn't yet agree
                # the lead foot (or both feet) are supported, hold here --
                # repeating the landed/sp=1 command with tau FROZEN at
                # t_step_i (not advancing the DCM/CoM reference past the
                # step's own domain, which would just reintroduce the U16/U17
                # amplification problem) -- for a bounded extra window.  If
                # touchdown still isn't confirmed after that, this is a
                # genuine failure: mark it and stop, don't advance the step
                # counter or report a completed handoff.
                want = {"l_": "LEFT_SUPPORT", "r_": "RIGHT_SUPPORT"}[lead]
                touchdown_ok = warm or last_obs[0] in (want, "DOUBLE_SUPPORT")
                if not touchdown_ok:
                    if os.environ.get("DCM_DBG"):
                        print(f"[touchdown gate] step {i} lead={lead} ended with obs={last_obs[0]} "
                              f"(wanted {want}) -- holding up to {TOUCHDOWN_WAIT_S}s", file=sys.stderr)
                    extra_budget = int(TOUCHDOWN_WAIT_S / dt)
                    extra_used = 0
                    for k_extra in range(extra_budget):
                        extra_used = k_extra + 1
                        if run_substep(1.0, True, nsub + k_extra):
                            break
                        if last_obs[0] in (want, "DOUBLE_SUPPORT"):
                            touchdown_ok = True
                            break
                    if os.environ.get("DCM_DBG"):
                        print(f"[touchdown gate] step {i} resolved after {extra_used} extra substeps: "
                              f"ok={touchdown_ok} fell={log['fell']}", file=sys.stderr)
                    if not touchdown_ok and not log["fell"]:
                        log["fell"] = True
                        log["fail_reason"] = "no_touchdown"
                        log["fell_t"] = t_global
                        log["fell_step"] = i

            cur.pop("_cc", None)
            if log["fell"]:
                break
            fw = {p: [float(data.geom_xpos[foot_gid[p]][0]), float(data.geom_xpos[foot_gid[p]][1])]
                  for p in ("l_", "r_")}
            cur = {n: float(data.qpos[7 + idx]) for idx, n in enumerate(jn)}
            log["step_x"].append(float(data.subtree_com[0][0]))
            log["step_t"].append(t_global)
            log["dcm_err_step"].append(step_err)
            if scoped_step:
                # Stop right after the scoped step resolves -- but do NOT use
                # --steps to do this: changing n_steps changes total_steps,
                # which changes the backward-recursion boundary condition
                # (xi_ini[total_steps] = plan_p[-1]) for the WHOLE planned
                # horizon, including every warm-up step and the pre-activation
                # part of this very step.  That silently made an earlier
                # comparison invalid (confirmed: baseline run with `--steps 1`
                # alone reproduces the "different" warm-up numbers).  Halting
                # here instead keeps the full default-length plan intact, so
                # everything up to activation is byte-identical to the
                # unflagged baseline.
                log["stopped_after_scoped_step"] = True
                break

        # tail: hold + settle
        stop = dict(cur)
        tt = td = 0.0
        tc = 99
        cf0 = np.array(data.subtree_com[0][:2])
        for _ in range(int(tail / dt)):
            apply_ctrl(stop, {})
            mujoco.mj_step(model, data)
            r2, p2, _ = wsh.quat_rpy(data.qpos[3:7])
            tt = max(tt, abs(r2), abs(p2))
            td = max(td, float(np.linalg.norm(np.array(data.subtree_com[0][:2]) - cf0)))
            tc = min(tc, foot_corners(foot_gid["l_"]), foot_corners(foot_gid["r_"]))
        log["tail_tilt"], log["tail_drift"], log["tail_corners"] = tt, td, tc
        log["total_advance"] = float(data.subtree_com[0][0]) - com_x0
        return log

    # ---------------------------------------------------------------- #
    if view:
        try:
            import mujoco.viewer
            import time
        except ImportError:
            print("error: mujoco.viewer unavailable", file=sys.stderr)
            return 2
        print("\nviewer: DCM-tracking walk (warm-up + forward) on a loop. close to stop.")
        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running():
                walk_sim(record=False, viewer=v)
                for _ in range(60):
                    if not v.is_running():
                        break
                    time.sleep(1 / 30)
        return 0

    print(f"DCM-tracking walk (U17)  base '{base_pose}'  {model_name}   "
          f"{'TORQUE ankles' if torque_ankles else 'position ankles (U14 mode)'}")
    print(f"{m_total:.2f} kg  omega0 {w0:.2f}  |  {n_warm} warm-up steps (T {t_warm:.2f}s) + "
          f"{n_steps} forward steps (T {t_step:.2f}s), stride {1e3*stride:.0f}mm -> "
          f"{1e3*stride/t_step:.0f} mm/s  |  "
          f"DCM k {k_dcm:.1f}, ankle PD {kp_att:.0f}/{kd_att:.1f}, CoP-tau gain {cop_gain:.1f}")

    log = walk_sim()
    v_cmd = stride / t_step
    sx = np.array(log["step_x"][n_warm:]); stt = np.array(log["step_t"][n_warm:])
    seg = np.diff(sx) / np.clip(np.diff(stt), 1e-6, None) if len(sx) > 1 else np.array([0.0])
    mean_v = float(np.mean(seg))
    done = log.get("fell_step", total_steps)

    fail_tag = f", {log['fail_reason']}" if log.get("fail_reason") else ""
    print(f"\n  steps: {done}/{total_steps} " + (f"(FELL at t={log.get('fell_t',0):.2f}s, "
          f"{'warm-up' if done < n_warm else 'forward'} step {done}{fail_tag})" if log["fell"] else "(all)"))
    print(f"  forward advance {1e3*log['total_advance']:.0f} mm   mean speed {1e3*mean_v:.0f} mm/s "
          f"(commanded {1e3*v_cmd:.0f})")
    print(f"  peak |DCM error| {1e3*log['dcm_err']:.0f} mm   peak tilt {math.degrees(log['tilt']):.1f}°   "
          f"peak torque {100*log['tq']:.0f}%   CoP clamp hits {log['cop_sat']}")
    sat_frac = log["n_sat"] / max(1, log["n_sub"])
    print(f"  peak |actual CoP - p_cmd| {1e3*log['cop_track_err']:.0f} mm (loaded substeps only)   "
          f"actuator saturated {100*sat_frac:.0f}% of substeps")
    sc = log["support_counts"]
    n_all = max(1, sum(sc.values()))
    print(f"  contact observer (shadow): DS {100*sc['DOUBLE_SUPPORT']/n_all:.0f}%  "
          f"L {100*sc['LEFT_SUPPORT']/n_all:.0f}%  R {100*sc['RIGHT_SUPPORT']/n_all:.0f}%  "
          f"NONE {100*sc['NO_SUPPORT']/n_all:.0f}%"
          + (f"  (first NO_SUPPORT: step {log['no_support_first'][0]}, "
             f"k={log['no_support_first'][1]}, t={log['no_support_first'][2]}s)"
             if log["no_support_first"] else "")
          + f"  |  scheduled stance actually unsupported: {log['stance_unsupported_n']} substeps")
    if log["dcm_err_step"]:
        print(f"  per-step DCM error: {', '.join(f'{1e3*e:.0f}' for e in log['dcm_err_step'])} mm  "
              f"(| = warm-up/forward boundary after {n_warm})")
    print(f"  stop + stand {tail:.0f}s: tilt {math.degrees(log['tail_tilt']):.1f}°, "
          f"drift {1e3*log['tail_drift']:.1f} mm, {log['tail_corners']} corners/foot")

    walked = (not log["fell"]) and mean_v > MIN_SPEED_FRAC * v_cmd
    tracked = log["dcm_err"] < MAX_DCM_ERR
    upright = log["tilt"] < MAX_TILT and log["tq"] <= MAX_TQ
    stopped = (log["tail_tilt"] < MAX_TILT and log["tail_drift"] < TAIL_DRIFT and log["tail_corners"] >= 3)
    ok = walked and tracked and upright and stopped

    results = {"model": model_name, "torque_ankles": torque_ankles,
               "n_warm": n_warm, "n_steps": n_steps, "steps_done": done,
               "t_step": t_step, "stride": stride, "v_cmd": v_cmd, "v_mean": mean_v,
               "total_advance_m": log["total_advance"], "peak_dcm_error_mm": 1e3 * log["dcm_err"],
               "peak_tilt_deg": math.degrees(log["tilt"]), "peak_torque_frac": log["tq"],
               "fell": log["fell"], "milestone_met": bool(ok)}
    if json_path:
        json.dump(results, open(json_path, "w", encoding="utf-8"), indent=2)
        print(f"\nsummary -> {json_path}")
    if baseline_path and os.path.exists(baseline_path):
        b = json.load(open(baseline_path, encoding="utf-8"))
        print(f"\nvs baseline: met {b.get('milestone_met')} -> {ok}, "
              f"advance {1e3*b.get('total_advance_m',0):.0f} -> {1e3*log['total_advance']:.0f} mm")

    print("\n" + "=" * 74)
    if ok:
        print(f"MILESTONE MET: warm-start into the lateral limit cycle, then {n_steps} forward "
              f"steps tracking the planned DCM ({1e3*log['total_advance']:.0f} mm at "
              f"~{1e3*mean_v:.0f} mm/s, peak DCM error {1e3*log['dcm_err']:.0f} mm), then stop and stand.")
    else:
        seg_lbl = "the warm-up" if done < n_warm else "the forward walk"
        print(f"MILESTONE NOT MET: got {done}/{total_steps} steps"
              + (f", fell during {seg_lbl}" if log["fell"] else "") + f"; peak DCM error {1e3*log['dcm_err']:.0f} mm.")
        if torque_ankles:
            print("  U15 unblocked the actuation; U16 fixed gait initiation (the rocking")
            print("  warm-up now survives cleanly, however many rocks); U17 shortened the")
            print("  forward-step duration toward U13's T_min (0.5s -> 0.22s, far less DCM")
            print("  error amplification per step) and realises the CoP across BOTH soles")
            print("  during a real step's own opening/closing double-support windows, not")
            print("  just the warm-up's. Peak DCM error is now 44mm (vs U16's 143mm) and she")
            print("  clears the ENTIRE warm-up plus the first real forward step every time.")
            print("  What still fails: the SECOND forward step -- once real single-support")
            print("  strides chain back-to-back, the same per-step error growth that U16 fixed")
            print("  for the warm-up reappears between forward steps. Remaining: extend the")
            print("  same fix to forward-step-to-forward-step handoffs, or a ZMP-preview / MPC")
            print("  formulation over a multi-step horizon. Not RL, not hardware.")
        else:
            print("  (position-ankle mode -- U14 showed this cannot place the CoP)")
    print("  LIPM plan + DCM feedback + capture-point step adjustment + torque-ankle CoP; no RL. "
          "(provisional masses / gains / friction / foot size)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--t-step", type=float, default=None)
    ap.add_argument("--warmup-t-step", type=float, default=None,
                    help="duration of each warm-up rocking half-step (s); short on purpose, see U16 notes")
    ap.add_argument("--no-torque-ankles", action="store_true",
                    help="keep position servos on the ankles (U14 mode -- fails)")
    ap.add_argument("--live-swing-test", action="store_true",
                    help="U18 experiment: world-frame swing IK, SCOPED to the first real "
                         "swing only (warm-up + opening double support untouched); "
                         "activates when genuine single support begins, stays active "
                         "through touchdown confirmation.")
    ap.add_argument("--ankle-kd-sweep", action="store_true",
                    help="U18 experiment: at the first real step's k=0 (swing ankle already "
                         "unloaded, static target), snapshot state and replay a short window "
                         "under kd in {1.2, 0.6, 0.3, 0} for that one joint only, everything "
                         "else unchanged.  Prints a comparison and exits -- does not walk.")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args(argv)
    return run(args.config, args.steps, args.t_step, args.warmup_t_step, not args.no_torque_ankles,
               args.view, args.json, args.baseline, args.live_swing_test, args.ankle_kd_sweep)


if __name__ == "__main__":
    sys.exit(main())
