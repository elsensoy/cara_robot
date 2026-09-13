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
real forward step.  The walk still doesn't complete -- see the printed report
for exactly where it now fails (the double-support -> single-support handoff).

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


def run(config, n_steps_arg, t_step_arg, t_warm_arg, torque_ankles, view, json_path, baseline_path):
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

    def foot_corners(fg):
        return sum(1 for i in range(data.ncon)
                   if {data.contact[i].geom1, data.contact[i].geom2} == {fg, floor_gid})

    def apply_ctrl(cmd, cop_axis_tau):
        """cmd: {joint: position target}.  cop_axis_tau: {ankle_joint: extra torque}.
        Position joints get the target; torque ankles get a software attitude PD
        + any CoP torque, clamped to the effort limit."""
        out = np.zeros(len(jn))
        for i, n in enumerate(jn):
            if is_torque[i]:
                th = float(data.qpos[7 + i])
                thd = float(data.qvel[6 + i])
                tau = kp_att * (cmd[n] - th) - kd_att * thd + cop_axis_tau.get(n, 0.0)
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
               "step_x": [com_x0], "step_t": [0.0], "dcm_err_step": [], "cop_sat": 0}
        t_global = 0.0
        cur = dict(FULL0)

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
            for k in range(nsub):
                p = k / nsub
                tau = p * t_step_i
                sp = 0.0 if p <= ss_lo else (1.0 if p >= ss_hi else (p - ss_lo) / (ss_hi - ss_lo))
                clr = lh * math.sin(math.pi * sp) if 0.0 < sp < 1.0 else 0.0

                cmd = dict(cur)
                # swing sagittal joints
                wc = float(data.geom_xpos[foot_gid[lead]][2]) - foot_z0[lead]
                cur.setdefault("_cc", 0.0)
                cc = min(C_TOP, max(0.0, cur["_cc"] + CG * (clr - wc)))
                cur["_cc"] = cc
                for n, v in zip(free, swing_lookup(grid, sp, cc)):
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
                if warm:
                    # Double support the whole time (no foot ever lifts): the CoP is
                    # realisable across BOTH soles, not just the nominal "stance" foot
                    # for this half-step.  As the rock builds amplitude, weight can
                    # (and does) transfer fully onto one foot *before* its half-step
                    # is officially over -- realising the CoP only through the
                    # index-alternated "stance" ankle then hands control to a foot
                    # carrying ~0 N, i.e. zero authority, right when it's needed most.
                    # Fix: drive each ankle from its OWN measured contact force and
                    # its OWN local CoP clamp (a global target beyond one sole's
                    # +-a_y just saturates that ankle at its own edge, it doesn't
                    # hand it an unrealisable target), so whichever foot is actually
                    # loaded is the one doing the work, within what its sole allows.
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

                apply_ctrl(cmd, cop_tau)
                mujoco.mj_step(model, data)
                t_global += dt

                roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
                log["tilt"] = max(log["tilt"], abs(roll), abs(pitch))
                if record:
                    tj_ = np.array([data.actuator_force[aid(n)] for n in jn])
                    log["tq"] = max(log["tq"], float(np.max(np.abs(tj_) / forcerng)))
                if max(abs(roll), abs(pitch)) > math.radians(45):
                    log["fell"] = True
                    log["fell_t"] = t_global
                    log["fell_step"] = i
                    break
                if viewer is not None and k % 5 == 0:
                    viewer.sync()
            cur.pop("_cc", None)
            if log["fell"]:
                break
            fw = {p: [float(data.geom_xpos[foot_gid[p]][0]), float(data.geom_xpos[foot_gid[p]][1])]
                  for p in ("l_", "r_")}
            cur = {n: float(data.qpos[7 + idx]) for idx, n in enumerate(jn)}
            log["step_x"].append(float(data.subtree_com[0][0]))
            log["step_t"].append(t_global)
            log["dcm_err_step"].append(step_err)

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

    print(f"DCM-tracking walk (U16)  base '{base_pose}'  {model_name}   "
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

    print(f"\n  steps: {done}/{total_steps} " + (f"(FELL at t={log.get('fell_t',0):.2f}s, "
          f"{'warm-up' if done < n_warm else 'forward'} step {done})" if log["fell"] else "(all)"))
    print(f"  forward advance {1e3*log['total_advance']:.0f} mm   mean speed {1e3*mean_v:.0f} mm/s "
          f"(commanded {1e3*v_cmd:.0f})")
    print(f"  peak |DCM error| {1e3*log['dcm_err']:.0f} mm   peak tilt {math.degrees(log['tilt']):.1f}°   "
          f"peak torque {100*log['tq']:.0f}%   CoP clamp hits {log['cop_sat']}")
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
            print("  U15 unblocked the actuation (torque ankles, byte-identical default,")
            print("  standing verified) and U16 fixed gait initiation itself: a rest-start")
            print("  rocking warm-up now needs its own (short) step duration and its own")
            print("  double-support CoP realisation -- driving each ankle from its OWN")
            print("  measured contact force, not a single index-alternated 'stance' foot --")
            print("  and it survives cleanly (DCM error single-digit-mm to ~40 mm across every")
            print("  rock, however many).  What still fails is narrower now: the HANDOFF from")
            print("  double support into the first genuine single-support forward step (the")
            print("  first real foot liftoff), where the DCM error jumps an order of magnitude.")
            print("  Remaining: settle that handoff specifically (e.g. widen double support")
            print("  right at first liftoff) or a ZMP-preview / MPC formulation.  Not RL, not hardware.")
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
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args(argv)
    return run(args.config, args.steps, args.t_step, args.warmup_t_step, not args.no_torque_ankles,
               args.view, args.json, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
