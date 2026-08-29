#!/usr/bin/env python3
"""U14 -- a DCM-tracking walk.

U13 (`walk_model.py`) showed a dynamically-consistent walk is within Cara's
morphology (lateral step time >= ~0.22 s).  U14 builds the controller U13 called
for: **plan the CoP + footholds and the divergent-component (DCM / capture
point) reference from the LIPM, then track that reference** -- no fixed pose
cycle to replay.

  * PLAN (offline, from `walk_model.LIPM`): a footstep sequence, the CoP held at
    each foot centre through single support, and the DCM reference by backward
    recursion  xi_ini[i] = p[i] + (xi_ini[i+1] - p[i]) e^{-w0 T}.
  * FEED-FORWARD (per step): the swing leg tracks a trajectory to the planned
    foothold (`gait.py`'s per-step IK table); the four roll joints track a
    lateral COM sway from `weight_shift`'s inverted table.
  * DCM FEEDBACK (per sim step): measure  xi = COM + COM_vel / w0 , command the
    CoP  p_cmd = p_ref + (1 + k/w0)(xi_meas - xi_ref) , clamp it into the stance
    foot, and realise it as stance ankle_roll / ankle_pitch trims (the ankle
    strategy -- U9's roll term, generalised to 2-D and driven by the DCM error).
  * STEP ADJUSTMENT (per footfall): shift the next foothold to null the DCM
    error predicted at end of step -- the capture point *is* a foothold target.

No fixed pose cycle, no RL.  Failure cases are reported, not hidden.

Requires `mujoco` (brings numpy).  Prints SKIPPED / exits 0 without it.

Usage:
    python3 dcm_walk.py                     # full body, N steps
    python3 dcm_walk.py --steps 10
    python3 dcm_walk.py --t-step 0.35
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


def run(config, n_steps_arg, t_step_arg, view, json_path, baseline_path):
    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0
    import generate_mjcf

    spec = lm.load_spec(config)
    model_name = spec["meta"]["name"]
    xml = generate_mjcf.build_mjcf(spec, dynamic=True)
    on_disk = os.path.join(_MJCF_DIR, model_name + "_dynamic.xml")
    if os.path.exists(on_disk) and open(on_disk, encoding="utf-8").read() != xml:
        print(f"WARNING: {on_disk} is stale -- fresh render used here\n")

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
    n_steps = int(n_steps_arg if n_steps_arg is not None else dw.get("n_steps", 8))
    ds_frac = float(dw.get("double_support_frac", 0.20))
    lift_h = float(dw.get("lift_height", 0.010))
    sway = float(dw.get("sway", 0.030))
    CG = float(dw.get("clearance_gain", 0.015))
    k_dcm = float(dw.get("k_dcm", 3.0))
    g_ankle_roll = float(dw.get("realize_gain_roll", 12.0))
    g_ankle_pitch = float(dw.get("realize_gain_pitch", 12.0))
    step_adj_gain = float(dw.get("step_adjust_gain", 1.0))
    lead_in = float(dw.get("lead_in_seconds", 2.0))
    settle = float(dw.get("settle_seconds", 1.5))
    tail = float(dw.get("tail_seconds", 3.0))
    acc = dw.get("accept", {}) or {}
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 10.0)))
    MAX_DCM_ERR = float(acc.get("max_dcm_error", 0.020))
    MIN_SPEED_FRAC = float(acc.get("min_speed_frac", 0.5))
    MAX_TQ = float(acc.get("max_torque_frac", 1.0))
    TAIL_DRIFT = float(acc.get("tail_hold_drift", 0.012))

    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)[base_pose]
    FULL0 = {n: float(base_cfg.get(n, 0.0)) for n in jn}
    nominal_ctrl = [FULL0[n] for n in jn]

    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose)
    foot_gid = {"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")}
    floor_gid = gid("floor")
    forcerng = np.array([model.actuator_forcerange[aid(n)][1] for n in jn])
    m_total = float(sum(model.body_mass))

    tf0 = lm.forward_kinematics(spec, base_cfg)
    SOLE0 = {p: lm.frame_world_position(spec, tf0, p + "foot_sole_center") for p in ("l_", "r_")}
    ROT0 = {p: tf0[p + "foot"][0] for p in ("l_", "r_")}
    SAG = {p: [p + j for j in ("hip_pitch", "knee_pitch", "ankle_pitch")] for p in ("l_", "r_")}
    ROLLJ = ["l_hip_roll", "l_ankle_roll", "r_hip_roll", "r_ankle_roll"]
    ROLL_IDX = {j: jn.index(j) for j in ROLLJ}
    AP_IDX = {p: jn.index(p + "ankle_pitch") for p in ("l_", "r_")}
    Z0 = SOLE0["l_"][2]
    X0 = SOLE0["l_"][0]

    _, com0, _ = lm.center_of_mass(spec, base_cfg)
    z_com = com0[2] - Z0
    lip = LIPM(z_com, g)
    w0 = lip.w
    sym = lm.resolve_symbols(spec)
    a_x, a_y = 0.5 * float(sym["foot_len"]), 0.5 * float(sym["foot_width"])
    s_half = float(sym["w_hip_half"])

    shift_table = wsh.build_ik_table(spec, base_cfg, sway + 0.02, 61)

    # at stand_nominal the whole-body COM sits `com_bias` from the foot centre
    # (mostly the +x foot_x_off lean).  The LIPM CoP is where the COM balances, so
    # shift every planned CoP by this so "COM over the plan" == standing naturally.
    com_bias = (float(com0[0]) - X0, float(com0[1]) - 0.0)

    def smooth01(u):
        u = min(1.0, max(0.0, u))
        return u * u * (3.0 - 2.0 * u)

    # ------------------------------------------------------------------ #
    # PLAN: nominal footholds (world), CoP per step, DCM reference
    # ------------------------------------------------------------------ #
    # step i (0-based): swing = l_ if i even else r_; it lands `stride` ahead of
    # the current stance foot.  foot y alternates about +-s_half.
    foot_w = {"l_": [X0, +s_half], "r_": [X0, -s_half]}   # current world foot centres
    plan_p = []            # effective CoP (world xy) held during single support of step i
    for i in range(n_steps):
        lead = "l_" if i % 2 == 0 else "r_"
        stance = OTHER[lead]
        plan_p.append([foot_w[stance][0] + com_bias[0], foot_w[stance][1] + com_bias[1]])
        foot_w[lead] = [foot_w[stance][0] + stride, foot_w[lead][1]]
    plan_foot = {i: None for i in range(n_steps)}         # (unused placeholder)

    # DCM reference: backward recursion.  end at rest over the final CoP.
    eT = math.exp(w0 * t_step)
    xi_ini = [None] * (n_steps + 1)
    xi_ini[n_steps] = [plan_p[-1][0], plan_p[-1][1]]      # final DCM = last CoP (at rest)
    for i in range(n_steps - 1, -1, -1):
        p = plan_p[i]
        xi_ini[i] = [p[k] + (xi_ini[i + 1][k] - p[k]) / eT for k in (0, 1)]

    def dcm_ref(i, tau):
        """DCM reference at time tau into single support of step i (CoP = plan_p[i])."""
        p = plan_p[i]
        e = math.exp(w0 * tau)
        xi = [p[k] + (xi_ini[i][k] - p[k]) * e for k in (0, 1)]
        xid = [w0 * (xi[k] - p[k]) for k in (0, 1)]
        return xi, xid

    # ------------------------------------------------------------------ #
    # swing-leg table (per step, current pelvis frame): progress s x clearance c
    # ------------------------------------------------------------------ #
    def build_swing_table(lead, roll_cfg, x0_pf, xt_pf, y_pf, ns=11, nc=6):
        free = SAG[lead]
        q0 = {**base_cfg, **roll_cfg}
        z0 = lm.frame_world_position(spec, lm.forward_kinematics(spec, q0),
                                     lead + "foot_sole_center")[2]
        rot = ROT0[lead]
        grid = []
        for i in range(ns):
            s = i / (ns - 1)
            row, q = [], dict(q0)
            for kk in range(nc):
                c = C_TOP * kk / (nc - 1)
                sol, _r = lm.leg_ik(spec, lead, lead + "foot_sole_center",
                                    (x0_pf + s * (xt_pf - x0_pf), y_pf, z0 + c), rot, q,
                                    free_joints=free, task_rows=[0, 2, 4], iters=180)
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

    # ------------------------------------------------------------------ #
    # contact / state helpers
    # ------------------------------------------------------------------ #
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

    # foot geom z at rest -- the swing-clearance datum
    mujoco.mj_resetDataKeyframe(model, data, kid)
    for _ in range(int(0.3 / dt)):
        data.ctrl[:] = nominal_ctrl
        mujoco.mj_step(model, data)
    foot_z0 = {p: float(data.geom_xpos[foot_gid[p]][2]) for p in ("l_", "r_")}

    # ------------------------------------------------------------------ #
    # the walk
    # ------------------------------------------------------------------ #
    def walk_sim(record=True, viewer=None):
        mujoco.mj_resetDataKeyframe(model, data, kid)
        for _ in range(int(settle / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)

        fw = {"l_": [float(data.geom_xpos[foot_gid['l_']][0]), float(data.geom_xpos[foot_gid['l_']][1])],
              "r_": [float(data.geom_xpos[foot_gid['r_']][0]), float(data.geom_xpos[foot_gid['r_']][1])]}
        com_x0 = float(data.subtree_com[0][0])
        prev_com = [float(data.subtree_com[0][0]), float(data.subtree_com[0][1])]

        # lead-in: settle, then shift the COM laterally to where the DCM plan
        # starts (xi_ini[0]) so step 0 begins consistent with the reference
        for _ in range(int(0.7 * lead_in / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)
        # cap the lead-in shift to the double-support weight-shift envelope
        # (~20 mm) -- xi_ini[0] can be well past it, which topples her before she
        # walks.  The DCM feedback + step adjustment build the rock up from here.
        _, q_start = wsh.table_lookup(shift_table, max(-0.018, min(0.018, xi_ini[0][1])))
        for k in range(int(0.3 * lead_in / dt)):
            f = smooth01(k / max(1, int(0.3 * lead_in / dt)))
            data.ctrl[:] = [(1 - f) * nominal_ctrl[j] + f * q_start[j] for j in range(len(jn))]
            mujoco.mj_step(model, data)
        prev_com = [float(data.subtree_com[0][0]), float(data.subtree_com[0][1])]

        log = {"tilt": 0.0, "tq": 0.0, "dcm_err": 0.0, "fell": False,
               "step_x": [com_x0], "step_t": [0.0], "dcm_err_series": []}
        t_global = 0.0
        cur_cfg = {n: float(data.qpos[7 + idx]) for idx, n in enumerate(jn)}

        for i in range(n_steps):
            lead = "l_" if i % 2 == 0 else "r_"
            stance = OTHER[lead]
            a_roll, h_roll = stance + "ankle_roll", stance + "hip_roll"
            p_i = plan_p[i]

            # --- step foothold adjustment from the measured DCM ------------- #
            # predict the DCM at end of this step; if it misses its reference
            # xi_ini[i+1], shift the NEXT CoP (= next foothold + com_bias) by the
            # miss so the following step's LIPM still closes.  Capture point ->
            # foothold.
            cx, cy = float(data.subtree_com[0][0]), float(data.subtree_com[0][1])
            vx = (cx - prev_com[0]) / dt
            vy = (cy - prev_com[1]) / dt
            xi_now = [cx + vx / w0, cy + vy / w0]
            xi_eos_pred = [p_i[k] + (xi_now[k] - p_i[k]) * eT for k in (0, 1)]
            xi_ref_eos = xi_ini[i + 1]
            nom_next_cop = (plan_p[i + 1] if i + 1 < n_steps
                            else [fw[stance][0] + stride + com_bias[0], fw[lead][1] + com_bias[1]])
            adj_next_cop = [nom_next_cop[k] + step_adj_gain * (xi_eos_pred[k] - xi_ref_eos[k])
                            for k in (0, 1)]
            adj_fh = [adj_next_cop[0] - com_bias[0], adj_next_cop[1] - com_bias[1]]
            nom_fh = [fw[stance][0] + stride, fw[lead][1]]
            adj_fh[0] = min(nom_fh[0] + 0.025, max(nom_fh[0] - 0.025, adj_fh[0]))
            adj_fh[1] = min(fw[lead][1] + 0.020, max(fw[lead][1] - 0.020, adj_fh[1]))

            px_now = float(data.qpos[0])

            # --- per-step LIPM roll-out from the MEASURED COM state --------- #
            # this is the feed-forward COM-y trajectory the roll joints track --
            # the pendulum arc, not a hand-picked half-sine (that was U12's bug).
            cx0, cy0 = float(data.subtree_com[0][0]), float(data.subtree_com[0][1])
            vx0 = (cx0 - prev_com[0]) / dt
            vy0 = (cy0 - prev_com[1]) / dt

            def com_ref(tau):
                c, sh = math.cosh(w0 * tau), math.sinh(w0 * tau)
                return (p_i[0] + (cx0 - p_i[0]) * c + (vx0 / w0) * sh,
                        p_i[1] + (cy0 - p_i[1]) * c + (vy0 / w0) * sh)

            # --- swing table to the adjusted foothold (pelvis frame) --------- #
            x0_pf = fw[lead][0] - px_now
            xt_pf = adj_fh[0] - px_now
            y_pf = lm.frame_world_position(spec, lm.forward_kinematics(spec, base_cfg),
                                           lead + "foot_sole_center")[1]
            grid, free = build_swing_table(lead, {}, x0_pf, xt_pf, y_pf)

            cmd = dict(cur_cfg)
            st = {"ccmd": 0.0}
            nsub = int(t_step / dt)
            ss_lo, ss_hi = ds_frac, 1.0 - ds_frac
            for k in range(nsub):
                p = k / nsub
                tau = p * t_step
                sp = 0.0 if p <= ss_lo else (1.0 if p >= ss_hi else (p - ss_lo) / (ss_hi - ss_lo))
                clear_target = lift_h * math.sin(math.pi * sp) if 0.0 < sp < 1.0 else 0.0
                wc = float(data.geom_xpos[foot_gid[lead]][2]) - foot_z0[lead]
                st["ccmd"] = min(C_TOP, max(0.0, st["ccmd"] + CG * (clear_target - wc)))
                for n, v in zip(free, swing_lookup(grid, sp, st["ccmd"])):
                    cmd[n] = v

                # feed-forward: roll joints track the LIPM COM-y arc (via weight_shift)
                _, qroll = wsh.table_lookup(shift_table, max(-0.049, min(0.049, com_ref(tau)[1])))
                for j in ROLLJ:
                    cmd[j] = qroll[jn.index(j)]

                # --- DCM feedback -> CoP command -> stance ankle trims ------- #
                cx, cy = float(data.subtree_com[0][0]), float(data.subtree_com[0][1])
                vx = (cx - prev_com[0]) / dt
                vy = (cy - prev_com[1]) / dt
                prev_com = [cx, cy]
                xi_meas = [cx + vx / w0, cy + vy / w0]
                xi_r, _xid_r = dcm_ref(i, tau)
                err = [xi_meas[k] - xi_r[k] for k in (0, 1)]
                log["dcm_err"] = max(log["dcm_err"], math.hypot(*err))
                p_cmd = [p_i[k] + (1.0 + k_dcm / w0) * err[k] for k in (0, 1)]
                sf = data.geom_xpos[foot_gid[stance]]
                p_cmd[0] = min(float(sf[0]) + a_x, max(float(sf[0]) - a_x, p_cmd[0]))
                p_cmd[1] = min(float(sf[1]) + a_y, max(float(sf[1]) - a_y, p_cmd[1]))

                ctrl = [cmd[n] for n in jn]
                # CoP command -> stance ankle trims (tau ~= Fz*d, kp=30 -> dtheta ~= 1.5*d)
                ctrl[ROLL_IDX[a_roll]] += (-SIDE[stance]) * g_ankle_roll * (p_cmd[1] - float(sf[1]))
                ctrl[ROLL_IDX[h_roll]] += (-SIDE[stance]) * 0.5 * g_ankle_roll * (p_cmd[1] - float(sf[1]))
                ctrl[AP_IDX[stance]] += -g_ankle_pitch * (p_cmd[0] - float(sf[0]))

                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
                t_global += dt

                roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
                log["tilt"] = max(log["tilt"], abs(roll), abs(pitch))
                if record:
                    tau_j = np.array([data.actuator_force[aid(n)] for n in jn])
                    log["tq"] = max(log["tq"], float(np.max(np.abs(tau_j) / forcerng)))
                if max(abs(roll), abs(pitch)) > math.radians(45):
                    log["fell"] = True
                    log["fell_t"] = t_global
                    log["fell_step"] = i
                    break
                if viewer is not None and k % 6 == 0:
                    viewer.sync()
            if log["fell"]:
                break

            # commit: update working foot positions + current config
            fw[lead] = [float(data.geom_xpos[foot_gid[lead]][0]), float(data.geom_xpos[foot_gid[lead]][1])]
            fw[stance] = [float(data.geom_xpos[foot_gid[stance]][0]), float(data.geom_xpos[foot_gid[stance]][1])]
            cur_cfg = {n: float(data.qpos[7 + idx]) for idx, n in enumerate(jn)}
            log["step_x"].append(float(data.subtree_com[0][0]))
            log["step_t"].append(t_global)
            log["dcm_err_series"].append(math.hypot(*err))

        # tail: hold the last config and let it settle
        stop_ctrl = [cur_cfg.get(n, 0.0) for n in jn]
        tail_tilt = tail_drift = 0.0
        tcorn = 99
        cf0 = np.array(data.subtree_com[0][:2])
        for _ in range(int(tail / dt)):
            data.ctrl[:] = stop_ctrl
            mujoco.mj_step(model, data)
            r2, p2, _ = wsh.quat_rpy(data.qpos[3:7])
            tail_tilt = max(tail_tilt, abs(r2), abs(p2))
            tail_drift = max(tail_drift, float(np.linalg.norm(np.array(data.subtree_com[0][:2]) - cf0)))
            tcorn = min(tcorn, foot_corners(foot_gid["l_"]), foot_corners(foot_gid["r_"]))
        log["tail_tilt"], log["tail_drift"], log["tail_corners"] = tail_tilt, tail_drift, tcorn
        log["total_advance"] = float(data.subtree_com[0][0]) - com_x0
        return log

    if view:
        try:
            import mujoco.viewer
            import time
        except ImportError:
            print("error: mujoco.viewer unavailable", file=sys.stderr)
            return 2
        print("\nviewer: DCM-tracking walk on a loop. close the window to stop.")
        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running():
                walk_sim(record=False, viewer=v)
                for _ in range(60):
                    if not v.is_running():
                        break
                    time.sleep(1 / 30)
        return 0

    print(f"DCM-tracking walk (U14)  base '{base_pose}'  {model_name}")
    print(f"{m_total:.2f} kg  |  omega0 {w0:.2f} rad/s (T_min~{2/w0*math.atanh(max(1e-3,1-a_y/s_half)):.2f}s)  "
          f"|  {n_steps} steps, T {t_step:.2f}s, stride {1e3*stride:.0f}mm -> {1e3*stride/t_step:.0f} mm/s  "
          f"|  DCM gain k {k_dcm:.1f}")

    log = walk_sim()
    v_cmd = stride / t_step
    sx = np.array(log["step_x"]); stt = np.array(log["step_t"])
    seg = np.diff(sx) / np.clip(np.diff(stt), 1e-6, None)
    mean_v = float(np.mean(seg)) if len(seg) else 0.0

    print(f"\n  steps completed: {log.get('fell_step', n_steps)} / {n_steps}"
          + (f"   (FELL at t={log.get('fell_t',0):.2f}s)" if log["fell"] else ""))
    print(f"  total COM advance: {1e3*log['total_advance']:.0f} mm   "
          f"mean forward speed {1e3*mean_v:.0f} mm/s (commanded {1e3*v_cmd:.0f})")
    print(f"  peak |DCM error| {1e3*log['dcm_err']:.0f} mm   peak tilt {math.degrees(log['tilt']):.1f}°   "
          f"peak torque {100*log['tq']:.0f}%")
    if log["dcm_err_series"]:
        print(f"  per-step DCM error: {', '.join(f'{1e3*e:.0f}' for e in log['dcm_err_series'])} mm")
    print(f"  stop + stand {tail:.0f}s: tilt {math.degrees(log['tail_tilt']):.1f}°, "
          f"drift {1e3*log['tail_drift']:.1f} mm, {log['tail_corners']} corners/foot")

    walked = (not log["fell"]) and mean_v > MIN_SPEED_FRAC * v_cmd
    tracked = log["dcm_err"] < MAX_DCM_ERR
    upright = log["tilt"] < MAX_TILT and log["tq"] <= MAX_TQ
    stopped = (log["tail_tilt"] < MAX_TILT and log["tail_drift"] < TAIL_DRIFT
               and log["tail_corners"] >= 3)
    ok = walked and tracked and upright and stopped

    results = {"model": model_name, "n_steps": n_steps, "steps_completed": log.get("fell_step", n_steps),
               "t_step": t_step, "stride": stride, "v_cmd": v_cmd, "v_mean": mean_v,
               "total_advance_m": log["total_advance"], "peak_dcm_error_m": log["dcm_err"],
               "peak_tilt_deg": math.degrees(log["tilt"]), "peak_torque_frac": log["tq"],
               "fell": log["fell"], "milestone_met": bool(ok),
               "peak_dcm_error_mm": 1e3 * log["dcm_err"], "xi_ini0_y_mm": 1e3 * xi_ini[0][1]}
    if json_path:
        json.dump(results, open(json_path, "w", encoding="utf-8"), indent=2)
        print(f"\nsummary -> {json_path}")
    if baseline_path and os.path.exists(baseline_path):
        base = json.load(open(baseline_path, encoding="utf-8"))
        print(f"\nvs baseline: milestone {base.get('milestone_met')} -> {ok}, "
              f"advance {1e3*base.get('total_advance_m',0):.0f} -> {1e3*log['total_advance']:.0f} mm")

    xi0y = xi_ini[0][1]
    print("\n" + "=" * 74)
    if ok:
        print(f"MILESTONE MET: Cara walks {n_steps} steps continuously by tracking the planned "
              f"DCM ({1e3*log['total_advance']:.0f} mm at ~{1e3*mean_v:.0f} mm/s), peak DCM error "
              f"{1e3*log['dcm_err']:.0f} mm, then stops and stands.")
    else:
        print(f"MILESTONE NOT MET: walked {log.get('fell_step', 0)}/{n_steps} steps "
              f"({'fell' if log['fell'] else 'too slow'}), peak DCM error {1e3*log['dcm_err']:.0f} mm.")
        print("  The PLANNER is right (U13 validated the DCM trajectory) and the feedback law")
        print("  (p_cmd = p_ref + (1+k/ω₀)·e) + capture-point step adjustment are in place -- the")
        print("  blocker is REALISATION on a position-PD model, two ways:")
        print(f"    1. from-rest start: the plan's first DCM sits at y = {1e3*xi0y:+.0f} mm, past the")
        print(f"       ~20 mm double-support weight-shift envelope -- she has no lateral momentum")
        print(f"       to enter the limit cycle and toppling begins in step 0.")
        print(f"    2. CoP authority: placing the LIPM CoP needs ANKLE TORQUE control; the")
        print(f"       position servos (kp 30) move the CoP only ~1.5·d and lag the {t_step:.1f}s step.")
        print("  Next: torque-controlled ankles + a limit-cycle warm-start, or a ZMP-preview")
        print("  controller.  Quasi-static stepping (U11) stays Cara's locomotion.")
    print("\n  LIPM-planned DCM + capture-point step adjustment + ankle CoP feedback; no RL. "
          "(provisional masses / gains / friction / foot size)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--t-step", type=float, default=None)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args(argv)
    return run(args.config, args.steps, args.t_step, args.view, args.json, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
