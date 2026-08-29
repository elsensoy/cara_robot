#!/usr/bin/env python3
"""U12 -- a continuous walk.

U11 (`gait.py`) chained the U10 step but stopped dead between steps (~14 s of
ramp per step).  U12 removes the stops: a single **periodic gait cycle** is
precomputed once as a dense joint trajectory, then played back on a loop.  The
COM advances at a roughly constant speed and never comes to rest -- this is
walking with momentum, not start-stop stepping.

Still transparent, still no RL:

  * one L+R cycle is precomputed by IK (sagittal joints track each foot's
    world trajectory; the four roll joints track a lateral COM sway -- the same
    two decoupled IK problems as `weight_shift` and the U10 swing table);
  * the cycle is translation-periodic, so replaying it produces steady forward
    progress;
  * U9's lateral COM-y roll trim rides on top for balance.

Milestone question:

> **Can Cara walk continuously for N cycles** (no stop between steps), the COM
> advancing at a roughly steady speed, staying upright -- and then stop and
> stand?

`analysis.walk` in the config holds the cycle timing / stride / sway, all
provisional.  Failure cases are reported, not hidden.

Requires `mujoco` (brings numpy).  Prints SKIPPED / exits 0 without it.

Usage:
    python3 walk.py                       # full body, N cycles
    python3 walk.py --cycles 5
    python3 walk.py --t-step 2.0          # slower / faster cadence (override)
    python3 walk.py --view
    python3 walk.py --json baselines/full_body_walk.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import leg_model as lm
import weight_shift as wsh

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf"))
DEFAULT_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_full_body.yaml"))

SIDE = {"l_": +1.0, "r_": -1.0}
OTHER = {"l_": "r_", "r_": "l_"}


def run(config, cycles, t_step_override, view, json_path, baseline_path):
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
    if os.path.exists(on_disk):
        with open(on_disk, encoding="utf-8") as fh:
            if fh.read() != xml:
                print(f"WARNING: {on_disk} is stale -- fresh render used here\n")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    g = lm.analysis_gravity(spec)

    wc = (spec.get("analysis", {}) or {}).get("walk", {}) or {}
    if not wc:
        print("this config has no analysis.walk block")
        return 2
    base_pose = wc.get("base_pose", "stand_nominal")
    stride = float(wc.get("stride", 0.024))
    t_step = float(t_step_override if t_step_override is not None else wc.get("t_step", 2.0))
    ds_frac = float(wc.get("double_support_frac", 0.30))
    lift_h = float(wc.get("lift_height", 0.012))
    sway = float(wc.get("sway", 0.024))
    n_cycles = int(cycles if cycles is not None else wc.get("n_cycles", 4))
    lead_in = float(wc.get("lead_in_seconds", 2.5))
    settle = float(wc.get("settle_seconds", 1.5))
    tail = float(wc.get("tail_seconds", 3.0))
    bal = wc.get("balance", {}) or {}
    KPA = float(bal.get("kp_ankle_roll", 50.0))
    KDA = float(bal.get("kd_ankle_roll", 10.0))
    KPH = float(bal.get("kp_hip_roll", 15.0))
    acc = wc.get("accept", {}) or {}
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 10.0)))
    MIN_MARGIN = float(acc.get("min_support_margin", -0.010))
    MAX_TQ = float(acc.get("max_torque_frac", 1.0))
    SPEED_TOL = float(acc.get("speed_consistency", 0.35))   # std/mean of per-cycle speed
    MIN_SPEED_FRAC = float(acc.get("min_speed_frac", 0.5))  # vs commanded
    TAIL_DRIFT = float(acc.get("tail_hold_drift", 0.012))

    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)[base_pose]
    FULL0 = {n: float(base_cfg.get(n, 0.0)) for n in jn}
    nominal_ctrl = [FULL0[n] for n in jn]

    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose)
    foot_gid = {"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")}
    floor_gid = gid("floor")
    pelvis_bid = bid(spec["frame_conventions"]["base_frame"])
    forcerng = np.array([model.actuator_forcerange[aid(n)][1] for n in jn])
    m_total = float(sum(model.body_mass))
    total_weight = m_total * g

    tf0 = lm.forward_kinematics(spec, base_cfg)
    SOLE0 = {p: lm.frame_world_position(spec, tf0, p + "foot_sole_center") for p in ("l_", "r_")}
    ROT0 = {p: tf0[p + "foot"][0] for p in ("l_", "r_")}
    SAG = {p: [p + j for j in ("hip_pitch", "knee_pitch", "ankle_pitch")] for p in ("l_", "r_")}
    ROLLJ = ["l_hip_roll", "l_ankle_roll", "r_hip_roll", "r_ankle_roll"]
    Z0 = SOLE0["l_"][2]
    v_cmd = stride / t_step                         # commanded forward speed

    # weight_shift's inverted table: desired world-COM-y -> (pelvis shift, roll
    # joint targets that actually put the COM there when the feet are planted)
    shift_table = wsh.build_ik_table(spec, base_cfg, sway + 0.02, 61)

    def smooth01(u):
        u = min(1.0, max(0.0, u))
        return u * u * (3.0 - 2.0 * u)

    # ---- precompute ONE step (lead l_, r_ stance), phase p in [0,1) -------- #
    # World frame for the precompute: r_ (stance) sole at x = 0, l_ (swing) sole
    # starts at x = -stride and lands at x = +stride (travels 2*stride).  The
    # pelvis advances `stride` over the step.  At p = 1 the config equals the
    # L<->R swap of its p = 0 config, so playing [step_L, swap(step_L), ...]
    # gives a seamless periodic walk.
    SWAP = {n: (("r_" + n[2:]) if n.startswith("l_") else
                ("l_" + n[2:]) if n.startswith("r_") else n) for n in jn}
    SWAP_IDX = [jn.index(SWAP[n]) for n in jn]

    # step timeline: opening double support [0, ds] -> single-support swing
    # [ds, 1-ds] -> closing double support [1-ds, 1].  The swing foot lands at
    # p = 1-ds so there is a real transfer window before the next step.
    def swing_progress(p):
        return max(0.0, min(1.0, (p - ds_frac) / (1.0 - 2.0 * ds_frac)))

    def swing_x_world(p):
        return -stride + 2.0 * stride * smooth01(swing_progress(p))

    def swing_z(p):
        sp = swing_progress(p)
        return 0.0 if sp <= 0.0 or sp >= 1.0 else lift_h * math.sin(math.pi * sp)

    def com_y_ref(p):
        """sway toward the r_ stance foot (-y).  A half-sine over the step: 0 at
        both ends (the L<->R hand-off points) and -sway at mid-step.  Its slope
        matches the next (mirrored) step's slope at the boundary, so the COM-y
        velocity is continuous across steps -- a trapezoid here put a velocity
        kink at every hand-off and she toppled."""
        return -sway * math.sin(math.pi * p)

    K = int(wc.get("cycle_samples", 121))
    step = []
    step_ref_ey = []          # intended (COM_y - r_stance_foot_y) at each sample, pelvis frame
    q_sag = dict(FULL0)
    for i in range(K):
        p = i / (K - 1)
        padv = stride * p                           # pelvis advance within the step
        cfg = dict(FULL0)
        # sagittal joints -- each foot's sole at its pelvis-frame (x, z)
        targets = {"r_": (0.0 - padv, Z0),
                   "l_": (swing_x_world(p) - padv, Z0 + swing_z(p))}
        for pfx in ("l_", "r_"):
            fx_pf, fz = targets[pfx]
            leg, _r = lm.leg_ik(spec, pfx, pfx + "foot_sole_center",
                                (fx_pf, SOLE0[pfx][1], fz), ROT0[pfx], {**base_cfg, **q_sag},
                                free_joints=SAG[pfx], task_rows=[0, 2, 4], iters=200)
            q_sag.update({k: leg[k] for k in SAG[pfx]})
            cfg.update({k: leg[k] for k in SAG[pfx]})
        # roll joints -- lateral COM sway, from weight_shift's inverted table so
        # the COM actually reaches com_y_ref(p) with the feet planted
        _, qt_roll = wsh.table_lookup(shift_table, com_y_ref(p))
        for j in ROLLJ:
            cfg[j] = qt_roll[jn.index(j)]
        step.append([cfg[n] for n in jn])
        # intended COM-y vs the r_ stance foot, in the pelvis frame -- the roll
        # trim corrects deviations from THIS, not from zero (else it fights the
        # feedforward sway)
        tf_c = lm.forward_kinematics(spec, cfg)
        com_pf = lm.center_of_mass(spec, cfg)[1]
        rfoot_pf = lm.frame_world_position(spec, tf_c, "r_foot_sole_center")
        step_ref_ey.append(float(com_pf[1] - rfoot_pf[1]))
    step = np.array(step)                           # (K, njoints), the lead-l_ step
    step_ref_ey = np.array(step_ref_ey)

    def step_at(phase, lead):
        u = (phase % 1.0) * (K - 1)
        a = int(u)
        b = min(K - 1, a + 1)
        f = u - a
        row = step[a] * (1 - f) + step[b] * f
        return row if lead == "l_" else row[SWAP_IDX]

    def ref_ey_at(phase, lead):
        u = (phase % 1.0) * (K - 1)
        a = int(u)
        b = min(K - 1, a + 1)
        f = u - a
        e = step_ref_ey[a] * (1 - f) + step_ref_ey[b] * f
        return float(e) if lead == "l_" else -float(e)   # mirror flips the sway side

    # ---- sim helpers ---------------------------------------------------- #
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

    def support_polygon():
        pts = []
        for fg in foot_gid.values():
            for i in range(data.ncon):
                c = data.contact[i]
                if {c.geom1, c.geom2} == {fg, floor_gid}:
                    pts.append((float(c.pos[0]), float(c.pos[1])))
        return lm.convex_hull_2d(pts)

    ROLL_IDX = {j: jn.index(j) for j in ROLLJ}

    n_steps = 2 * n_cycles

    def walk_sim(record=True):
        mujoco.mj_resetDataKeyframe(model, data, kid)
        for _ in range(int(settle / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)
        # lead-in: ramp stand_nominal -> the lead-l_ step at phase 0
        c0 = step_at(0.0, "l_")
        for k in range(int(lead_in / dt)):
            f = smooth01(k * dt / lead_in)
            data.ctrl[:] = [(1 - f) * nominal_ctrl[i] + f * c0[i] for i in range(len(jn))]
            mujoco.mj_step(model, data)

        com_x0 = float(data.subtree_com[0][0])
        prev_drift = 0.0
        n = int(n_steps * t_step / dt)
        log = {"tilt": 0.0, "margin": 1e9, "tq": 0.0, "sway": 0.0,
               "fell": False, "cycle_x": [com_x0], "cycle_t": [0.0]}
        for k in range(n):
            t = k * dt
            step_i = int(t / t_step)
            phase = (t - step_i * t_step) / t_step
            lead = "l_" if step_i % 2 == 0 else "r_"
            stance = OTHER[lead]
            ref = step_at(phase, lead)
            sfn = -SIDE[stance]
            com = data.subtree_com[0]
            sf = data.geom_xpos[foot_gid[stance]]
            drift = float(com[1] - sf[1]) - ref_ey_at(phase, lead)
            ddrift = (drift - prev_drift) / dt          # deviation velocity, not raw COM velocity
            prev_drift = drift
            # the roll trim is a single-support balance aid (like U9); during the
            # opening double support the feed-forward sway is quasi-statically
            # stable (it is just weight_shift), so fade the trim in over the swing
            ph = phase % 1.0
            ss = (max(0.0, min(1.0, (ph - ds_frac) / 0.12))
                  * max(0.0, min(1.0, (1.0 - ds_frac - ph) / 0.12)))
            ctrl = list(ref)
            ctrl[ROLL_IDX[stance + "ankle_roll"]] += ss * sfn * (KPA * drift + KDA * ddrift)
            ctrl[ROLL_IDX[stance + "hip_roll"]] += ss * sfn * KPH * drift
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
            if not record:
                if max(abs(roll), abs(pitch)) > math.radians(45):
                    return None
                continue
            tau = np.array([data.actuator_force[aid(nm)] for nm in jn])
            log["tilt"] = max(log["tilt"], abs(roll), abs(pitch))
            if max(abs(roll), abs(pitch)) < math.radians(8):    # exclude the topple
                log["margin"] = min(log["margin"], lm.polygon_signed_margin(
                    support_polygon(), (float(com[0]), float(com[1]))))
                log["tq"] = max(log["tq"], float(np.max(np.abs(tau) / forcerng)))
                log["sway"] = max(log["sway"], abs(float(com[1])))
            if max(abs(roll), abs(pitch)) > math.radians(45):
                log["fell"] = True
                log["fell_t"] = t
                break
            if abs(t - (step_i + 1) * t_step) < dt:   # step boundary
                log["cycle_x"].append(float(com[0]))
                log["cycle_t"].append(t)
        log["cycle_x"].append(float(data.subtree_com[0][0]))
        log["cycle_t"].append((k + 1) * dt)

        # tail: stop and stand.  which lead is "next" determines the neutral pose
        next_lead = "l_" if (int((k * dt) / t_step) + 1) % 2 == 0 else "r_"
        stop = step_at(0.0, next_lead)
        tail_tilt = tail_drift = 0.0
        cf0 = np.array(data.subtree_com[0][:2])
        tcorn = 99
        for _ in range(int(tail / dt)):
            data.ctrl[:] = stop
            mujoco.mj_step(model, data)
            r2, p2, _ = wsh.quat_rpy(data.qpos[3:7])
            tail_tilt = max(tail_tilt, abs(r2), abs(p2))
            tail_drift = max(tail_drift, float(np.linalg.norm(
                np.array(data.subtree_com[0][:2]) - cf0)))
            tcorn = min(tcorn, foot_corners(foot_gid["l_"]), foot_corners(foot_gid["r_"]))
        log["tail_tilt"] = tail_tilt
        log["tail_drift"] = tail_drift
        log["tail_corners"] = tcorn
        log["total_advance"] = float(data.subtree_com[0][0]) - com_x0
        return log

    # ================================================================== #
    if view:
        return _view(model, data, dt, step_at, ref_ey_at, nominal_ctrl, settle, lead_in, ds_frac,
                     t_step, n_steps, ROLL_IDX, jn, foot_gid, SIDE, OTHER,
                     KPA, KDA, KPH, mujoco, smooth01, wsh, math)

    print(f"Continuous walk (U12)  base '{base_pose}'  {model_name}")
    print(f"{m_total:.2f} kg  |  {n_steps} steps ({t_step:.2f} s each), stride {1e3*stride:.0f} mm "
          f"-> {1e3*v_cmd:.0f} mm/s commanded  |  sway {1e3*sway:.0f} mm, "
          f"lift {1e3*lift_h:.0f} mm, double-support {100*ds_frac:.0f}%")

    log = walk_sim()
    if log["fell"]:
        print(f"\n  FELL at t = {log.get('fell_t', 0):.1f} s "
              f"(step {log['fell_t']/t_step:.1f} of {n_steps})")

    cx = np.array(log["cycle_x"])
    ct = np.array(log["cycle_t"])
    seg_v = np.diff(cx) / np.clip(np.diff(ct), 1e-6, None)
    seg_v = seg_v[np.isfinite(seg_v)]
    mean_v = float(np.mean(seg_v)) if len(seg_v) else 0.0
    cons = float(np.std(seg_v) / abs(mean_v)) if abs(mean_v) > 1e-6 else 9.9

    print(f"\n  advanced {1e3*log['total_advance']:.0f} mm total")
    print(f"  forward speed: {1e3*mean_v:.0f} mm/s mean (commanded {1e3*v_cmd:.0f}), "
          f"per-step {', '.join(f'{1e3*s:.0f}' for s in seg_v)} mm/s")
    print(f"  consistency std/mean = {cons:.2f}  (lower = steadier; start-stop stepping would be ~1)")
    marg_s = f"{1e3*log['margin']:.1f} mm" if log["margin"] > -0.1 else "COM left the polygon"
    sway_s = f"+-{1e3*log['sway']:.0f} mm" + (" (pre-topple)" if log["fell"] else "")
    print(f"  peak pelvis tilt {math.degrees(log['tilt']):.1f}°, min COM margin {marg_s}, "
          f"lateral sway {sway_s}, peak torque {100*log['tq']:.0f}%")
    print(f"  stop + stand {tail:.0f}s: tilt {math.degrees(log['tail_tilt']):.1f}°, "
          f"drift {1e3*log['tail_drift']:.1f} mm, {log['tail_corners']} corners/foot")

    walked = (not log["fell"]) and mean_v > MIN_SPEED_FRAC * v_cmd
    steady = cons < SPEED_TOL
    upright = log["tilt"] < MAX_TILT and log["margin"] > MIN_MARGIN and log["tq"] <= MAX_TQ
    stopped = (log["tail_tilt"] < MAX_TILT and log["tail_drift"] < TAIL_DRIFT
               and log["tail_corners"] >= 3)
    ok = walked and steady and upright and stopped

    results = {"model": model_name, "n_cycles": n_cycles, "stride_m": stride, "t_step": t_step,
               "v_cmd_mps": v_cmd, "v_mean_mps": mean_v, "consistency": cons,
               "total_advance_m": log["total_advance"], "peak_tilt_deg": math.degrees(log["tilt"]),
               "min_margin_m": log["margin"], "sway_m": log["sway"], "peak_torque_frac": log["tq"],
               "fell": log["fell"], "fell_t": log.get("fell_t"),
               "failure_mode": ("toppled_at_transfer" if log["fell"] and (log.get("fell_t", 0) % t_step) > 0.6 * t_step
                                else "toppled_in_swing" if log["fell"]
                                else "stepped_in_place" if mean_v < MIN_SPEED_FRAC * v_cmd
                                else None),
               "milestone_met": bool(ok)}
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsummary -> {json_path}")
    if baseline_path and os.path.exists(baseline_path):
        base = json.load(open(baseline_path, encoding="utf-8"))
        print(f"\nvs baseline: milestone {base.get('milestone_met')} -> {ok}, "
              f"speed {1e3*base.get('v_mean_mps',0):.0f} -> {1e3*mean_v:.0f} mm/s")

    # characterise the failure mode
    stepped_in_place = (not log["fell"]) and mean_v < MIN_SPEED_FRAC * v_cmd
    toppled_transfer = log["fell"] and (log.get("fell_t", 0) % t_step) > 0.6 * t_step

    print("\n" + "=" * 74)
    if ok:
        print(f"MILESTONE MET: Cara walks continuously for {n_steps} steps "
              f"({1e3*log['total_advance']:.0f} mm) at ~{1e3*mean_v:.0f} mm/s "
              f"with no stop between steps, then stops and stands.")
    else:
        print("MILESTONE NOT MET -- a continuous walk is past Cara's current limit:")
        if log["fell"]:
            where = "the double-support transfer to the next stance foot" if toppled_transfer \
                    else "the single-support swing"
            print(f"  * FELL at t = {log.get('fell_t',0):.1f} s ({log['fell_t']/t_step:.1f} steps in), "
                  f"during {where}.")
            print(f"    the lateral COM momentum from the sway (+-{1e3*log['sway']:.0f} mm) cannot be "
                  f"arrested by the 22.5 mm foot -- the U9 balance_margin wall (~1 N*m ankle roll,")
            print(f"    ~6.5 mm single-support margin).  U9's static-hold roll gains (kp 50) destabilise "
                  f"a *moving* reference, so this uses a gentler gated trim -- still not enough.")
        elif stepped_in_place:
            print(f"  * she stays upright (tilt {math.degrees(log['tilt']):.1f}°) but advances only "
                  f"~{1e3*mean_v*t_step:.0f} mm/step of the {1e3*stride:.0f} mm stride -- kinematic")
            print(f"    pose-playback does not induce forward translation without push-off / momentum;")
            print(f"    the stance foot slides instead of the pelvis advancing.")
        print("  A continuous walk needs a dynamic gait controller (ZMP / capture-point pattern with")
        print("  push-off) and/or a wider foot.  Quasi-static stepping (U11) stays Cara's locomotion.")
    print("\n  precomputed periodic cycle + a gated roll trim; no ZMP / capture-point, no RL. "
          "(provisional masses / gains / friction / foot size)")
    return 0 if ok else 1


def _view(model, data, dt, step_at, ref_ey_at, nominal_ctrl, settle, lead_in, ds_frac, t_step,
          n_steps, ROLL_IDX, jn, foot_gid, SIDE, OTHER, KPA, KDA, KPH,
          mujoco, smooth01, wsh, math):
    import time
    try:
        import mujoco.viewer
    except ImportError:
        print("error: mujoco.viewer unavailable (needs a display)", file=sys.stderr)
        return 2
    print("\nviewer: Cara walks continuously on a loop. close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_resetDataKeyframe(model, data,
                                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand_nominal"))
            for _ in range(int(settle / dt)):
                data.ctrl[:] = nominal_ctrl
                mujoco.mj_step(model, data)
            c0 = step_at(0.0, "l_")
            for k in range(int(lead_in / dt)):
                f = smooth01(k * dt / lead_in)
                data.ctrl[:] = [(1 - f) * nominal_ctrl[i] + f * c0[i] for i in range(len(jn))]
                mujoco.mj_step(model, data)
                if k % 8 == 0:
                    v.sync()
            prev_drift = 0.0
            for k in range(int(n_steps * t_step / dt)):
                t = k * dt
                step_i = int(t / t_step)
                phase = (t - step_i * t_step) / t_step
                lead = "l_" if step_i % 2 == 0 else "r_"
                stance = OTHER[lead]
                ref = list(step_at(phase, lead))
                sfn = -SIDE[stance]
                com = data.subtree_com[0]
                sf = data.geom_xpos[foot_gid[stance]]
                drift = float(com[1] - sf[1]) - ref_ey_at(phase, lead)
                ddrift = (drift - prev_drift) / dt
                prev_drift = drift
                ph = phase % 1.0
                ss = (max(0.0, min(1.0, (ph - ds_frac) / 0.12))
                      * max(0.0, min(1.0, (1.0 - ds_frac - ph) / 0.12)))
                ref[ROLL_IDX[stance + "ankle_roll"]] += ss * sfn * (KPA * drift + KDA * ddrift)
                ref[ROLL_IDX[stance + "hip_roll"]] += ss * sfn * KPH * drift
                data.ctrl[:] = ref
                mujoco.mj_step(model, data)
                if k % 8 == 0:
                    v.sync()
                    time.sleep(dt * 8)
                if not v.is_running():
                    return 0
                r2, p2, _ = wsh.quat_rpy(data.qpos[3:7])
                if max(abs(r2), abs(p2)) > math.radians(50):
                    break
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--cycles", type=int, default=None, help="L+R cycles to walk")
    ap.add_argument("--t-step", type=float, default=None, help="seconds per step (override)")
    ap.add_argument("--view", action="store_true", help="watch the walk in the viewer")
    ap.add_argument("--json", default=None, help="write the run summary here")
    ap.add_argument("--baseline", default=None, help="compare against this summary JSON")
    args = ap.parse_args(argv)
    return run(args.config, args.cycles, args.t_step, args.view, args.json, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
