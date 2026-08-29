#!/usr/bin/env python3
"""U11 -- chain the U10 step into a short walk.

U10 (`step_once.py`) took ONE forward step from rest and stopped.  U11 repeats
it, alternating legs, so Cara actually walks a few steps forward.

Nothing new in the controller: each step is the same six quasi-static phases as
U10 (A shift / B lift / C swing / D place / E transfer / F settle) plus U9's
lateral COM-y roll trim while she is on one foot.  The only new machinery is
making a step start from the *staggered* stance the previous step left her in
(instead of always from `stand_nominal`) and alternating which foot leads.

  step 1:  lead l_ , from rest        -> staggered (l_ forward)
  step 2:  lead r_ , from staggered   -> staggered (r_ forward)   ... periodic
  ...

Milestone question:

> **Can Cara take N consecutive quasi-static steps** (alternating legs),
> advancing steadily, and end standing -- COM inside the support polygon every
> step, pelvis near level, feet not slipping, no actuator saturated?

No gait optimisation, no dynamic walking, no RL.  Failure cases are reported,
not hidden.

Requires `mujoco` (brings numpy).  Prints SKIPPED / exits 0 without it.

Usage:
    python3 gait.py                       # full body, N steps forward
    python3 gait.py --steps 6
    python3 gait.py --view
    python3 gait.py --json baselines/full_body_gait.json
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
C_TOP = 0.035


def run(config, n_steps, view, json_path, baseline_path):
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

    gc = (spec.get("analysis", {}) or {}).get("gait", {}) or {}
    if not gc:
        print("this config has no analysis.gait block")
        return 2
    base_pose = gc.get("base_pose", "stand_nominal")
    com_target = float(gc.get("com_target", 0.028))
    lift_h = float(gc.get("lift_height", 0.010))
    stride = float(gc.get("stride", 0.024))
    n_steps = int(n_steps if n_steps is not None else gc.get("n_steps", 4))
    ramp = float(gc.get("ramp_seconds", 4.0))
    swing_s = float(gc.get("swing_seconds", 4.0))
    settle = float(gc.get("settle_seconds", 1.5))
    step_settle = float(gc.get("step_settle_seconds", 1.0))
    final_hold = float(gc.get("final_hold_seconds", 3.0))
    CG = float(gc.get("clearance_gain", 0.012))
    bal = gc.get("balance", {}) or {}
    KPA = float(bal.get("kp_ankle_roll", 50.0))
    KDA = float(bal.get("kd_ankle_roll", 10.0))
    KPH = float(bal.get("kp_hip_roll", 15.0))
    acc = gc.get("accept", {}) or {}
    PLACE_TOL = float(acc.get("place_tol", 0.012))
    MIN_MARGIN = float(acc.get("min_support_margin", 0.004))
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 8.0)))
    MAX_SLIP = float(acc.get("max_step_slip", 0.010))
    MAX_TQ = float(acc.get("max_torque_frac", 1.0))
    MIN_CORNERS = int(acc.get("min_stance_corners", 3))
    MIN_ADVANCE_FRAC = float(acc.get("min_advance_frac", 0.6))
    FINAL_DRIFT = float(acc.get("final_hold_drift", 0.008))

    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)[base_pose]
    nominal_ctrl = [float(base_cfg.get(n, 0.0)) for n in jn]

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

    # roll-joint trajectory for a lateral COM shift -- decoupled from the sagittal
    # stagger, so the SAME roll deltas work from any staggered stance
    shift_table = wsh.build_ik_table(spec, base_cfg, com_target + 0.03, 61)
    tf0 = lm.forward_kinematics(spec, base_cfg)
    SOLE0 = {p: lm.frame_world_position(spec, tf0, p + "foot_sole_center") for p in ("l_", "r_")}
    ROT0 = {p: tf0[p + "foot"][0] for p in ("l_", "r_")}
    SAG = {p: [p + j for j in ("hip_pitch", "knee_pitch", "ankle_pitch")] for p in ("l_", "r_")}
    ROLLJ = ["l_hip_roll", "l_ankle_roll", "r_hip_roll", "r_ankle_roll"]
    Z_GROUND = SOLE0["l_"][2]

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

    def foot_world_x(pfx):
        return float(data.geom_xpos[foot_gid[pfx]][0])

    def pelvis_x():
        return float(data.qpos[0])

    # --- solve a leg's sagittal joints to place its sole at a pelvis-frame pt -- #
    def solve_leg(pfx, target_pf, seed):
        sol, r = lm.leg_ik(spec, pfx, pfx + "foot_sole_center", target_pf, ROT0[pfx], seed,
                           free_joints=SAG[pfx], task_rows=[0, 2, 4], iters=400)
        return {k: sol[k] for k in SAG[pfx]}, r

    # --- swing-leg table for one step (built per step, in the current frame) -- #
    def build_swing_table(lead, roll_cfg, x0_pf, xt_pf, y_pf, ns=13, nc=8):
        free = SAG[lead]
        q0 = {**base_cfg, **roll_cfg}
        tf = lm.forward_kinematics(spec, q0)
        z0 = lm.frame_world_position(spec, tf, lead + "foot_sole_center")[2]
        rot = tf[lead + "foot"][0]
        grid = []
        for i in range(ns):
            s = i / (ns - 1)
            row, q = [], dict(q0)
            for k in range(nc):
                c = C_TOP * k / (nc - 1)
                sol, _r = lm.leg_ik(spec, lead, lead + "foot_sole_center",
                                    (x0_pf + s * (xt_pf - x0_pf), y_pf, z0 + c), rot, q,
                                    free_joints=free, task_rows=[0, 2, 4], iters=250)
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
                    return [rl[b][1][k] * (1 - fb) + rl[b + 1][1][k] * fb
                            for k in range(len(rl[b][1]))]
            return rl[-1][1]

        va, vb = at(grid[a][1]), at(grid[a + 1][1])
        return [va[k] * (1 - fa) + vb[k] * fa for k in range(len(va))]

    FULL0 = {n: float(base_cfg.get(n, 0.0)) for n in jn}   # every actuated joint, 0-filled

    # --- one step ----------------------------------------------------------- #
    state = {"cfg": dict(FULL0)}

    def take_step(lead, foothold_world_x, dbg=None):
        stance = OTHER[lead]
        a_roll, h_roll = stance + "ankle_roll", stance + "hip_roll"
        cur = dict(state["cfg"])

        # phase-A target: current sagittal cfg + the shift table's roll deltas
        _, qt = wsh.table_lookup(shift_table, SIDE[stance] * com_target)
        shift_roll = {j: qt[jn.index(j)] for j in ROLLJ}
        roll_cfg = {**cur, **shift_roll}
        roll_ctrl = [float(roll_cfg.get(n, 0.0)) for n in jn]
        cur_ctrl = [float(cur.get(n, 0.0)) for n in jn]

        # record the run-in state
        foot0 = {p: np.array(data.geom_xpos[foot_gid[p]][:2]) for p in ("l_", "r_")}
        swing_z0 = float(data.geom_xpos[foot_gid[lead]][2])
        com_x0 = float(data.subtree_com[0][0])

        lead_x_start = foot_world_x(lead)
        stance_x_start = foot_world_x(stance)
        px_start = pelvis_x()

        M = {"margin": 1e9, "tilt": 0.0, "slip": 0.0, "sat": 0, "tq": 0.0,
             "stance_corners": 99, "swing_clear": 1e9, "swing_lift": 0.0}
        cmd = dict(roll_cfg)
        st = {"ccmd": 0.0, "ref_ey": None, "pcy": float(data.subtree_com[0][1])}
        grid = [None]
        free = [None]

        def ss_step(clear_target, s_prog, record=True):
            wc = float(data.geom_xpos[foot_gid[lead]][2]) - swing_z0
            M["swing_lift"] = max(M["swing_lift"], wc)
            st["ccmd"] = min(C_TOP, max(0.0, st["ccmd"] + CG * (clear_target - wc)))
            if grid[0] is not None:
                for n, v in zip(free[0], swing_lookup(grid[0], s_prog, st["ccmd"])):
                    cmd[n] = v
            com = data.subtree_com[0]
            sf = data.geom_xpos[foot_gid[stance]]
            ey = float(com[1] - sf[1])
            if st["ref_ey"] is None:
                st["ref_ey"] = ey
            dy = ey - st["ref_ey"]
            vy = (float(com[1]) - st["pcy"]) / dt
            st["pcy"] = float(com[1])
            sfn = -SIDE[stance]
            c2 = dict(cmd)
            c2[a_roll] = roll_cfg[a_roll] + sfn * (KPA * dy + KDA * vy)
            c2[h_roll] = roll_cfg[h_roll] + sfn * KPH * dy
            data.ctrl[:] = [c2[n] for n in jn]
            mujoco.mj_step(model, data)
            if not record:
                return
            roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
            tau = np.array([data.actuator_force[aid(n)] for n in jn])
            M["margin"] = min(M["margin"], lm.polygon_signed_margin(
                support_polygon(), (float(com[0]), float(com[1]))))
            M["tilt"] = max(M["tilt"], abs(roll), abs(pitch))
            M["slip"] = max(M["slip"], float(np.linalg.norm(
                data.geom_xpos[foot_gid[stance]][:2] - foot0[stance])))
            M["sat"] += int(np.any(np.abs(tau) >= forcerng - 0.02))
            M["tq"] = max(M["tq"], float(np.max(np.abs(tau) / forcerng)))
            M["stance_corners"] = min(M["stance_corners"], foot_corners(foot_gid[stance]))
            M["swing_clear"] = min(M["swing_clear"], wc)

        # A: lateral COM shift onto the stance foot
        for k in range(int(ramp / dt)):
            f = wsh.smoothstep(k * dt / ramp)
            data.ctrl[:] = [(1 - f) * cur_ctrl[i] + f * roll_ctrl[i] for i in range(len(jn))]
            mujoco.mj_step(model, data)
        st["pcy"] = float(data.subtree_com[0][1])
        if dbg:
            dbg("A")

        # build the swing table now that the pelvis has settled over the stance foot
        px = pelvis_x()
        tfc = lm.forward_kinematics(spec, {**base_cfg, **roll_cfg})
        lead_pf = lm.frame_world_position(spec, tfc, lead + "foot_sole_center")
        x0_pf = foot_world_x(lead) - px
        grid[0], free[0] = build_swing_table(
            lead, roll_cfg, x0_pf, foothold_world_x - px, lead_pf[1])

        # B: lift straight up
        for k in range(int(ramp / dt)):
            ss_step(lift_h * wsh.smoothstep(k * dt / ramp), 0.0, record=False)
        if dbg:
            dbg("B")
        # C: swing forward to the foothold
        for k in range(int(swing_s / dt)):
            ss_step(lift_h, wsh.smoothstep(k * dt / swing_s))
        if dbg:
            dbg("C")
        # D: place
        for k in range(int(ramp / dt)):
            ss_step(lift_h * (1.0 - wsh.smoothstep(k * dt / ramp)), 1.0)
        for _ in range(int(0.4 / dt)):
            ss_step(0.0, 1.0, record=False)
        place_xy = np.array(data.geom_xpos[foot_gid[lead]][:2])
        if dbg:
            dbg("D")

        # E: transfer to the new staggered stance -- pelvis centred between the
        # two feet at their world x, both feet level, roll joints back to 0
        final_cfg = {k: (0.0 if k in ROLLJ else v) for k, v in cur.items()}
        new_px = 0.5 * (foothold_world_x + foot_world_x(stance))
        for pfx in (lead, stance):
            wx = foothold_world_x if pfx == lead else foot_world_x(stance)
            leg, _r = solve_leg(pfx, (wx - new_px, SOLE0[pfx][1], Z_GROUND), {**base_cfg, **cur})
            final_cfg.update(leg)
        final_ctrl = [float(final_cfg.get(n, 0.0)) for n in jn]
        start_ctrl = np.array(data.ctrl[:])
        for k in range(int(ramp / dt)):
            f = wsh.smoothstep(k * dt / ramp)
            data.ctrl[:] = (1 - f) * start_ctrl + f * np.array(final_ctrl)
            mujoco.mj_step(model, data)
        # F: brief settle
        for _ in range(int(step_settle / dt)):
            data.ctrl[:] = final_ctrl
            mujoco.mj_step(model, data)
        if dbg:
            dbg("E")

        state["cfg"] = final_cfg
        com_xf = float(data.subtree_com[0][0])
        roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
        place_err = float(np.linalg.norm(
            place_xy - np.array([foothold_world_x, foot0[lead][1]])))
        return {
            "lead": lead, "advance": com_xf - com_x0,
            "place_err": place_err,
            "ss_margin": M["margin"], "ss_tilt": M["tilt"], "ss_slip": M["slip"],
            "ss_sat": M["sat"], "ss_tq": M["tq"], "ss_stance_corners": M["stance_corners"],
            "ss_swing_clear": M["swing_clear"],
            "end_tilt": max(abs(roll), abs(pitch)),
            "end_corners": min(foot_corners(foot_gid["l_"]), foot_corners(foot_gid["r_"])),
            "lead_x0": lead_x_start, "lead_x1": foot_world_x(lead),
            "stance_x": stance_x_start, "px0": px_start, "px1": pelvis_x(),
            "swing_lift": M["swing_lift"],
        }

    def classify(m):
        bits = []
        if m["place_err"] > PLACE_TOL:                   bits.append(f"place({1e3*m['place_err']:.0f}mm)")
        if m["ss_margin"] < MIN_MARGIN:                  bits.append("COM-outside")
        if m["ss_tilt"] >= MAX_TILT:                     bits.append("tilt")
        if m["ss_slip"] >= MAX_SLIP:                     bits.append(f"slip({1e3*m['ss_slip']:.0f}mm)")
        if m["ss_sat"] != 0 or m["ss_tq"] > MAX_TQ:      bits.append("torque")
        if m["ss_stance_corners"] < MIN_CORNERS:         bits.append("stance-lifting")
        if m["end_corners"] < MIN_CORNERS:               bits.append("foot-lift")
        if m["end_tilt"] >= MAX_TILT:                    bits.append("end-tilt")
        if m["advance"] < MIN_ADVANCE_FRAC * stride:
            bits.append(f"no-advance({1e3*m['advance']:.0f}/{1e3*stride:.0f}mm)")
        return (not bits), bits

    # ================================================================== #
    def walk(dbg=None):
        mujoco.mj_resetDataKeyframe(model, data, kid)
        for _ in range(int(settle / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)
        state["cfg"] = dict(FULL0)
        com_start = float(data.subtree_com[0][0])
        rows = []
        for i in range(n_steps):
            lead = "l_" if i % 2 == 0 else "r_"
            foothold = foot_world_x(OTHER[lead]) + stride
            m = take_step(lead, foothold, dbg=dbg)
            ok, bits = classify(m)
            rows.append({"i": i + 1, **m, "ok": bool(ok), "fail": bits})
        # final hold
        fc = [float(state["cfg"].get(n, 0.0)) for n in jn]
        com_f0 = np.array(data.subtree_com[0][:2])
        hold = {"tilt": 0.0, "drift": 0.0, "corners": 99}
        for _ in range(int(final_hold / dt)):
            data.ctrl[:] = fc
            mujoco.mj_step(model, data)
            com = data.subtree_com[0]
            roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
            hold["tilt"] = max(hold["tilt"], abs(roll), abs(pitch))
            hold["drift"] = max(hold["drift"], float(np.linalg.norm(np.array(com[:2]) - com_f0)))
            hold["corners"] = min(hold["corners"], foot_corners(foot_gid["l_"]),
                                  foot_corners(foot_gid["r_"]))
        hold["total_advance"] = float(data.subtree_com[0][0]) - com_start
        return rows, hold

    if view:
        return _view(model, data, dt, walk, mujoco)

    print(f"Short walk (U11)  base '{base_pose}'  {model_name}")
    print(f"{m_total:.2f} kg ({total_weight:.1f} N)  |  {n_steps} steps, stride {1e3*stride:.0f} mm, "
          f"COM shift {com_target:.3f} m, lift {1e3*lift_h:.0f} mm  |  U9 lateral roll trim "
          f"kp {KPA:.0f}/{KPH:.0f} kd {KDA:.0f}")

    rows, hold = walk()
    hdr = (f"  {'step':>4} {'lead':>5} {'advance':>8} {'place':>7} {'ss margin':>9} {'ss tilt':>8} "
           f"{'ss slip':>8} {'τ%':>4} {'end tilt':>8}  verdict")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['i']:>4} {r['lead']:>5} {1e3*r['advance']:>6.1f}mm {1e3*r['place_err']:>5.1f}mm "
              f"{1e3*r['ss_margin']:>7.1f}mm {math.degrees(r['ss_tilt']):>6.1f}° "
              f"{1e3*r['ss_slip']:>6.1f}mm {100*r['ss_tq']:>3.0f} {math.degrees(r['end_tilt']):>6.1f}°  "
              f"{'PASS' if r['ok'] else 'FAIL: ' + ','.join(r['fail'])}")

    # footwork: where each foot actually is, in world x, step by step
    print(f"\n  footwork (world x, mm -- shows each foot swinging past the other):")
    print(f"  {'step':>4} {'swing foot':>10} {'from x':>8} {'-> to x':>8} {'travel':>8} "
          f"{'peak lift':>10} {'stance foot x':>14} {'pelvis x':>16}")
    for r in rows:
        print(f"  {r['i']:>4} {r['lead'][:-1]:>10} {1e3*r['lead_x0']:>7.0f} {1e3*r['lead_x1']:>8.0f} "
              f"{1e3*(r['lead_x1']-r['lead_x0']):>7.0f} {1e3*r['swing_lift']:>8.1f}mm "
              f"{1e3*r['stance_x']:>13.0f} {1e3*r['px0']:>7.0f} -> {1e3*r['px1']:>4.0f}")

    steps_ok = all(r["ok"] for r in rows)
    end_ok = (hold["tilt"] < MAX_TILT and hold["drift"] < FINAL_DRIFT
              and hold["corners"] >= MIN_CORNERS)
    print(f"\n  final hold {final_hold:.0f}s: tilt {math.degrees(hold['tilt']):.1f}°, "
          f"COM drift {1e3*hold['drift']:.1f} mm, {hold['corners']} corners/foot  "
          f"[{'OK' if end_ok else 'FAIL'}]")
    adv = [1e3 * r["advance"] for r in rows]
    print(f"  total COM advance over {n_steps} steps: {1e3*hold['total_advance']:.0f} mm "
          f"(steps: {', '.join(f'{a:.0f}' for a in adv)} mm; commanded stride {1e3*stride:.0f} mm/step)")
    if len(adv) >= 3 and max(adv[1:]) - min(adv[1:]) < 3.0:
        print("  (steps 2+ are within 3 mm of each other -- the gait has settled to a periodic cycle)")

    both = steps_ok and end_ok
    results = {"model": model_name, "n_steps": n_steps, "stride_m": stride,
               "total_advance_m": hold["total_advance"], "milestone_met": bool(both),
               "steps": [{k: (v if isinstance(v, (bool, str, int)) else float(v)) if not isinstance(v, list) else v
                          for k, v in r.items()} for r in rows],
               "final_hold": {k: float(v) for k, v in hold.items()}}
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsummary -> {json_path}")
    if baseline_path and os.path.exists(baseline_path):
        base = json.load(open(baseline_path, encoding="utf-8"))
        print(f"\nvs baseline '{base.get('model','?')}': milestone "
              f"{base.get('milestone_met')} -> {both}, advance "
              f"{1e3*base.get('total_advance_m',0):.0f} -> {1e3*hold['total_advance']:.0f} mm")

    print("\n" + "=" * 74)
    if both:
        print(f"MILESTONE MET: Cara walks {n_steps} quasi-static steps forward "
              f"({1e3*hold['total_advance']:.0f} mm), alternating legs, and ends standing.")
    else:
        print("MILESTONE NOT MET: " + ("a step fails; " if not steps_ok else "")
              + ("the final stance is not stable; " if not end_ok else "") + "see above.")
    print("  U10's step chained + alternated; U9's roll trim on the position PD. no gait "
          "optimisation, no dynamic walking, no RL. (provisional masses / gains / friction / foot size)")
    return 0 if both else 1


def _view(model, data, dt, walk, mujoco):
    import time
    try:
        import mujoco.viewer
    except ImportError:
        print("error: mujoco.viewer unavailable (needs a display)", file=sys.stderr)
        return 2
    print("\nviewer: Cara walks forward on a loop. close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            walk()
            v.sync()
            for _ in range(60):
                if not v.is_running():
                    break
                time.sleep(1 / 30)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--steps", type=int, default=None, help="number of steps (default: analysis.gait.n_steps)")
    ap.add_argument("--view", action="store_true", help="watch the walk in the viewer")
    ap.add_argument("--json", default=None, help="write the run summary here")
    ap.add_argument("--baseline", default=None, help="compare against this summary JSON")
    args = ap.parse_args(argv)
    return run(args.config, args.steps, args.view, args.json, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
