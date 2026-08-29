#!/usr/bin/env python3
"""U10 -- one deliberate step.

U9 got Cara balancing on one foot; `balance_margin.py` showed the lateral
disturbance envelope is bounded by the tiny feet -- past ~1 N toward the swing
side she has to move a foot.  U10 is that move, done deliberately and
quasi-statically: from double support, shift onto one foot, lift the other,
swing it FORWARD to a new foothold, place it, and transfer the weight so she
ends in a **stable staggered stance with the pelvis advanced ~half a step** --
the first building block of a gait (U11).

Six quasi-static phases (scripted joint targets + the SAME lateral COM-y roll
trim as U9 while she is on one foot):

  A. shift the COM onto the stance foot        (weight_shift frontal IK table)
  B. lift the swing foot to `lift_height`      (closed-loop world clearance)
  C. swing it forward to the new foothold      (swing-leg IK table over progress)
  D. place it -- lower the clearance to the ground
  E. transfer -- ramp both legs to the final staggered pose (pelvis advanced),
     bringing the COM into the new, larger double-support polygon
  F. hold the new stance `hold_seconds` and check it is stable

Only the lateral roll trim runs during B-D; sagittal (COM-x) feedback is
deliberately left out (the quasi-static trajectory keeps COM-x safe on its own,
and a COM-x -> ankle_pitch term fought the swing at every gain tried -- see the
config note).  A sideways / widening step is past Cara's current lateral balance
envelope with these provisional feet (the U9 finding) -- not attempted here.

No gait, no RL.  Failure cases are reported, not hidden.

Requires `mujoco` (brings numpy).  Prints SKIPPED / exits 0 without it.

Usage:
    python3 step_once.py                       # full body, both legs + length sweep
    python3 step_once.py config/cara_lower_body.yaml
    python3 step_once.py --view --lead l_      # watch the left foot step forward
    python3 step_once.py --json baselines/full_body_step.json
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
C_TOP = 0.035          # max pelvis-frame swing-foot rise the swing table covers


def run(config, view, view_lead, json_path, baseline_path):
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

    stc = (spec.get("analysis", {}) or {}).get("step", {}) or {}
    if not stc:
        print("this config has no analysis.step block")
        return 2
    base_pose = stc.get("base_pose", "stand_nominal")
    com_target = float(stc.get("com_target", 0.028))
    lift_h = float(stc.get("lift_height", 0.010))
    step_lengths = [float(x) for x in stc.get("step_lengths", [0.02, 0.03, 0.04])]
    ramp = float(stc.get("ramp_seconds", 4.0))
    swing_s = float(stc.get("swing_seconds", 4.0))
    hold = float(stc.get("hold_seconds", 3.0))
    settle = float(stc.get("settle_seconds", 1.5))
    CG = float(stc.get("clearance_gain", 0.012))
    bal = stc.get("balance", {}) or {}
    KPA = float(bal.get("kp_ankle_roll", 50.0))
    KDA = float(bal.get("kd_ankle_roll", 10.0))
    KPH = float(bal.get("kp_hip_roll", 15.0))
    acc = stc.get("accept", {}) or {}
    PLACE_TOL = float(acc.get("place_tol", 0.010))
    MIN_MARGIN = float(acc.get("min_support_margin", 0.005))
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 8.0)))
    MAX_SLIP = float(acc.get("max_foot_slip", 0.008))
    MAX_TQ = float(acc.get("max_torque_frac", 1.0))
    MIN_STANCE_CORNERS = int(acc.get("min_stance_corners", 3))
    MIN_FINAL_CORNERS = int(acc.get("min_final_corners", 3))
    MIN_PROGRESS = float(acc.get("min_progress_frac", 0.6))
    FINAL_DRIFT = float(acc.get("final_hold_drift", 0.006))

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

    shift_table = wsh.build_ik_table(spec, base_cfg, com_target + 0.03, 61)
    tf0 = lm.forward_kinematics(spec, base_cfg)
    SOLE0 = {p: lm.frame_world_position(spec, tf0, p + "foot_sole_center") for p in ("l_", "r_")}
    ROT0 = {p: tf0[p + "foot"][0] for p in ("l_", "r_")}
    SAG = {p: [p + j for j in ("hip_pitch", "knee_pitch", "ankle_pitch")] for p in ("l_", "r_")}

    # --- contact helpers -------------------------------------------------- #
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

    # --- swing-leg trajectory table: (progress s, clearance c) -> joints -- #
    def build_swing_table(lead, com_cfg, foothold_x, ns=13, nc=8):
        """s in [0,1] walks the swing foot forward from its lifted start to
        `foothold_x` (pelvis frame); c is the extra world clearance the closed
        loop asks for.  Foot kept level; y held at its COM-shifted value."""
        free = SAG[lead]
        q0 = {**base_cfg, **com_cfg}
        tf = lm.forward_kinematics(spec, q0)
        x0, y0, z0 = lm.frame_world_position(spec, tf, lead + "foot_sole_center")
        rot = tf[lead + "foot"][0]
        grid = []
        for i in range(ns):
            s = i / (ns - 1)
            row, q = [], dict(q0)
            for k in range(nc):
                c = C_TOP * k / (nc - 1)
                sol, _r = lm.leg_ik(spec, lead, lead + "foot_sole_center",
                                    (x0 + s * (foothold_x - x0), y0, z0 + c), rot, q,
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

    # --- one step ------------------------------------------------------- #
    def maneuver(lead, step_len):
        stance = OTHER[lead]
        cy_t = SIDE[stance] * com_target
        _, qt_com = wsh.table_lookup(shift_table, cy_t)
        com_cfg = {jn[i]: qt_com[i] for i in range(len(jn))}
        a_roll, h_roll = stance + "ankle_roll", stance + "hip_roll"
        prog = 0.5 * step_len                       # how far the pelvis advances

        # final staggered pose: lead foot forward `step_len`, pelvis advanced `prog`
        final_cfg = dict(base_cfg)
        q = dict(base_cfg)
        for pfx, fwd in ((lead, step_len), (stance, 0.0)):
            sol, _r = lm.leg_ik(spec, pfx, pfx + "foot_sole_center",
                                (SOLE0[pfx][0] + fwd - prog, SOLE0[pfx][1], SOLE0[pfx][2]),
                                ROT0[pfx], q, free_joints=SAG[pfx], task_rows=[0, 2, 4], iters=400)
            q.update(sol)
            final_cfg.update({k: sol[k] for k in SAG[pfx]})
        final_ctrl = [float(final_cfg.get(n, 0.0)) for n in jn]

        grid, free = build_swing_table(lead, com_cfg, SOLE0[lead][0] + step_len)

        DBG = os.environ.get("STEP_DBG")

        def dbg(tag):
            if not DBG:
                return
            com = data.subtree_com[0]
            r, p, _ = wsh.quat_rpy(data.qpos[3:7])
            print(f"    [{tag}] com=({com[0]:+.3f},{com[1]:+.3f},{com[2]:.3f}) "
                  f"tilt={math.degrees(max(abs(r), abs(p))):.1f}deg "
                  f"FzL={foot_normal_force(foot_gid['l_']):.1f} FzR={foot_normal_force(foot_gid['r_']):.1f} "
                  f"swing_wc={1e3 * (float(data.geom_xpos[foot_gid[lead]][2]) - swing_z0):.1f}mm",
                  file=sys.stderr)

        # --- run it (STEP_DBG=1 prints per-phase state to stderr) ----- #
        mujoco.mj_resetDataKeyframe(model, data, kid)
        for _ in range(int(settle / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)
        foot_w0 = {p: np.array(data.geom_xpos[foot_gid[p]][:2]) for p in ("l_", "r_")}
        swing_z0 = float(data.geom_xpos[foot_gid[lead]][2])
        com_x0 = float(data.subtree_com[0][0])

        M = {"margin": 1e9, "tilt": 0.0, "slip": 0.0, "sat": 0, "tq": 0.0,
             "stance_corners": 99, "swing_clear_min": 1e9}
        cmd = dict(com_cfg)
        st = {"ccmd": 0.0, "ref_ey": None, "pcy": 0.0}

        def ss_step(clear_target, s_prog, record=True):
            """One sim step on one foot: closed-loop swing clearance + the swing
            table at progress `s_prog`, plus U9's lateral COM-y roll trim on the
            stance ankle_roll / hip_roll."""
            wc = float(data.geom_xpos[foot_gid[lead]][2]) - swing_z0
            st["ccmd"] = min(C_TOP, max(0.0, st["ccmd"] + CG * (clear_target - wc)))
            for n, v in zip(free, swing_lookup(grid, s_prog, st["ccmd"])):
                cmd[n] = v
            com = data.subtree_com[0]
            sf = data.geom_xpos[foot_gid[stance]]
            ey = float(com[1] - sf[1])
            if st["ref_ey"] is None:
                st["ref_ey"] = ey
            dy = ey - st["ref_ey"]
            vy = (float(com[1]) - st["pcy"]) / dt
            st["pcy"] = float(com[1])
            sfn = -SIDE[stance]          # the mirror flips the stance ankle_roll axis
            c2 = dict(cmd)
            c2[a_roll] = com_cfg[a_roll] + sfn * (KPA * dy + KDA * vy)
            c2[h_roll] = com_cfg[h_roll] + sfn * KPH * dy
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
                data.geom_xpos[foot_gid[stance]][:2] - foot_w0[stance])))
            M["sat"] += int(np.any(np.abs(tau) >= forcerng - 0.02))
            M["tq"] = max(M["tq"], float(np.max(np.abs(tau) / forcerng)))
            M["stance_corners"] = min(M["stance_corners"], foot_corners(foot_gid[stance]))
            M["swing_clear_min"] = min(M["swing_clear_min"], wc)

        # A: COM onto the stance foot
        for k in range(int(ramp / dt)):
            _, qt = wsh.table_lookup(shift_table, cy_t * wsh.smoothstep(k * dt / ramp))
            data.ctrl[:] = qt
            mujoco.mj_step(model, data)
        st["pcy"] = float(data.subtree_com[0][1])
        dbg("A")
        # B: lift the swing foot straight up
        for k in range(int(ramp / dt)):
            ss_step(lift_h * wsh.smoothstep(k * dt / ramp), 0.0, record=False)
        dbg("B")
        # C: swing it forward to the foothold
        for k in range(int(swing_s / dt)):
            ss_step(lift_h, wsh.smoothstep(k * dt / swing_s))
        dbg("C")
        # D: place -- lower the clearance to the ground, then settle briefly
        for k in range(int(ramp / dt)):
            ss_step(lift_h * (1.0 - wsh.smoothstep(k * dt / ramp)), 1.0)
        for _ in range(int(0.4 / dt)):
            ss_step(0.0, 1.0, record=False)
        place_xy = np.array(data.geom_xpos[foot_gid[lead]][:2])
        dbg("D")

        # E: transfer -- ramp both legs to the final staggered pose
        start_ctrl = np.array(data.ctrl[:])
        for k in range(int(ramp / dt)):
            f = wsh.smoothstep(k * dt / ramp)
            data.ctrl[:] = (1 - f) * start_ctrl + f * np.array(final_ctrl)
            mujoco.mj_step(model, data)
        dbg("E")
        # F: hold the new stance and check it is stable
        F = {"tilt": 0.0, "margin": 1e9, "corners": 99, "lead_corners": 99,
             "lead_fz": 1e9, "tq": 0.0}
        com_f0 = np.array(data.subtree_com[0][:2])
        drift = 0.0
        for _ in range(int(hold / dt)):
            data.ctrl[:] = final_ctrl
            mujoco.mj_step(model, data)
            com = data.subtree_com[0]
            roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
            tau = np.array([data.actuator_force[aid(n)] for n in jn])
            drift = max(drift, float(np.linalg.norm(np.array(com[:2]) - com_f0)))
            F["tilt"] = max(F["tilt"], abs(roll), abs(pitch))
            F["margin"] = min(F["margin"], lm.polygon_signed_margin(
                support_polygon(), (float(com[0]), float(com[1]))))
            F["corners"] = min(F["corners"], foot_corners(foot_gid["l_"]), foot_corners(foot_gid["r_"]))
            F["lead_corners"] = min(F["lead_corners"], foot_corners(foot_gid[lead]))
            F["lead_fz"] = min(F["lead_fz"], foot_normal_force(foot_gid[lead]))
            F["tq"] = max(F["tq"], float(np.max(np.abs(tau) / forcerng)))
        dbg("F")

        placed = bool(F["lead_corners"] >= MIN_FINAL_CORNERS
                      and F["lead_fz"] > 0.10 * total_weight)
        slip_final = max(M["slip"], float(np.linalg.norm(
            data.geom_xpos[foot_gid[stance]][:2] - foot_w0[stance])))
        progress = float(data.subtree_com[0][0]) - com_x0
        place_err = float(np.linalg.norm(
            place_xy - np.array([foot_w0[lead][0] + step_len, foot_w0[lead][1]])))
        return {
            "placed": placed, "place_err": place_err, "lead_fz": F["lead_fz"],
            "ss_margin": M["margin"], "ss_tilt": M["tilt"], "ss_slip": slip_final,
            "ss_sat": M["sat"], "ss_tq": M["tq"], "ss_stance_corners": M["stance_corners"],
            "ss_swing_clear": M["swing_clear_min"],
            "final_drift": drift, "final_tilt": F["tilt"], "final_margin": F["margin"],
            "final_corners": F["corners"], "final_tq": F["tq"],
            "progress": progress, "progress_target": prog,
        }

    def classify(m):
        bits = []
        if not m["placed"]:                              bits.append("foot-not-placed")
        if m["place_err"] > PLACE_TOL:                   bits.append(f"placement({1e3*m['place_err']:.0f}mm)")
        if m["ss_margin"] < MIN_MARGIN:                  bits.append("COM-outside-support")
        if m["ss_tilt"] >= MAX_TILT:                     bits.append("tilt(swing)")
        if m["ss_slip"] >= MAX_SLIP:                     bits.append("slip")
        if m["ss_sat"] != 0 or m["ss_tq"] > MAX_TQ:      bits.append("torque")
        if m["ss_stance_corners"] < MIN_STANCE_CORNERS:  bits.append("stance-lifting")
        if m["final_corners"] < MIN_FINAL_CORNERS:       bits.append("final-foot-lift")
        if m["final_margin"] < MIN_MARGIN:               bits.append("final-COM-outside")
        if m["final_tilt"] >= MAX_TILT:                  bits.append("final-tilt")
        if m["final_drift"] > FINAL_DRIFT:               bits.append("final-drift")
        if m["progress"] < MIN_PROGRESS * m["progress_target"]:
            bits.append(f"no-progress({1e3*m['progress']:.0f}/{1e3*m['progress_target']:.0f}mm)")
        return (not bits), bits

    # ================================================================== #
    if view:
        return _view(model, data, dt, maneuver, view_lead, step_lengths, mujoco)

    print(f"One deliberate step (U10)  base '{base_pose}'  {model_name}")
    print(f"{m_total:.2f} kg ({total_weight:.1f} N)  |  COM shift {com_target:.3f} m, lift "
          f"{1e3*lift_h:.0f} mm, swing forward, transfer  |  U9 lateral roll trim "
          f"kp {KPA:.0f}/{KPH:.0f}, kd {KDA:.0f}")

    hdr = (f"  {'step':>5} {'lead':>5} {'placed':>7} {'place':>7} {'ss margin':>9} {'ss tilt':>8} "
           f"{'slip':>6} {'ss τ%':>6} {'progress':>10} {'fin margin':>10} {'fin tilt':>8}  verdict")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))

    valid = {"l_": None, "r_": None}
    results = {"model": model_name, "total_mass_kg": m_total, "runs": []}
    for lead in ("l_", "r_"):
        for step_len in step_lengths:
            m = maneuver(lead, step_len)
            ok, bits = classify(m)
            print(f"  {1e3*step_len:>4.0f}m {lead:>5} {str(m['placed']):>7} "
                  f"{1e3*m['place_err']:>5.1f}mm {1e3*m['ss_margin']:>7.1f}mm "
                  f"{math.degrees(m['ss_tilt']):>6.1f}° {1e3*m['ss_slip']:>4.1f}mm "
                  f"{100*m['ss_tq']:>5.0f} {1e3*m['progress']:>7.1f}mm {1e3*m['final_margin']:>8.1f}mm "
                  f"{math.degrees(m['final_tilt']):>6.1f}°  {'PASS' if ok else 'FAIL: ' + ','.join(bits)}")
            results["runs"].append({"step_len_m": step_len, "lead": lead, "ok": bool(ok), "fail": bits,
                                    **{k: (v if isinstance(v, bool) else float(v))
                                       for k, v in m.items() if not isinstance(v, list)}})
            if ok and valid[lead] is None:
                valid[lead] = m

    both = valid["l_"] is not None and valid["r_"] is not None
    results["milestone_met"] = both
    if (valid["l_"] and valid["r_"]
            and abs(valid["l_"]["progress"] - valid["r_"]["progress"]) < 0.003):
        print("\n(the l_ and r_ leading rows match -- Cara is sagittally symmetric, as expected)")

    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsummary -> {json_path}")
    if baseline_path and os.path.exists(baseline_path):
        base = json.load(open(baseline_path, encoding="utf-8"))
        print(f"\nvs baseline '{base.get('model','?')}': milestone "
              f"{base.get('milestone_met')} -> {both}")

    print("\n" + "=" * 74)
    if both:
        allp = [r for r in results["runs"] if r["ok"]]
        pr = [1e3 * r["progress"] for r in allp]
        pe = max(1e3 * r["place_err"] for r in allp)
        ft = max(math.degrees(r["final_tilt"]) for r in allp)
        print(f"MILESTONE MET: Cara takes one forward step -- shift, lift, swing, place, "
              f"transfer -- and ends in a stable staggered stance, both legs leading "
              f"(COM advances ~{min(pr):.0f}-{max(pr):.0f} mm over the 20-40 mm step sweep, "
              f"foot placed within {pe:.0f} mm, final tilt < {ft:.1f}°).")
    else:
        print("MILESTONE NOT MET: " + ", ".join(
            f"{ld[:-1]} lead unmet" for ld in ("l_", "r_") if valid[ld] is None) + " -- see above.")
    print("  quasi-static + U9's lateral roll trim on the position PD; no gait, no RL. "
          "(provisional masses / gains / friction / foot size)")
    return 0 if both else 1


def _view(model, data, dt, maneuver, lead, step_lengths, mujoco):
    import time
    try:
        import mujoco.viewer
    except ImportError:
        print("error: mujoco.viewer unavailable (needs a display)", file=sys.stderr)
        return 2
    step_len = step_lengths[-1]
    print(f"\nviewer: the {lead[:-1]} foot takes one forward step ({1e3*step_len:.0f} mm), on a loop. "
          f"close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            maneuver(lead, step_len)        # drives `data` through all six phases
            v.sync()
            for _ in range(90):
                if not v.is_running():
                    break
                time.sleep(1 / 30)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--view", action="store_true", help="watch one step loop in the viewer")
    ap.add_argument("--lead", default="l_", choices=("l_", "r_"),
                    help="--view: which foot steps forward (default l_)")
    ap.add_argument("--json", default=None, help="write the run summary here")
    ap.add_argument("--baseline", default=None, help="compare against this summary JSON")
    args = ap.parse_args(argv)
    return run(args.config, args.view, args.lead, args.json, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
