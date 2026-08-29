#!/usr/bin/env python3
"""U8 -- first single-support milestone: lift one foot, hold briefly, return.

Roadmap Phase U8 (only after U7 succeeds):

    lift the unloaded foot by 5-10 mm, hold briefly, return to double support.
    Do not walk yet.

This is U7's shift + unweight, plus:

  C. raise the swing foot to `lift_height` clear of the ground (closed loop on
     the *world* clearance, since the pelvis sags on the swing side as load
     transfers) and HOLD in single support for `hold_seconds`;
  D. lower it, ramp the COM back to centre, settle -> double support.

A MINIMAL balance trim is layered on the position PD during the single-support
phase: a proportional pelvis-roll -> stance {ankle_roll, hip_roll} target
correction (gains in `analysis.lift_foot`).  Without it Cara rolls ~10 deg and
catches the swing-foot edge.  A full disturbance-rejecting balance controller is
U9 -- this is just enough to hold still.

Everything else is transparent quasi-static tracking of scripted joint targets.
No RL.  Failure cases are reported, not hidden.

Requires `mujoco` (brings numpy).  Prints SKIPPED / exits 0 without it.

Usage:
    python3 lift_foot.py                          # full body, both sides + height sweep
    python3 lift_foot.py --ankle-effort 3.0       # test with a stronger ankle (provisional bump)
    python3 lift_foot.py --view --stance r_       # watch the left foot lift & return
    python3 lift_foot.py --json baselines/full_body_lift.json
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


def run(config, view, view_stance, ankle_effort, json_path, baseline_path):
    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    import generate_mjcf

    spec = lm.load_spec(config)
    model_name = spec["meta"]["name"]
    if ankle_effort is not None:
        ov = spec.setdefault("dynamics", {}).setdefault("actuators", {}).setdefault("overrides", {})
        for j in ("l_ankle_roll", "r_ankle_roll", "l_ankle_pitch", "r_ankle_pitch"):
            ov[j] = {**(ov.get(j) or {}), "effort": float(ankle_effort)}

    xml = generate_mjcf.build_mjcf(spec, dynamic=True)
    if ankle_effort is None:
        on_disk = os.path.join(_MJCF_DIR, model_name + "_dynamic.xml")
        if os.path.exists(on_disk):
            with open(on_disk, encoding="utf-8") as fh:
                if fh.read() != xml:
                    print(f"WARNING: {on_disk} is stale -- run "
                          f"`generate_mjcf.py --dynamic {config or ''}` (fresh render used here)\n")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    lf = (spec.get("analysis", {}) or {}).get("lift_foot", {}) or {}
    if not lf:
        print("this config has no analysis.lift_foot block")
        return 2
    base_pose = lf.get("base_pose", "stand_nominal")
    com_target = float(lf.get("com_target", 0.032))
    lift_heights = [float(x) for x in lf.get("lift_heights", [0.005, 0.007, 0.010])]
    ramp = float(lf.get("ramp_seconds", 4.0))
    hold = float(lf.get("hold_seconds", 1.5))
    settle = float(lf.get("settle_seconds", 1.5))
    KA = float(lf.get("roll_trim_kp_ankle", 1.6))
    KH = float(lf.get("roll_trim_kp_hip", 0.8))
    KD = float(lf.get("roll_trim_kd_ankle", 0.10))
    CG = float(lf.get("clearance_gain", 0.012))
    acc = lf.get("accept", {}) or {}
    MIN_LIFT = float(acc.get("min_lift_clearance", 0.004))
    MAX_SWING_FRAC = float(acc.get("max_swing_load_frac", 0.03))
    MIN_STANCE_MARGIN = float(acc.get("min_stance_margin", 0.005))
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 6.0)))
    MAX_SLIP = float(acc.get("max_foot_slip", 0.006))
    MAX_TQ_FRAC = float(acc.get("max_torque_frac", 1.0))
    MIN_CORNERS = int(acc.get("min_stance_corners", 3))
    MIN_RETURN_CORNERS = int(acc.get("min_return_corners", 3))

    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)[base_pose]
    nominal_ctrl = [float(base_cfg.get(n, 0.0)) for n in jn]

    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose)
    foot_gid = {"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")}
    floor_gid = gid("floor")
    forcerng = np.array([model.actuator_forcerange[aid(n)][1] for n in jn])
    total_weight = float(sum(model.body_mass)) * lm.analysis_gravity(spec)

    table = wsh.build_ik_table(spec, base_cfg, com_target + 0.03, 61)
    sagittal = {s: [s + "hip_pitch", s + "knee_pitch", s + "ankle_pitch"] for s in ("l_", "r_")}
    C_TOP = 0.030

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

    def foot_polygon(fg):
        s = model.geom_size[fg]
        p = data.geom_xpos[fg]
        rmat = data.geom_xmat[fg].reshape(3, 3)
        return lm.convex_hull_2d([
            (float((p + rmat @ np.array([ex * s[0], ey * s[1], -s[2]]))[0]),
             float((p + rmat @ np.array([ex * s[0], ey * s[1], -s[2]]))[1]))
            for ex in (-1, 1) for ey in (-1, 1)])

    def build_swing_table(unld, com_cfg, n=25):
        """clearance (pelvis frame) -> swing-leg sagittal joint angles that raise
        the foot while keeping it level, from its actual COM-shifted position."""
        free = sagittal[unld]
        q0 = {**base_cfg, **com_cfg}
        tf0 = lm.forward_kinematics(spec, q0)
        sw = lm.frame_world_position(spec, tf0, unld + "foot_sole_center")
        rot = tf0[unld + "foot"][0]
        rows, q = [], dict(q0)
        for i in range(n):
            c = C_TOP * i / (n - 1)
            sol, _r = lm.leg_ik(spec, unld, unld + "foot_sole_center",
                                (sw[0], sw[1], sw[2] + c), rot, q,
                                free_joints=free, task_rows=[2, 4], iters=200)
            q = {**q, **sol}
            rows.append((c, [sol[j] for j in free]))
        return rows

    def swing_lookup(rows, c):
        if c <= rows[0][0]:
            return rows[0][1]
        if c >= rows[-1][0]:
            return rows[-1][1]
        for i in range(len(rows) - 1):
            if rows[i][0] <= c <= rows[i + 1][0]:
                a = (c - rows[i][0]) / (rows[i + 1][0] - rows[i][0])
                return [rows[i][1][k] * (1 - a) + rows[i + 1][1][k] * a
                        for k in range(len(rows[i][1]))]
        return rows[-1][1]

    def maneuver(stance, lift_h):
        unld = OTHER[stance]
        cy_target = SIDE[stance] * com_target
        _, qt_com = wsh.table_lookup(table, cy_target)
        com_cfg = {jn[i]: qt_com[i] for i in range(len(jn))}
        swing_rows = build_swing_table(unld, com_cfg)
        a_roll, h_roll = stance + "ankle_roll", stance + "hip_roll"

        mujoco.mj_resetDataKeyframe(model, data, kid)
        for _ in range(int(settle / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)
        swing_z0 = float(data.geom_xpos[foot_gid[unld]][2])

        for step in range(int(ramp / dt)):        # Phase A: ramp COM
            _, qt = wsh.table_lookup(table, cy_target * wsh.smoothstep(step * dt / ramp))
            data.ctrl[:] = qt
            mujoco.mj_step(model, data)

        cmd = dict(com_cfg)
        ccmd = [0.0]
        prev_roll = [wsh.quat_rpy(data.qpos[3:7])[0]]

        def step_ctrl(clear_target):
            wc = float(data.geom_xpos[foot_gid[unld]][2]) - swing_z0
            ccmd[0] = min(C_TOP, max(0.0, ccmd[0] + CG * (clear_target - wc)))
            for n, v in zip(sagittal[unld], swing_lookup(swing_rows, ccmd[0])):
                cmd[n] = v
            roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
            roll_rate = (roll - prev_roll[0]) / dt
            prev_roll[0] = roll
            # the mirror flips the ankle_roll / hip_roll axis sign, so the trim
            # correction is sign-flipped between the two stance sides.
            c2 = dict(cmd)
            c2[a_roll] = com_cfg[a_roll] + SIDE[stance] * (KA * roll + KD * roll_rate)
            c2[h_roll] = com_cfg[h_roll] - SIDE[stance] * KH * roll
            data.ctrl[:] = [c2[n] for n in jn]
            mujoco.mj_step(model, data)
            return roll, pitch, wc

        # Phase C: ramp the world clearance up, then hold
        H = {"clear": [], "sw_fz": [], "st_fz": [], "margin": [], "tilt": [], "slip": [],
             "sat": 0, "corners_st": 99, "tau": np.zeros(len(jn))}
        for step in range(int(ramp / dt)):
            step_ctrl(lift_h * wsh.smoothstep(step * dt / ramp))
        stance_hold0 = np.array(data.geom_xpos[foot_gid[stance]][:2])   # slip datum = start of the hold
        for step in range(int(hold / dt)):
            roll, pitch, wc = step_ctrl(lift_h)
            if step < int(0.4 / dt):
                continue
            com = data.subtree_com[0]
            cxy = (float(com[0]), float(com[1]))
            tau = np.array([data.actuator_force[aid(n)] for n in jn])
            H["clear"].append(wc)
            H["sw_fz"].append(foot_normal_force(foot_gid[unld]))
            H["st_fz"].append(foot_normal_force(foot_gid[stance]))
            H["margin"].append(lm.polygon_signed_margin(foot_polygon(foot_gid[stance]), cxy))
            H["tilt"].append(max(abs(roll), abs(pitch)))
            H["slip"].append(float(np.linalg.norm(data.geom_xpos[foot_gid[stance]][:2] - stance_hold0)))
            H["corners_st"] = min(H["corners_st"], foot_corners(foot_gid[stance]))
            if np.any(np.abs(tau) >= forcerng - 0.02):
                H["sat"] += 1
            H["tau"] = np.maximum(H["tau"], np.abs(tau))

        # Phase D: lower, ramp COM back, settle
        for step in range(int(ramp / dt)):
            step_ctrl(lift_h * (1.0 - wsh.smoothstep(step * dt / ramp)))
        for step in range(int(ramp / dt)):
            _, qt = wsh.table_lookup(table, cy_target * (1.0 - wsh.smoothstep(step * dt / ramp)))
            data.ctrl[:] = qt
            mujoco.mj_step(model, data)
        ret = {"corners": 99, "tilt": 0.0}
        for _ in range(int(1.0 / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)
            roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
            ret["corners"] = min(ret["corners"], foot_corners(foot_gid["l_"]), foot_corners(foot_gid["r_"]))
            ret["tilt"] = max(ret["tilt"], abs(roll), abs(pitch))

        worst_j = jn[int(np.argmax(H["tau"] / forcerng))]
        return {
            "clear": float(np.min(H["clear"])), "sw_fz": float(np.mean(H["sw_fz"])),
            "st_fz": float(np.mean(H["st_fz"])), "margin": float(np.min(H["margin"])),
            "tilt": float(np.max(H["tilt"])), "slip": float(np.max(H["slip"])),
            "sat": H["sat"], "corners_st": H["corners_st"],
            "tq_frac": float(np.max(H["tau"] / forcerng)), "worst_j": worst_j,
            "ret_corners": ret["corners"], "ret_tilt": ret["tilt"],
        }

    def classify(m):
        bits = []
        if m["clear"] < MIN_LIFT:                          bits.append("foot-not-clear")
        if m["sw_fz"] > MAX_SWING_FRAC * total_weight:     bits.append("swing-loaded")
        if m["margin"] <= MIN_STANCE_MARGIN:              bits.append("COM-outside-stance")
        if m["tilt"] >= MAX_TILT:                         bits.append("tilt")
        if m["slip"] >= MAX_SLIP:                         bits.append("stance-slip")
        if m["sat"] != 0 or m["tq_frac"] > MAX_TQ_FRAC:   bits.append(f"torque({m['worst_j']})")
        if m["corners_st"] < MIN_CORNERS:                 bits.append("stance-lifting")
        if m["ret_corners"] < MIN_RETURN_CORNERS:         bits.append("bad-return")
        return (not bits), bits

    # ================================================================== #
    ar = float(forcerng[jn.index("l_ankle_roll")])
    print(f"Single-support: lift one foot (U8)  base '{base_pose}'  {model_name}")
    print(f"{sum(model.body_mass):.2f} kg ({total_weight:.1f} N)  |  COM shift {com_target:.3f} m, "
          f"then lift the free foot; hold {hold:.1f} s.  ankle effort +-{ar:.1f} N*m  "
          f"|  roll trim kp {KA:.1f}/{KH:.1f} kd {KD:.2f} (ankle/hip)")

    if view:
        cy = SIDE[view_stance] * com_target
        _, qt_v = wsh.table_lookup(table, cy)
        sw = build_swing_table(OTHER[view_stance], {jn[i]: qt_v[i] for i in range(len(jn))})
        return _view(model, data, dt, table, sw, swing_lookup, sagittal, base_pose, settle,
                     ramp, hold, nominal_ctrl, com_target, max(lift_heights), KA, KH, KD, CG,
                     view_stance, jn, foot_gid, floor_gid, mujoco, np)

    hdr = (f"  {'lift':>6} {'stance':>6} {'clear mm':>9} {'sw Fz':>7} {'st Fz %W':>9} "
           f"{'st.margin':>9} {'roll°':>6} {'slip':>6} {'τ%(worst)':>16} {'ret':>4}  verdict")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))

    valid = {"l_": None, "r_": None}
    best = {"l_": None, "r_": None}
    for lift_h in lift_heights:
        for stance in ("l_", "r_"):
            m = maneuver(stance, lift_h)
            ok, bits = classify(m)
            print(f"  {1e3*lift_h:>6.1f} {stance:>6} {1e3*m['clear']:>9.2f} {m['sw_fz']:>7.2f} "
                  f"{100*m['st_fz']/total_weight:>8.0f}% {1e3*m['margin']:>9.1f} "
                  f"{math.degrees(m['tilt']):>6.2f} {1e3*m['slip']:>6.2f} "
                  f"{100*m['tq_frac']:>6.0f} ({m['worst_j']:>6}) {m['ret_corners']:>4d}  "
                  f"{'PASS' if ok else 'FAIL: ' + ','.join(bits)}")
            row = {"lift_height_m": lift_h, "clearance_mm": 1e3 * m["clear"],
                   "swing_Fz_N": m["sw_fz"], "stance_Fz_pct": 100 * m["st_fz"] / total_weight,
                   "stance_margin_mm": 1e3 * m["margin"], "pelvis_tilt_deg": math.degrees(m["tilt"]),
                   "stance_slip_mm": 1e3 * m["slip"], "peak_torque_pct": 100 * m["tq_frac"],
                   "worst_actuator": m["worst_j"], "return_corners": m["ret_corners"]}
            if ok and valid[stance] is None:
                valid[stance] = row
            if best[stance] is None or m["clear"] > best[stance]["_c"]:
                best[stance] = {**row, "_c": m["clear"], "_bits": bits}

    if (valid["l_"] and valid["r_"]
            and abs(valid["l_"]["stance_margin_mm"] - valid["r_"]["stance_margin_mm"]) < 0.1):
        print("\n(the l_ and r_ rows match -- Cara is sagittally symmetric, as expected)")

    print("\nper-side result:")
    for stance in ("l_", "r_"):
        free = OTHER[stance][:-1]
        v, b = valid[stance], best[stance]
        if v:
            print(f"  stance {stance[:-1]:>5}: single-support HELD -- {free} foot {v['clearance_mm']:.1f} mm "
                  f"clear at {v['swing_Fz_N']:.2f} N, stance carries {v['stance_Fz_pct']:.0f}% weight, "
                  f"COM margin {v['stance_margin_mm']:.1f} mm, tilt {v['pelvis_tilt_deg']:.2f}°, clean return")
        else:
            print(f"  stance {stance[:-1]:>5}: the lift/hold/return MOTION works ({b['clearance_mm']:.1f} mm clear, "
                  f"COM margin {b['stance_margin_mm']:.1f} mm, tilt {b['pelvis_tilt_deg']:.2f}°) "
                  f"but fails: {','.join(b['_bits'])}")

    both_ok = valid["l_"] is not None and valid["r_"] is not None
    results = {"model": model_name, "total_mass_kg": float(sum(model.body_mass)),
               "total_weight_N": total_weight, "ankle_effort_Nm": ar,
               "roll_trim_kp": [KA, KH], "com_target_m": com_target,
               "left_stance": valid["l_"] or {"best": {k: v for k, v in best["l_"].items() if not k.startswith("_")}},
               "right_stance": valid["r_"] or {"best": {k: v for k, v in best["r_"].items() if not k.startswith("_")}},
               "milestone_met": both_ok}
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsummary -> {json_path}")
    if baseline_path and os.path.exists(baseline_path):
        base = json.load(open(baseline_path, encoding="utf-8"))
        print(f"\nvs baseline '{base.get('model','?')}': ankle effort "
              f"{base.get('ankle_effort_Nm','?')} -> {ar} N*m, milestone "
              f"{base.get('milestone_met')} -> {both_ok}")

    print("\n" + "=" * 70)
    if both_ok:
        print("MILESTONE MET: Cara briefly stands on one foot (free foot a few mm clear, "
              "COM inside the stance polygon) and returns to double support, both sides.")
    else:
        print("MILESTONE NOT MET: the lift/hold/return MOTION runs but a criterion above fails.")
    print("  minimal roll trim on the position PD; no stepping, no RL. (provisional masses / gains / friction)")
    return 0 if both_ok else 1


def _view(model, data, dt, table, swing_rows, swing_lookup, sagittal, base_pose, settle, ramp,
          hold, nominal_ctrl, com_target, lift_h, KA, KH, KD, CG, stance, jn, foot_gid, floor_gid,
          mujoco, np):
    import time
    try:
        import mujoco.viewer
    except ImportError:
        print("error: mujoco.viewer unavailable (needs a display)", file=sys.stderr)
        return 2
    unld = OTHER[stance]
    cy_target = SIDE[stance] * com_target
    a_roll, h_roll = stance + "ankle_roll", stance + "hip_roll"
    floor_z = float(model.geom_pos[floor_gid][2])
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose)
    print(f"\nviewer: shift onto the {stance[:-1]} foot, lift the {unld[:-1]} foot ~{1e3*lift_h:.0f} mm, "
          f"hold, return. green = COM target, orange = measured COM. close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_resetDataKeyframe(model, data, kid)
            for _ in range(int(settle / dt)):
                data.ctrl[:] = nominal_ctrl
                mujoco.mj_step(model, data)
            swing_z0 = float(data.geom_xpos[foot_gid[unld]][2])
            for s in range(int(ramp / dt)):
                _, qt = wsh.table_lookup(table, cy_target * wsh.smoothstep(s * dt / ramp))
                data.ctrl[:] = qt
                mujoco.mj_step(model, data)
            _, qt_com = wsh.table_lookup(table, cy_target)
            com_cfg = {jn[i]: qt_com[i] for i in range(len(jn))}
            cmd = dict(com_cfg)
            ccmd = [0.0]
            prev_roll = [wsh.quat_rpy(data.qpos[3:7])[0]]

            def draw(cy_marker):
                com = data.subtree_com[0]
                v.user_scn.ngeom = 0
                for pos, rgba in (((float(com[0]), cy_marker, floor_z + 0.001), (0.2, 0.8, 0.3, 1)),
                                  ((float(com[0]), float(com[1]), floor_z + 0.002), (0.95, 0.55, 0.15, 1))):
                    g = v.user_scn.geoms[v.user_scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                                        list(pos), np.eye(3).flatten(), list(rgba))
                    v.user_scn.ngeom += 1
                v.sync()

            # C: lift -> hold -> lower, with the closed-loop clearance + roll trim
            phases = [("up", int(ramp / dt), lift_h), ("hold", int(hold / dt), lift_h),
                      ("down", int(ramp / dt), 0.0)]
            for name, nsteps, tgt in phases:
                for s in range(nsteps):
                    wc = float(data.geom_xpos[foot_gid[unld]][2]) - swing_z0
                    fr = wsh.smoothstep(s / max(1, nsteps))
                    ct = tgt if name == "hold" else (lift_h * fr if name == "up" else lift_h * (1 - fr))
                    ccmd[0] = min(0.030, max(0.0, ccmd[0] + CG * (ct - wc)))
                    for n, vv in zip(sagittal[unld], swing_lookup(swing_rows, ccmd[0])):
                        cmd[n] = vv
                    roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
                    rr = (roll - prev_roll[0]) / dt
                    prev_roll[0] = roll
                    c2 = dict(cmd)
                    c2[a_roll] = com_cfg[a_roll] + SIDE[stance] * (KA * roll + KD * rr)
                    c2[h_roll] = com_cfg[h_roll] - SIDE[stance] * KH * roll
                    data.ctrl[:] = [c2[n] for n in jn]
                    mujoco.mj_step(model, data)
                    draw(cy_target)
                    if not v.is_running():
                        return 0
                    time.sleep(dt)
            # D: ramp the COM smoothly back to centre (foot down, no trim -- as headless),
            #    then settle.  A hard snap to nominal here is what makes her lurch and fall.
            for s in range(int(ramp / dt)):
                cy = cy_target * (1.0 - wsh.smoothstep(s * dt / ramp))
                _, qt = wsh.table_lookup(table, cy)
                data.ctrl[:] = qt
                mujoco.mj_step(model, data)
                draw(cy)
                if not v.is_running():
                    return 0
                time.sleep(dt)
            for _ in range(int(settle / dt)):
                data.ctrl[:] = nominal_ctrl
                mujoco.mj_step(model, data)
                draw(0.0)
                if not v.is_running():
                    return 0
                time.sleep(dt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--view", action="store_true",
                    help="watch one side lift & return in the MuJoCo viewer (needs a display)")
    ap.add_argument("--stance", default="r_", choices=("l_", "r_"),
                    help="--view: which foot stays as stance (default r_ -> the left foot lifts)")
    ap.add_argument("--ankle-effort", type=float, default=None,
                    help="override the ankle actuator effort limit [N*m] (provisional what-if)")
    ap.add_argument("--json", default=None, help="write the run summary here")
    ap.add_argument("--baseline", default=None, help="compare against this summary JSON")
    args = ap.parse_args(argv)
    return run(args.config, args.view, args.stance, args.ankle_effort, args.json, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
