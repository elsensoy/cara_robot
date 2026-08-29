#!/usr/bin/env python3
"""U7 -- controlled single-foot unloading (the first balance/control phase).

Milestone question (roadmap Phase U7):

    "Can Cara reach a physically valid PRE-SINGLE-SUPPORT configuration?"

i.e. gradually transfer weight toward one foot until the OTHER foot's vertical
load approaches zero, WITHOUT committing to a lift, while the whole-body COM
sits inside the STANCE foot's own support polygon (with margin), the pelvis
stays near level, the feet do not slip and no actuator saturates.

Two transparent quasi-static phases (no gains tuned, no RL):

  A. the frontal-plane IK from `weight_shift.py` (free = {hip_roll, ankle_roll}
     per leg, feet flat + planted) shifts the lateral COM target toward the
     stance foot;
  B. the swing leg is then gently shortened -- a 6-DoF foot IK with the foot
     target raised a fraction of a millimetre at a time -- until its vertical
     load crosses `accept.unloaded_frac_target`.  The clearance is FROZEN there;
     the foot is not raised any further.

A state counts as valid pre-single-support when, at that frozen point:
  * the unloaded foot Fz is at/below the target fraction of body weight,
  * it got there before the swing foot rose more than `accept.not_lifted_rise`,
  * the whole-body COM is inside the STANCE foot polygon with margin,
  * pelvis tilt / foot slip / actuator load stay within limits.

Failure cases are reported, not hidden.

Requires `mujoco` (brings numpy).  Prints SKIPPED / exits 0 without it.

Usage:
    python3 unload_foot.py                          # full body, both sides
    python3 unload_foot.py config/cara_lower_body.yaml
    python3 unload_foot.py --view --stance r_       # watch the left foot unload
    python3 unload_foot.py --json baselines/full_body_unload.json
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

SIDE = {"l_": +1.0, "r_": -1.0}          # stance foot -> sign of the lateral COM target
OTHER = {"l_": "r_", "r_": "l_"}


def run(config, view, view_stance, json_path, baseline_path):
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
                print(f"WARNING: {on_disk} is stale -- run "
                      f"`generate_mjcf.py --dynamic {config or ''}` (fresh render used here)\n")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    uf = (spec.get("analysis", {}) or {}).get("unload_foot", {}) or {}
    if not uf:
        print("this config has no analysis.unload_foot block")
        return 2
    base_pose = uf.get("base_pose", "stand_nominal")
    com_sweep = [float(x) for x in uf.get("com_sweep", [0.02, 0.025, 0.03, 0.035])]
    C_MAX = float(uf.get("max_unweight_clearance", 0.006))
    ramp = float(uf.get("ramp_seconds", 3.0))
    hold = float(uf.get("hold_seconds", 2.0))
    settle = float(uf.get("settle_seconds", 1.5))
    KA = float(uf.get("roll_trim_kp_ankle", 1.6))
    KH = float(uf.get("roll_trim_kp_hip", 0.8))
    KD = float(uf.get("roll_trim_kd_ankle", 0.10))
    acc = uf.get("accept", {}) or {}
    UNL_FRAC = float(acc.get("unloaded_frac_target", 0.05))
    NOT_LIFTED_RISE = float(acc.get("not_lifted_rise", 0.0015))
    MIN_STANCE_MARGIN = float(acc.get("min_stance_margin", 0.005))
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 6.0)))
    MAX_SLIP = float(acc.get("max_foot_slip", 0.004))
    MAX_TQ_FRAC = float(acc.get("max_torque_frac", 1.0))
    MIN_CORNERS = int(acc.get("min_stance_corners", 3))

    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)[base_pose]
    nominal_ctrl = [float(base_cfg.get(n, 0.0)) for n in jn]
    nfp = lm.nominal_foot_poses(spec, base_cfg)

    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose)
    foot_gid = {"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")}
    floor_gid = gid("floor")
    forcerng = np.array([model.actuator_forcerange[aid(n)][1] for n in jn])
    total_weight = float(sum(model.body_mass)) * lm.analysis_gravity(spec)
    unl_target_N = UNL_FRAC * total_weight

    table = wsh.build_ik_table(spec, base_cfg, max(com_sweep) + 0.03, 61)

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

    def settle_centre():
        mujoco.mj_resetDataKeyframe(model, data, kid)
        for _ in range(int(settle / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)

    j_index = {n: i for i, n in enumerate(jn)}
    sagittal = {s: [s + "hip_pitch", s + "knee_pitch", s + "ankle_pitch"] for s in ("l_", "r_")}

    def build_swing_table(unld, com_cfg, n=13):
        """clearance -> swing-leg SAGITTAL joint angles that raise the foot while
        keeping it level, measured from the foot's ACTUAL position in the
        COM-shifted config (task = {foot z, foot pitch}).  The frontal-plane
        balance (hip_roll / ankle_roll) is left at the COM-shift solution.
        Built offline, interpolated per sim step."""
        free = sagittal[unld]
        q0 = {**base_cfg, **com_cfg}
        tf0 = lm.forward_kinematics(spec, q0)
        sw = lm.frame_world_position(spec, tf0, unld + "foot_sole_center")
        rot = tf0[unld + "foot"][0]
        rows, q = [], dict(q0)
        for i in range(n):
            c = C_MAX * i / (n - 1)
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

    def maneuver(com_target, stance):
        """Phase A (COM shift) then Phase B (freeze-on-crossing swing unweight).
        Returns a dict of the frozen-point + held-window measurements."""
        unld = OTHER[stance]
        cy_target = SIDE[stance] * com_target
        _, qt_com = wsh.table_lookup(table, cy_target)
        com_cfg = {jn[i]: qt_com[i] for i in range(len(jn))}
        swing_rows = build_swing_table(unld, com_cfg)

        settle_centre()
        foot0 = {p: np.array(data.geom_xpos[foot_gid[p]][:2]) for p in ("l_", "r_")}
        swing_z0 = float(data.geom_xpos[foot_gid[unld]][2])

        # ---- Phase A: ramp the lateral COM target (both feet planted) --------
        for step in range(int(ramp / dt)):
            cy = cy_target * wsh.smoothstep(step * dt / ramp)
            _, qt = wsh.table_lookup(table, cy)
            data.ctrl[:] = qt
            mujoco.mj_step(model, data)
        for _ in range(int(0.6 / dt)):
            data.ctrl[:] = qt_com
            mujoco.mj_step(model, data)

        cmd = dict(com_cfg)
        a_roll, h_roll = stance + "ankle_roll", stance + "hip_roll"
        prev_roll = wsh.quat_rpy(data.qpos[3:7])[0]

        # ---- Phase B: shorten the swing leg until its Fz crosses, then freeze.
        #      A minimal stance-roll trim (same as lift_foot / U8) keeps her upright.
        clearance = 0.0
        frozen = None                      # measurement at the Fz crossing
        nB = int((ramp + hold) / dt)
        ramp_steps = max(1, int(ramp / dt))
        held = {"fz_st": [], "fz_unld": [], "margin": [], "tilt": [], "slip": [],
                "sat": 0, "corners_st": 99}
        for step in range(nB):
            if frozen is None and step < ramp_steps:
                clearance = C_MAX * (step / ramp_steps)
            for n, v in zip(sagittal[unld], swing_lookup(swing_rows, clearance)):
                cmd[n] = v
            roll_now, _p, _y = wsh.quat_rpy(data.qpos[3:7])
            roll_rate = (roll_now - prev_roll) / dt
            prev_roll = roll_now
            c2 = dict(cmd)
            c2[a_roll] = com_cfg[a_roll] + SIDE[stance] * (KA * roll_now + KD * roll_rate)
            c2[h_roll] = com_cfg[h_roll] - SIDE[stance] * KH * roll_now
            data.ctrl[:] = [c2[n] for n in jn]
            mujoco.mj_step(model, data)

            fz = {"l_": foot_normal_force(foot_gid["l_"]), "r_": foot_normal_force(foot_gid["r_"])}
            com = data.subtree_com[0]
            cxy = (float(com[0]), float(com[1]))
            st_margin = lm.polygon_signed_margin(foot_polygon(foot_gid[stance]), cxy)
            roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
            tilt = max(abs(roll), abs(pitch))
            slip = max(float(np.linalg.norm(data.geom_xpos[foot_gid[p]][:2] - foot0[p]))
                       for p in ("l_", "r_"))
            swing_rise = float(data.geom_xpos[foot_gid[unld]][2]) - swing_z0
            tau = np.array([data.actuator_force[aid(n)] for n in jn])
            sat = bool(np.any(np.abs(tau) >= forcerng - 1e-6))

            if frozen is None and fz[unld] <= unl_target_N:
                frozen = {"clearance": clearance, "swing_rise": swing_rise,
                          "st_margin": st_margin, "tilt": tilt, "slip": slip,
                          "fz_unld": fz[unld], "fz_st": fz[stance],
                          "tq_frac": float(np.max(np.abs(tau) / forcerng))}

            if step >= nB - int(min(hold, 1.0) / dt):     # final <=1 s window
                held["fz_st"].append(fz[stance]); held["fz_unld"].append(fz[unld])
                held["margin"].append(st_margin); held["tilt"].append(tilt)
                held["slip"].append(slip)
                held["sat"] += int(sat)
                held["corners_st"] = min(held["corners_st"], foot_corners(foot_gid[stance]))

        h = {"fz_unld": float(np.mean(held["fz_unld"])),
             "fz_st": float(np.mean(held["fz_st"])),
             "st_margin": float(np.min(held["margin"])),
             "tilt": float(np.max(held["tilt"])),
             "slip": float(np.max(held["slip"])),
             "sat": held["sat"], "corners_st": held["corners_st"]}
        return {"stance": stance, "com_target": com_target, "frozen": frozen, "held": h,
                "crossed": frozen is not None}


    def classify(m):
        f, h = m["frozen"], m["held"]
        if f is None:
            return False, ["never-reached-Fz-target"]
        bits = []
        if f["swing_rise"] > NOT_LIFTED_RISE:      bits.append("only-by-lifting")
        if h["st_margin"] <= MIN_STANCE_MARGIN:    bits.append("COM-outside-stance")
        if h["tilt"] >= MAX_TILT:                  bits.append("tilt")
        if h["slip"] >= MAX_SLIP:                  bits.append("slip")
        if h["sat"] != 0:                          bits.append("torque")
        if h["corners_st"] < MIN_CORNERS:          bits.append("stance-lifting")
        return (not bits), bits

    # ================================================================== #
    print(f"Controlled single-foot unloading (U7)  base '{base_pose}'  {model_name}")
    print(f"{sum(model.body_mass):.2f} kg ({total_weight:.1f} N)  |  "
          f"target: unloaded foot Fz <= {100*UNL_FRAC:.0f}% weight ({unl_target_N:.1f} N), "
          f"reached with swing-foot rise < {1e3*NOT_LIFTED_RISE:.1f} mm")

    if view:
        cy_t = SIDE[view_stance] * max(com_sweep)
        _, qt_v = wsh.table_lookup(table, cy_t)
        sw_rows = build_swing_table(OTHER[view_stance],
                                    {jn[i]: qt_v[i] for i in range(len(jn))})
        return _view(model, data, dt, table, sw_rows, swing_lookup, sagittal, base_pose, settle,
                     ramp, hold, nominal_ctrl, max(com_sweep), C_MAX, view_stance, jn,
                     j_index, floor_gid, mujoco, np)

    hdr = (f"  {'COMtgt':>7} {'stance':>6} {'Fz_unld':>8} {'%W':>5} {'rise mm':>8} "
           f"{'st.margin':>9} {'roll°':>6} {'slip mm':>8} {'st.corn':>7} {'τ%':>5}  verdict")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))

    best = {"l_": None, "r_": None}
    valid = {"l_": None, "r_": None}
    for com_t in com_sweep:
        for stance in ("l_", "r_"):
            m = maneuver(com_t, stance)
            ok, bits = classify(m)
            f, h = m["frozen"], m["held"]
            if f is None:
                print(f"  {com_t:>7.3f} {stance:>6} {'--':>8} {'--':>5} {'--':>8} "
                      f"{'--':>9} {'--':>6} {'--':>8} {'--':>7} {'--':>5}  FAIL: never reached Fz target")
            else:
                print(f"  {com_t:>7.3f} {stance:>6} {f['fz_unld']:>8.2f} "
                      f"{100*f['fz_unld']/total_weight:>4.1f}% {1e3*f['swing_rise']:>8.2f} "
                      f"{1e3*h['st_margin']:>9.1f} {math.degrees(h['tilt']):>6.2f} "
                      f"{1e3*h['slip']:>8.2f} {h['corners_st']:>7d} {100*f['tq_frac']:>4.0f}  "
                      f"{'PASS' if ok else 'FAIL: ' + ','.join(bits)}")
            row = None if f is None else {
                "com_target": com_t, "Fz_unloaded_N": f["fz_unld"],
                "unloaded_frac": f["fz_unld"] / total_weight,
                "swing_rise_mm": 1e3 * f["swing_rise"],
                "stance_margin_mm": 1e3 * h["st_margin"],
                "pelvis_tilt_deg": math.degrees(h["tilt"]), "slip_mm": 1e3 * h["slip"],
                "stance_corners": h["corners_st"]}
            if row and (best[stance] is None or row["Fz_unloaded_N"] < best[stance]["Fz_unloaded_N"]):
                best[stance] = row
            if ok and valid[stance] is None:
                valid[stance] = row

    if (valid["l_"] and valid["r_"]
            and abs(valid["l_"]["Fz_unloaded_N"] - valid["r_"]["Fz_unloaded_N"]) < 0.05
            and abs(valid["l_"]["stance_margin_mm"] - valid["r_"]["stance_margin_mm"]) < 0.1):
        print("\n(the l_ and r_ rows match -- Cara is sagittally symmetric, as expected)")

    print("\nper-side result:")
    for stance in ("l_", "r_"):
        unld = OTHER[stance][:-1]
        v, b = valid[stance], best[stance]
        if v:
            print(f"  stance {stance[:-1]:>5}: valid pre-single-support at COM target "
                  f"{v['com_target']:.3f} m -- {unld} foot Fz {v['Fz_unloaded_N']:.2f} N "
                  f"({100*v['unloaded_frac']:.1f}% weight), swing rise {v['swing_rise_mm']:.2f} mm, "
                  f"stance margin {v['stance_margin_mm']:.1f} mm")
        elif b:
            print(f"  stance {stance[:-1]:>5}: NO valid state -- best {unld} foot Fz "
                  f"{b['Fz_unloaded_N']:.2f} N ({100*b['unloaded_frac']:.1f}% weight) "
                  f"at COM target {b['com_target']:.3f} m (rise {b['swing_rise_mm']:.2f} mm, "
                  f"stance margin {b['stance_margin_mm']:.1f} mm)")
        else:
            print(f"  stance {stance[:-1]:>5}: NO valid state -- Fz target never reached")

    both_ok = valid["l_"] is not None and valid["r_"] is not None
    results = {"model": model_name, "total_mass_kg": float(sum(model.body_mass)),
               "total_weight_N": total_weight, "unloaded_frac_target": UNL_FRAC,
               "left_stance": valid["l_"] or {"best": best["l_"]},
               "right_stance": valid["r_"] or {"best": best["r_"]},
               "milestone_met": both_ok}
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsummary -> {json_path}")
    if baseline_path and os.path.exists(baseline_path):
        base = json.load(open(baseline_path, encoding="utf-8"))
        print(f"\nDelta vs baseline '{base.get('model','?')}' "
              f"({base.get('total_mass_kg',0):.2f} -> {results['total_mass_kg']:.2f} kg):")
        for side in ("left_stance", "right_stance"):
            b, c = base.get(side, {}), results.get(side, {})
            if isinstance(b, dict) and "Fz_unloaded_N" in b and "Fz_unloaded_N" in c:
                print(f"  {side:<13} Fz_unloaded {c['Fz_unloaded_N']:.2f} N "
                      f"({c['Fz_unloaded_N']-b['Fz_unloaded_N']:+.2f})")

    print("\n" + "=" * 70)
    print(f"MILESTONE {'MET' if both_ok else 'NOT MET'}: Cara "
          f"{'can' if both_ok else 'cannot yet'} reach a valid pre-single-support "
          f"configuration on {'both' if both_ok else 'both'} feet.")
    print("  (the swing foot is unweighted, not lifted; provisional masses / PD gains / friction; no RL)")
    return 0 if both_ok else 1


def _view(model, data, dt, table, swing_rows, swing_lookup, sagittal, base_pose, settle, ramp,
          hold, nominal_ctrl, com_t, C_MAX, stance, jn, j_index, floor_gid, mujoco, np):
    import time
    try:
        import mujoco.viewer
    except ImportError:
        print("error: mujoco.viewer unavailable (needs a display)", file=sys.stderr)
        return 2
    unld = OTHER[stance]
    cy_target = SIDE[stance] * com_t
    floor_z = float(model.geom_pos[floor_gid][2])
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose)

    print(f"\nviewer: shift COM onto the {stance[:-1]} foot, then unweight the {unld[:-1]} foot. "
          f"green = COM target, orange = measured COM. close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_resetDataKeyframe(model, data, kid)
            for _ in range(int(settle / dt)):
                data.ctrl[:] = nominal_ctrl
                mujoco.mj_step(model, data)
            seg = [("A", int(ramp / dt)), ("hold", int(0.4 / dt)),
                   ("B", int(ramp / dt)), ("hold", int(hold / dt))]
            cmd = {n: float(data.qpos[7 + j_index[n]]) for n in jn}
            for name, n in seg:
                for s in range(n):
                    frac = s / max(1, n)
                    if name == "A":
                        cy = cy_target * wsh.smoothstep(frac)
                        _, qt = wsh.table_lookup(table, cy)
                        cmd = {jn[i]: qt[i] for i in range(len(jn))}
                    else:
                        cy = cy_target
                        _, qt = wsh.table_lookup(table, cy_target)
                        cmd = {jn[i]: qt[i] for i in range(len(jn))}
                        c = C_MAX * (frac if name == "B" else 1.0)
                        for nm, val in zip(sagittal[unld], swing_lookup(swing_rows, c)):
                            cmd[nm] = val
                    data.ctrl[:] = [cmd[x] for x in jn]
                    mujoco.mj_step(model, data)
                    com = data.subtree_com[0]
                    v.user_scn.ngeom = 0
                    for pos, rgba in (((float(com[0]), cy, floor_z + 0.001), (0.2, 0.8, 0.3, 1)),
                                      ((float(com[0]), float(com[1]), floor_z + 0.002), (0.95, 0.55, 0.15, 1))):
                        g = v.user_scn.geoms[v.user_scn.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                                            list(pos), np.eye(3).flatten(), list(rgba))
                        v.user_scn.ngeom += 1
                    v.sync()
                    if not v.is_running():
                        return 0
                    time.sleep(dt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--view", action="store_true",
                    help="watch one side unload in the MuJoCo viewer (needs a display)")
    ap.add_argument("--stance", default="r_", choices=("l_", "r_"),
                    help="--view: which foot stays as stance (default r_ -> the left foot unloads)")
    ap.add_argument("--json", default=None, help="write the run summary here")
    ap.add_argument("--baseline", default=None, help="print deltas vs this summary JSON")
    args = ap.parse_args(argv)
    return run(args.config, args.view, args.stance, args.json, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
