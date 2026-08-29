#!/usr/bin/env python3
"""Sweep one model parameter and report its whole-body effect on standing +
weight shifting.  Workhorse for the staged upper-body study (U2 head-mass sweep,
U3 electronics, U6 summary).  Its `measure()` is reused by placement_study.py.

Unlike `morphology_sweep.py` (pure-Python COM + analytic torque), this steps
MuJoCo: for each value it settles the model at every standing pose (keeps the
worst torque), then ramps a lateral COM trajectory up in amplitude to find the
quasi-static double-support limit.  Per value it reports:

    m_total       whole-body mass                                  [kg]
    COM height    whole-body COM above the floor, standing         [m]
    COM x / z_pel whole-body COM: fore/aft, and relative to pelvis [m / mm]
    tilt          pelvis tilt from vertical, standing (base pose)   [deg]
    margin        COM distance inside the support polygon           [mm]
    hip/knee/ankle  peak |actuator torque|, worst over standing poses  [N*m]
    shift limit   largest COM target still in controlled double support  [m]

Nothing is auto-tuned; failure cases keep the smaller limit.

Requires `mujoco`.  Prints SKIPPED / exits 0 without it.

Usage:
    python3 subsystem_sweep.py config/cara_full_body.yaml \
        --param dynamics.links.head.mass --values 0.2,0.35,0.6
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import leg_model as lm
import weight_shift as wsh   # pure IK-table / trajectory helpers

STAND_SETTLE_S = 4.0
PROBE_AMPLITUDES = (0.02, 0.03, 0.04, 0.05, 0.06)
PROBE_RAMP_S = 1.8
PROBE_HOLD_S = 1.8


def set_dotted(spec, path, value):
    node = spec
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"path '{path}' not found in the spec")
    node[keys[-1]] = value


def _joint_group(name):
    for g in ("hip", "knee", "ankle"):
        if g in name:
            return g
    return "other"


def measure(spec, standing_poses=None, base_pose="stand_nominal", accept=None):
    """Step MuJoCo and return the whole-body standing + weight-shift metrics for
    an (already loaded / overridden) spec.  Dict with the keys listed in the
    module docstring, plus 'com_x'."""
    import mujoco
    import numpy as np
    import generate_mjcf

    acc = accept or {}
    MIN_MARGIN = float(acc.get("min_support_margin", 0.005))
    MIN_OPP_FRAC = float(acc.get("min_opposite_load_frac", 0.05))
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 6.0)))
    MAX_SLIP = float(acc.get("max_foot_slip", 0.003))
    standing_poses = standing_poses or [base_pose]

    model = mujoco.MjModel.from_xml_string(generate_mjcf.build_mjcf(spec, dynamic=True))
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    jn = lm.actuated_joint_names(spec)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    kid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, n)
    pel_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                               spec["frame_conventions"]["base_frame"])
    foot_g = {"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")}
    floor_g = gid("floor")
    forcerng = {n: float(model.actuator_forcerange[aid(n)][1]) for n in jn}
    total_w = float(sum(model.body_mass)) * lm.analysis_gravity(spec)
    rp = lm.reference_poses(spec)

    def contacts():
        pts, fz, per = [], {"l_": 0.0, "r_": 0.0}, {"l_": 0, "r_": 0}
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {c.geom1, c.geom2}
            for s, fgv in foot_g.items():
                if fgv in pair and floor_g in pair:
                    pts.append((float(c.pos[0]), float(c.pos[1])))
                    per[s] += 1
                    f6 = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, f6)
                    fr = c.frame
                    fz[s] += fr[2] * f6[0] + fr[5] * f6[1] + fr[8] * f6[2]
        return pts, fz, per

    # ---- standing: settle every pose, keep the worst torque -------------
    grp = {"hip": 0.0, "knee": 0.0, "ankle": 0.0}
    com = com_h = com_x = com_z_pel = tilt = margin = None
    for pose in standing_poses:
        mujoco.mj_resetDataKeyframe(model, data, kid(pose))
        cfg_ctrl = [float(rp[pose].get(n, 0.0)) for n in jn]
        pk = np.zeros(len(jn))
        for k in range(int(STAND_SETTLE_S / dt)):
            data.ctrl[:] = cfg_ctrl
            mujoco.mj_step(model, data)
            if k > int(1.0 / dt):
                pk = np.maximum(pk, [abs(float(data.actuator_force[aid(n)])) for n in jn])
        for i, n in enumerate(jn):
            grp[_joint_group(n)] = max(grp[_joint_group(n)], float(pk[i]))
        if pose == base_pose:
            mujoco.mj_forward(model, data)
            com = np.array(data.subtree_com[0])
            com_h = float(com[2])
            com_x = float(com[0])
            com_z_pel = float((com - np.array(data.xpos[pel_bid]))[2]) * 1e3
            tilt = math.degrees(2 * math.acos(min(1.0, abs(float(data.qpos[3])))))
            pts, _f, _p = contacts()
            margin = lm.polygon_signed_margin(lm.convex_hull_2d(pts), (com_x, float(com[1]))) * 1e3

    # ---- weight-shift limit probe --------------------------------------
    table = wsh.build_ik_table(spec, rp[base_pose], py_max=max(PROBE_AMPLITUDES) + 0.03, n=45)
    nominal_ctrl = [float(rp[base_pose].get(n, 0.0)) for n in jn]
    shift_limit = 0.0
    for A in PROBE_AMPLITUDES:
        traj, total, wnd = wsh.make_trajectory(A, PROBE_RAMP_S, PROBE_HOLD_S, 0.4)
        mujoco.mj_resetDataKeyframe(model, data, kid(base_pose))
        for _ in range(int(0.6 / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)
        foot0 = {s: np.array(data.geom_xpos[foot_g[s]][:2]) for s in foot_g}
        w = {"margin": 1e9, "opp": 1e9, "tilt": 0.0, "slip": 0.0, "sat": 0, "planted": True}
        fz0 = None
        lo, hi = wnd["+A"]
        for step in range(int(total / dt)):
            cy = traj(step * dt)
            _, qt = wsh.table_lookup(table, cy)
            data.ctrl[:] = qt
            mujoco.mj_step(model, data)
            t = step * dt
            pts, fz, per = contacts()
            if fz0 is None and t > 0.3:
                fz0 = (fz["l_"], fz["r_"])
            if lo <= t <= hi:
                cm = data.subtree_com[0]
                w["margin"] = min(w["margin"], lm.polygon_signed_margin(
                    lm.convex_hull_2d(pts), (float(cm[0]), float(cm[1]))))
                w["opp"] = min(w["opp"], min(fz["l_"], fz["r_"]))
                r, p, _ = wsh.quat_rpy(data.qpos[3:7])
                w["tilt"] = max(w["tilt"], abs(r), abs(p))
                for s in foot_g:
                    w["slip"] = max(w["slip"], float(np.linalg.norm(
                        data.geom_xpos[foot_g[s]][:2] - foot0[s])))
                if per["l_"] < 3 or per["r_"] < 3:
                    w["planted"] = False
                if any(abs(float(data.actuator_force[aid(n)])) >= forcerng[n] - 1e-6 for n in jn):
                    w["sat"] += 1
        ok = (w["planted"] and w["margin"] > MIN_MARGIN and w["opp"] > MIN_OPP_FRAC * total_w
              and w["tilt"] < MAX_TILT and w["slip"] < MAX_SLIP and w["sat"] == 0
              and fz0 is not None and fz["l_"] > fz0[0])
        if ok:
            shift_limit = A
        else:
            break

    return {"m_total": float(sum(model.body_mass)), "com_h": com_h, "com_x": com_x,
            "com_z_pel": com_z_pel, "tilt": tilt, "margin": margin,
            "hip": grp["hip"], "knee": grp["knee"], "ankle": grp["ankle"],
            "shift_limit": shift_limit}


def run(config, param, values, baseline_path):
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    base_spec = lm.load_spec(config)
    ws = (base_spec.get("analysis", {}) or {}).get("weight_shift", {}) or {}
    base_pose = ws.get("base_pose", "stand_nominal")
    stand_poses = (base_spec.get("analysis", {}) or {}).get("standing_poses") or [base_pose]

    print(f"Subsystem sweep:  {param}  in  {values}   (model {base_spec['meta']['name']})")
    print(f"standing metrics at '{base_pose}'; peak torque = worst over {stand_poses}\n")
    hdr = (f"  {'value':>10} {'m_tot':>7} {'COM h':>7} {'COMz_pel':>9} {'tilt°':>6} "
           f"{'margin':>7} {'hip τ':>7} {'knee τ':>7} {'ankle τ':>8} {'shift lim':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for raw in values:
        try:
            val = float(raw)
        except ValueError:
            val = raw
        spec = copy.deepcopy(base_spec)
        set_dotted(spec, param, val)
        if param.startswith("electronics."):
            lm._resolve_mounts(spec)
        r = measure(spec, stand_poses, base_pose, ws.get("accept", {}))
        rows.append((val, r))
        vs = f"{val:>10.4f}" if isinstance(val, float) else f"{str(val):>10}"
        print(f"  {vs} {r['m_total']:>7.3f} {r['com_h']:>7.3f} {r['com_z_pel']:>+9.1f} "
              f"{r['tilt']:>6.2f} {r['margin']:>7.1f} {r['hip']:>7.3f} {r['knee']:>7.3f} "
              f"{r['ankle']:>8.3f} {r['shift_limit']:>9.3f}")

    if len(rows) >= 2:
        (v0, a), (v1, b) = rows[0], rows[-1]
        print(f"\n  {param}  {v0} -> {v1}:")
        print(f"    whole-body mass   {b['m_total'] - a['m_total']:+.3f} kg")
        print(f"    COM height        {(b['com_h'] - a['com_h'])*1e3:+.1f} mm  "
              f"(vs pelvis {b['com_z_pel'] - a['com_z_pel']:+.1f} mm)")
        print(f"    standing tilt     {b['tilt'] - a['tilt']:+.2f} deg    "
              f"support margin {b['margin'] - a['margin']:+.1f} mm")
        print(f"    peak hip/knee/ankle torque  "
              f"{b['hip']-a['hip']:+.3f} / {b['knee']-a['knee']:+.3f} / {b['ankle']-a['ankle']:+.3f} N*m")
        print(f"    weight-shift limit {b['shift_limit'] - a['shift_limit']:+.3f} m")

    if baseline_path and os.path.exists(baseline_path):
        base = json.load(open(baseline_path, encoding="utf-8"))
        bp = base.get("poses", {}).get(base_pose, {})
        if bp:
            print(f"\n  standing baseline ('{base.get('model','?')}', {base.get('total_mass',0):.2f} kg): "
                  f"tilt {bp.get('tilt_deg',0):.2f} deg, margin {bp.get('com_margin_mm',0):.1f} mm, "
                  f"peak torque {bp.get('peak_torque_Nm',0):.3f} N*m")

    print("\n(provisional masses / PD gains; locked neck joints; no auto-tuning)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", required=True, help="comma-separated (numbers or mount names)")
    ap.add_argument("--baseline", default=None,
                    help="lower-body standing baseline JSON, for context")
    args = ap.parse_args(argv)
    return run(args.config, args.param, args.values.split(","), args.baseline)


if __name__ == "__main__":
    sys.exit(main())
