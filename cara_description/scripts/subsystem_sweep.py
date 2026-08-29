#!/usr/bin/env python3
"""Sweep one model parameter and report its whole-body effect on standing +
weight shifting.  The workhorse for the staged upper-body study (U2 head-mass
sweep, U3 electronics placement, U6 summary table).

Unlike `morphology_sweep.py` (pure-Python COM + analytic torque), this steps
MuJoCo: for each parameter value it settles the model standing at the base
pose, then ramps a lateral COM trajectory up in amplitude to find the
quasi-static double-support limit.  Per value it reports:

    m_total       whole-body mass                                  [kg]
    COM height    whole-body COM above the floor, standing         [m]
    COM z_pelvis  whole-body COM relative to the pelvis origin     [mm]
    tilt          pelvis tilt from vertical, standing              [deg]
    margin        COM distance inside the support polygon          [mm]
    hip/knee/ankle  peak |actuator torque| (max over both legs)    [N*m]
    shift limit   largest COM target still in controlled double support  [m]

Nothing is auto-tuned; failure cases keep the smaller limit.

Requires `mujoco`.  Prints SKIPPED / exits 0 without it.

Usage:
    python3 subsystem_sweep.py config/cara_full_body.yaml \
        --param dynamics.links.head.mass --values 0.2,0.35,0.6
    python3 subsystem_sweep.py config/cara_full_body.yaml \
        --param upper_body.head.com_z --values 0.02,0.05,0.09
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import leg_model as lm
import weight_shift as wsh   # reuse the pure IK-table / trajectory helpers

_HERE = os.path.dirname(os.path.abspath(__file__))

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


def run(config, param, values, baseline_path):
    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    import generate_mjcf

    base_spec = lm.load_spec(config)
    ws = (base_spec.get("analysis", {}) or {}).get("weight_shift", {}) or {}
    base_pose = ws.get("base_pose", "stand_nominal")
    acc = ws.get("accept", {}) or {}
    MIN_MARGIN = float(acc.get("min_support_margin", 0.005))
    MIN_OPP_FRAC = float(acc.get("min_opposite_load_frac", 0.05))
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 6.0)))
    MAX_SLIP = float(acc.get("max_foot_slip", 0.003))

    print(f"Subsystem sweep:  {param}  in  {values}   (base pose '{base_pose}')")
    print(f"model {base_spec['meta']['name']}   -- MuJoCo standing + weight-shift-limit probe\n")
    print(f"(standing metrics at '{base_pose}'; peak torque = worst over "
          f"{(base_spec.get('analysis', {}) or {}).get('standing_poses', [base_pose])})\n")
    hdr = (f"  {'value':>9} {'m_tot':>7} {'COM h':>7} {'COMz_pel':>9} {'tilt°':>6} "
           f"{'margin':>7} {'hip τ':>7} {'knee τ':>7} {'ankle τ':>8} {'shift lim':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for val in values:
        spec = copy.deepcopy(base_spec)
        set_dotted(spec, param, val)
        xml = generate_mjcf.build_mjcf(spec, dynamic=True)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        dt = model.opt.timestep
        jn = lm.actuated_joint_names(spec)
        aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        pel_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                   spec["frame_conventions"]["base_frame"])
        foot_g = {"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")}
        floor_g = gid("floor")
        forcerng = {n: float(model.actuator_forcerange[aid(n)][1]) for n in jn}
        nominal_ctrl = [float(lm.reference_poses(spec)[base_pose].get(n, 0.0)) for n in jn]

        def contact_pts_and_fz():
            pts, fz = [], {"l_": 0.0, "r_": 0.0}
            per = {"l_": 0, "r_": 0}
            for i in range(data.ncon):
                c = data.contact[i]
                pair = {c.geom1, c.geom2}
                for s, fg in foot_g.items():
                    if fg in pair and floor_g in pair:
                        pts.append((float(c.pos[0]), float(c.pos[1])))
                        per[s] += 1
                        f6 = np.zeros(6)
                        mujoco.mj_contactForce(model, data, i, f6)
                        fr = c.frame
                        fz[s] += fr[2] * f6[0] + fr[5] * f6[1] + fr[8] * f6[2]
            return pts, fz, per

        # ---- standing: settle at every standing pose, keep the worst ---- #
        stand_poses = ((base_spec.get("analysis", {}) or {}).get("standing_poses")
                       or [base_pose])
        grp = {"hip": 0.0, "knee": 0.0, "ankle": 0.0}
        com_h = com_z_pel = tilt = margin = None
        for pose in stand_poses:
            mujoco.mj_resetDataKeyframe(model, data,
                                       mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, pose))
            cfg_ctrl = [float(lm.reference_poses(spec)[pose].get(n, 0.0)) for n in jn]
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
                com = data.subtree_com[0]
                com_h = float(com[2])
                com_z_pel = float((np.array(com) - np.array(data.xpos[pel_bid]))[2]) * 1e3
                tilt = math.degrees(2 * math.acos(min(1.0, abs(float(data.qpos[3])))))
                pts, _fz, _per = contact_pts_and_fz()
                margin = lm.polygon_signed_margin(lm.convex_hull_2d(pts),
                                                  (float(com[0]), float(com[1]))) * 1e3

        # ---- weight-shift limit probe ------------------------------- #
        table = wsh.build_ik_table(spec, lm.reference_poses(spec)[base_pose],
                                   py_max=max(PROBE_AMPLITUDES) + 0.03, n=45)
        shift_limit = 0.0
        for A in PROBE_AMPLITUDES:
            traj, total, wnd = wsh.make_trajectory(A, PROBE_RAMP_S, PROBE_HOLD_S, 0.4)
            mujoco.mj_resetDataKeyframe(model, data,
                                       mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose))
            for _ in range(int(0.6 / dt)):
                data.ctrl[:] = nominal_ctrl
                mujoco.mj_step(model, data)
            fz0 = None
            worst = {"margin": 1e9, "opp": 1e9, "tilt": 0.0, "slip": 0.0, "sat": 0, "planted": True}
            foot0 = {s: np.array(data.geom_xpos[foot_g[s]][:2]) for s in foot_g}
            hold_lo, hold_hi = wnd["+A"]
            for step in range(int(total / dt)):
                cy = traj(step * dt)
                _, qt = wsh.table_lookup(table, cy)
                data.ctrl[:] = qt
                mujoco.mj_step(model, data)
                t = step * dt
                pts, fz, per = contact_pts_and_fz()
                if fz0 is None and t > 0.3:
                    fz0 = (fz["l_"], fz["r_"])
                if hold_lo <= t <= hold_hi:
                    cm = data.subtree_com[0]
                    worst["margin"] = min(worst["margin"],
                                          lm.polygon_signed_margin(lm.convex_hull_2d(pts),
                                                                   (float(cm[0]), float(cm[1]))))
                    worst["opp"] = min(worst["opp"], min(fz["l_"], fz["r_"]))
                    worst["tilt"] = max(worst["tilt"], abs(wsh.quat_rpy(data.qpos[3:7])[0]),
                                        abs(wsh.quat_rpy(data.qpos[3:7])[1]))
                    for s in foot_g:
                        worst["slip"] = max(worst["slip"],
                                            float(np.linalg.norm(data.geom_xpos[foot_g[s]][:2] - foot0[s])))
                    if per["l_"] < 3 or per["r_"] < 3:
                        worst["planted"] = False
                    if any(abs(float(data.actuator_force[aid(n)])) >= forcerng[n] - 1e-6 for n in jn):
                        worst["sat"] += 1
            total_w = float(sum(model.body_mass)) * lm.analysis_gravity(spec)
            ok = (worst["planted"] and worst["margin"] > MIN_MARGIN
                  and worst["opp"] > MIN_OPP_FRAC * total_w
                  and worst["tilt"] < MAX_TILT and worst["slip"] < MAX_SLIP
                  and worst["sat"] == 0
                  and fz0 is not None and fz["l_"] > fz0[0])   # load actually transferred
            if ok:
                shift_limit = A
            else:
                break

        rows.append({"value": val, "m_total": float(sum(model.body_mass)),
                     "com_h": com_h, "com_z_pel": com_z_pel, "tilt": tilt,
                     "margin": margin, "hip": grp["hip"], "knee": grp["knee"],
                     "ankle": grp["ankle"], "shift_limit": shift_limit})
        r = rows[-1]
        print(f"  {val:>9.4f} {r['m_total']:>7.3f} {r['com_h']:>7.3f} {r['com_z_pel']:>+9.1f} "
              f"{r['tilt']:>6.2f} {r['margin']:>7.1f} {r['hip']:>7.3f} {r['knee']:>7.3f} "
              f"{r['ankle']:>8.3f} {r['shift_limit']:>9.3f}")

    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        print(f"\n  {param}  {a['value']} -> {b['value']}:")
        print(f"    whole-body mass   {b['m_total'] - a['m_total']:+.3f} kg")
        print(f"    COM height        {(b['com_h'] - a['com_h'])*1e3:+.1f} mm  "
              f"(vs pelvis {b['com_z_pel'] - a['com_z_pel']:+.1f} mm)")
        print(f"    standing tilt     {b['tilt'] - a['tilt']:+.2f} deg")
        print(f"    support margin    {b['margin'] - a['margin']:+.1f} mm")
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

    print("\n(provisional masses / PD gains; neck joints locked at 0; no arms/ears/electronics)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", required=True, help="comma-separated")
    ap.add_argument("--baseline", default=None,
                    help="lower-body standing baseline JSON, for context")
    args = ap.parse_args(argv)
    return run(args.config, args.param, [float(v) for v in args.values.split(",")], args.baseline)


if __name__ == "__main__":
    sys.exit(main())
