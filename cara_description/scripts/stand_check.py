#!/usr/bin/env python3
"""Milestone check: can Cara's lower body hold standing poses under PD control?

Loads the FLOATING-base dynamic MJCF (pelvis + both legs, gravity on, a PD
<position> servo per leg joint, feet <-> ground contact) and, for each pose in
`analysis.standing_poses`, commands the servos to that pose and holds for
`analysis.hold_seconds` (default 10 s).  For each pose it reports:

  upright   pelvis stayed near its rest height and near-vertical
  settled   no residual drift / velocity (not slowly tipping or oscillating)
  COM/poly  horizontal COM stays inside the foot support polygon, with margin
  effort    peak / RMS actuator torque, and whether any servo saturated
  contact   both feet planted, no floor penetration
  FK        MuJoCo body positions match leg_model.forward_kinematics

MILESTONE: hold every pose for the full duration with all checks green.

Requires `mujoco` (brings numpy). Prints SKIPPED / exits 0 without it.

Usage:
    python3 stand_check.py [config/cara_lower_body.yaml]
    python3 stand_check.py --hold 15 --verbose
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import leg_model as lm

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_lower_body.yaml"))
DYN_MJCF = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf", "cara_lower_body_dynamic.xml"))

TILT_TOL_DEG = 8.0       # pelvis tilt from vertical
HEIGHT_DROP_TOL = 0.10   # fraction of rest height the pelvis may sink
MARGIN_TOL = 0.005       # m, COM must stay this far inside the support polygon
DRIFT_TOL = 0.02         # m, pelvis horizontal drift over the hold window
QVEL_TOL = 0.10          # rad/s (joints) and m/s (pelvis) residual
PENETRATION_TOL = -3e-3  # m
FK_TOL = 1e-4            # m
SETTLE_SKIP = 2.0        # s ignored at the start before scoring


# --------------------------------------------------------------------------- #
# 2-D geometry: convex hull + signed distance to a polygon
# --------------------------------------------------------------------------- #
def convex_hull(points):
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]   # CCW


def polygon_margin(poly, q):
    """Signed distance from point q to polygon boundary. + inside, - outside."""
    if len(poly) < 3:
        return -1e9
    inside = True
    mind = 1e9
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        nx, ny = ey, -ex                       # outward normal for CCW poly
        L = math.hypot(nx, ny) or 1.0
        d = ((q[0] - a[0]) * nx + (q[1] - a[1]) * ny) / L
        if d > 1e-12:
            inside = False
        mind = min(mind, abs(d))
    return mind if inside else -mind


# --------------------------------------------------------------------------- #
def run(config, hold_override, verbose):
    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    import generate_mjcf

    spec = lm.load_spec(config)
    xml = generate_mjcf.build_mjcf(spec, dynamic=True)
    if os.path.exists(DYN_MJCF):
        with open(DYN_MJCF, encoding="utf-8") as fh:
            if fh.read() != xml:
                print(f"WARNING: {DYN_MJCF} is stale -- run "
                      "`generate_mjcf.py --dynamic config/cara_lower_body.yaml "
                      "-o mjcf/cara_lower_body_dynamic.xml` (using a fresh render here)\n")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    ana = spec.get("analysis", {}) or {}
    hold_t = float(hold_override or ana.get("hold_seconds", 10.0))
    poses = ana.get("standing_poses") or list(lm.reference_poses(spec))
    n_steps = int(hold_t / dt)
    score_from = int(SETTLE_SKIP / dt)

    jnames = lm.joint_names(spec)
    physical = list(lm.link_inertials(spec))
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    pelvis_bid = bid(spec["frame_conventions"]["base_frame"])
    foot_gids = {gid(f"{lk}_collision") for lk in physical if lk.endswith("foot")}
    floor_gid = gid("floor")
    forcerng = {n: float(model.actuator_forcerange[aid(n)][1]) for n in jnames}
    rest_h = float(model.key_qpos[0][2]) if model.nkey else 0.3
    total_mass = float(sum(model.body_mass))

    print(f"Standing milestone check  ({model.nu} PD servos, "
          f"{total_mass:.2f} kg lower body, hold {hold_t:.0f}s each)")
    hdr = (f"  {'pose':<15} {'verdict':<6} {'tilt°':>6} {'sink mm':>8} {'drift mm':>9} "
           f"{'COM margin mm':>14} {'peak|tau|':>12} {'RMS tau':>8} {'FKerr':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_pass = True
    for pose_name in poses:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, pose_name)
        mujoco.mj_resetDataKeyframe(model, data, kid)
        cfg = lm.reference_poses(spec).get(pose_name, {})
        pose_rest_h = float(data.qpos[2])   # this pose's own reset pelvis height

        pelvis_xy, tau_sq, margins, min_gap = [], np.zeros(len(jnames)), [], 0.0
        peak_tau = np.zeros(len(jnames))
        sat = np.zeros(len(jnames))
        n_scored = 0
        feet_planted = True
        for step in range(n_steps):
            mujoco.mj_step(model, data)
            if step < score_from:
                continue
            n_scored += 1
            tau = np.array([data.actuator_force[aid(n)] for n in jnames])
            tau_sq += tau * tau
            peak_tau = np.maximum(peak_tau, np.abs(tau))
            sat += (np.abs(tau) >= np.array([forcerng[n] for n in jnames]) - 1e-6)
            pelvis_xy.append((float(data.xpos[pelvis_bid][0]), float(data.xpos[pelvis_bid][1])))

            cpts, per_foot = [], {g: 0 for g in foot_gids}
            for i in range(data.ncon):
                c = data.contact[i]
                pair = {c.geom1, c.geom2}
                fg = (pair & foot_gids)
                if fg and floor_gid in pair:
                    cpts.append((float(c.pos[0]), float(c.pos[1])))
                    per_foot[next(iter(fg))] += 1
                    min_gap = min(min_gap, float(c.dist))
            if any(v == 0 for v in per_foot.values()):
                feet_planted = False
            com = data.subtree_com[0]
            margins.append(polygon_margin(convex_hull(cpts), (float(com[0]), float(com[1]))))

        mujoco.mj_forward(model, data)
        rms_tau = float(np.sqrt(tau_sq / max(n_scored, 1)).max())
        peak = float(peak_tau.max())
        peak_j = jnames[int(np.argmax(peak_tau))]
        saturated = bool((sat / max(n_scored, 1) > 0.02).any())
        tilt = math.degrees(2 * math.acos(min(1.0, abs(float(data.qpos[3])))))
        sink = pose_rest_h - float(data.xpos[pelvis_bid][2])
        xs = [p[0] for p in pelvis_xy]; ys = [p[1] for p in pelvis_xy]
        drift = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if xs else 0.0
        qvel_j = float(np.abs(data.qvel[6:]).max())
        pel_v = float(np.abs(data.qvel[:3]).max())
        min_margin = min(margins) if margins else -1e9

        # FK consistency: MuJoCo body positions expressed in the PELVIS frame
        # (rotate out the free-joint orientation) vs analytic forward_kinematics.
        tf = lm.forward_kinematics(spec, {n: float(data.qpos[7 + i]) for i, n in enumerate(jnames)})
        base_name = spec["frame_conventions"]["base_frame"]
        R_pel = np.array(data.xmat[pelvis_bid]).reshape(3, 3)
        fk_err = 0.0
        for name in physical:
            rel_world = np.array(data.xpos[bid(name)]) - np.array(data.xpos[pelvis_bid])
            rel_mj = R_pel.T @ rel_world
            rel_fk = lm.vec_sub(tf[name][1], tf[base_name][1])
            fk_err = max(fk_err, float(np.max(np.abs(rel_mj - np.array(rel_fk)))))

        notes = []
        ok = True
        if tilt > TILT_TOL_DEG:
            notes.append(f"TILT {tilt:.1f}deg"); ok = False
        if sink > HEIGHT_DROP_TOL * pose_rest_h:
            notes.append(f"SANK {sink*1e3:.0f}mm"); ok = False
        if drift > DRIFT_TOL:
            notes.append(f"DRIFT {drift*1e3:.0f}mm"); ok = False
        if qvel_j > QVEL_TOL or pel_v > QVEL_TOL:
            notes.append("NOT SETTLED"); ok = False
        if min_margin < MARGIN_TOL:
            notes.append(f"COM MARGIN {min_margin*1e3:.1f}mm"); ok = False
        if not feet_planted:
            notes.append("FOOT LIFTED"); ok = False
        if min_gap < PENETRATION_TOL:
            notes.append(f"PENETRATION {min_gap*1e3:.1f}mm"); ok = False
        if saturated:
            notes.append("SERVO SATURATED"); ok = False
        if fk_err > FK_TOL:
            notes.append(f"FK {fk_err:.1e}"); ok = False
        all_pass &= ok

        print(f"  {pose_name:<15} {'PASS' if ok else 'FAIL':<6} {tilt:>6.2f} {sink*1e3:>8.1f} "
              f"{drift*1e3:>9.1f} {min_margin*1e3:>14.1f} {peak:>8.3f}@{peak_j.split('_')[-1]:<3} "
              f"{rms_tau:>8.3f} {fk_err:>8.1e}"
              + (f"   {'; '.join(notes)}" if notes else ""))
        if verbose:
            print(f"      target {[round(cfg.get(n, 0.0), 2) for n in jnames]}")
            print(f"      final  {[round(float(data.qpos[7 + i]), 2) for i in range(len(jnames))]}")
            print(f"      tau    {[round(float(data.actuator_force[aid(n)]), 3) for n in jnames]}")

    print("\n" + "=" * 62)
    print(f"MILESTONE {'MET' if all_pass else 'NOT MET'}: "
          f"held {len(poses)} pose(s) for {hold_t:.0f}s "
          + ("with all checks green." if all_pass else "-- see failures above."))
    print("(masses / PD gains are provisional; no head/arms/battery yet)")
    return 0 if all_pass else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--hold", type=float, default=None, help="override hold seconds")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    return run(args.config, args.hold, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
