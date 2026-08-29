#!/usr/bin/env python3
"""Sweep one model parameter and report the effect on foot workspace, COM and
gravitational joint torque.

Answers questions like:
    10 vs 12 vs 14 cm thigh
    heavier vs lighter thigh
    battery mass at the pelvis vs out at the foot

by overriding a single dotted path in the spec (in memory only -- the YAML on
disk is never touched) and recomputing the analysis quantities.

Metrics per swept value:
    m_total        total model mass (physical links)                    [kg]
    COMz@zero      COM height below pelvis, zero pose                   [mm]
    COMz@crouch    COM height below pelvis, deep_crouch pose            [mm]
    reach          max hip->ankle distance (sampled)                    [m]
    footX / footZ  sole travel span over a hip_pitch x knee grid        [m]
    crouchZ        vertical span of the sole over that grid             [m]
    |tau|max/j     peak |gravity hold torque| per joint over all poses  [N*m]

Usage:
    python3 morphology_sweep.py
    python3 morphology_sweep.py --param provisional_geometry.L_thigh --values 0.10,0.12,0.14
    python3 morphology_sweep.py --param dynamics.links.l_thigh.mass   --values 0.10,0.15,0.20
    python3 morphology_sweep.py --param dynamics.links.pelvis.mass    --values 0.6,1.1,1.6
"""

from __future__ import annotations

import argparse
import copy
import sys

import leg_model as lm

GRID = 13          # samples per axis for the workspace scan
SOLE = "l_foot_sole_center"


def set_dotted(spec: dict, path: str, value) -> None:
    keys = path.split(".")
    node = spec
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"path '{path}' does not exist in the spec")
    node[keys[-1]] = value


def workspace_scan(spec):
    """Span of the sole point over a grid of hip_pitch x knee_pitch."""
    lims = lm.joint_limits(spec)
    hp_lo, hp_hi = lims["l_hip_pitch"]
    kn_lo, kn_hi = lims["l_knee_pitch"]
    xs, zs = [], []
    reach = 0.0
    for i in range(GRID):
        hp = hp_lo + (hp_hi - hp_lo) * i / (GRID - 1)
        for j in range(GRID):
            kn = kn_lo + (kn_hi - kn_lo) * j / (GRID - 1)
            q = {"l_hip_pitch": hp, "l_knee_pitch": kn}
            tf = lm.forward_kinematics(spec, q)
            sole = lm.frame_world_position(spec, tf, SOLE)
            xs.append(sole[0])
            zs.append(sole[2])
            d = lm.vec_norm(lm.vec_sub(lm.link_origin(tf, "l_ankle_link"),
                                      lm.link_origin(tf, "l_hip_yaw_link")))
            reach = max(reach, d)
    return (max(xs) - min(xs), max(zs) - min(zs), reach, min(zs))


def peak_torques(spec):
    out = {j: 0.0 for j in lm.actuated_joint_names(spec)}
    for cfg in lm.reference_poses(spec).values():
        tau = lm.gravity_joint_torques(spec, cfg)
        for j, t in tau.items():
            out[j] = max(out[j], abs(t))
    return out


def analyse(spec):
    _, com0, _ = lm.center_of_mass(spec, {})
    _, comc, _ = lm.center_of_mass(spec, lm.reference_poses(spec).get("deep_crouch", {}))
    dx, dz, reach, zmin = workspace_scan(spec)
    return {
        "m_total": lm.total_mass(spec),
        "comz0": -com0[2] * 1000.0,
        "comzc": -comc[2] * 1000.0,
        "reach": reach,
        "footX": dx,
        "footZ": dz,
        "deepest": -zmin,
        "tau": peak_torques(spec),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--param", default="provisional_geometry.L_thigh")
    ap.add_argument("--values", default="0.10,0.12,0.14")
    args = ap.parse_args(argv)

    base = lm.load_spec(args.config)
    values = [float(v) for v in args.values.split(",")]
    jnames = [j for j in lm.actuated_joint_names(base) if j.startswith("l_")]

    print(f"Morphology sweep:  {args.param}  in  {values}")
    print("(all masses / lengths PROVISIONAL; +Z up, origin at pelvis)\n")
    head = (f"  {'value':>8} {'m_tot':>7} {'COMz0':>7} {'COMzc':>7} "
            f"{'reach':>7} {'footX':>7} {'footZ':>7} {'deepZ':>7}   peak |tau| per joint [N*m]")
    print(head)
    print("  " + "-" * (len(head) + 24))

    rows = []
    for v in values:
        spec = copy.deepcopy(base)
        set_dotted(spec, args.param, v)
        r = analyse(spec)
        rows.append((v, r))
        taus = "  ".join(f"{j.split('_',1)[1]}={r['tau'][j]:.3f}" for j in jnames)
        print(f"  {v:>8.4f} {r['m_total']:>7.3f} {r['comz0']:>7.1f} {r['comzc']:>7.1f} "
              f"{r['reach']:>7.3f} {r['footX']:>7.3f} {r['footZ']:>7.3f} {r['deepest']:>7.3f}   {taus}")

    if len(rows) >= 2:
        (v0, a), (v1, b) = rows[0], rows[-1]
        print(f"\n  delta ({v0} -> {v1}):")
        print(f"    total mass   {b['m_total']-a['m_total']:+.3f} kg")
        print(f"    COM depth    {b['comz0']-a['comz0']:+.1f} mm (zero)   "
              f"{b['comzc']-a['comzc']:+.1f} mm (crouch)")
        print(f"    max reach    {b['reach']-a['reach']:+.3f} m")
        print(f"    foot X span  {b['footX']-a['footX']:+.3f} m    "
              f"foot Z span {b['footZ']-a['footZ']:+.3f} m")
        for j in jnames:
            print(f"    peak |tau| {j:<15} {b['tau'][j]-a['tau'][j]:+.4f} N*m")

    print("\nColumns: COMz0/COMzc = COM mm below pelvis at zero / deep_crouch;")
    print("reach = max hip-ankle distance; footX/footZ/deepZ from a hip_pitch x knee scan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
