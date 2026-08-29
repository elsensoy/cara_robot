#!/usr/bin/env python3
"""Whole-model centre of mass for an arbitrary joint configuration.

    r_COM = (sum_i m_i r_i) / (sum_i m_i)

over the PHYSICAL links (virtual coupling links carry no mass). Optional extra
point masses (e.g. a battery, or body weight carried at the foot) can be added.

All masses are PROVISIONAL (see config/left_leg.yaml -> dynamics). This script
establishes the computation, not final numbers.

Usage:
    python3 center_of_mass.py                      # COM at every reference pose
    python3 center_of_mass.py --pose deep_crouch   # detailed breakdown, one pose
    python3 center_of_mass.py --q l_knee_pitch=1.2,l_hip_pitch=-0.4
    python3 center_of_mass.py --pose half_crouch --extra 1.5@l_foot_sole_center
    python3 center_of_mass.py path/to/other.yaml
"""

from __future__ import annotations

import argparse
import sys

import leg_model as lm


def parse_q(text: str) -> dict:
    q = {}
    for tok in filter(None, (t.strip() for t in text.split(","))):
        name, _, val = tok.partition("=")
        q[name.strip()] = float(val)
    return q


def parse_extra(items) -> list:
    out = []
    for it in items or []:
        mass, _, frame = it.partition("@")
        out.append((float(mass), frame.strip() or "l_foot_sole_center"))
    return out


def _fmt_vec(v) -> str:
    return "(" + ", ".join(f"{c:+.5f}" for c in v) + ")"


def show_one(spec, label, q, extra):
    m_tot, com, rows = lm.center_of_mass(spec, q, extra)
    print(f"\n=== {label} ===")
    print(f"  joint config: {q if q else '(zero)'}")
    if extra:
        print(f"  extra point masses: {extra}")
    print(f"  {'contributor':<22} {'mass [kg]':>10}   world position [m]")
    for name, mass, pos in rows:
        print(f"  {name:<22} {mass:>10.4f}   {_fmt_vec(pos)}")
    print(f"  {'-'*22} {'-'*10}")
    print(f"  {'TOTAL':<22} {m_tot:>10.4f}   COM = {_fmt_vec(com)}")
    print(f"  COM height below pelvis: {-com[2]*1000:.1f} mm")
    I = lm.whole_body_inertia(spec, q, about="com")
    print(f"  whole-body inertia about COM [kg·m²]:  "
          f"Ixx {I[0][0]:.5f} (roll)   Iyy {I[1][1]:.5f} (pitch)   Izz {I[2][2]:.5f} (yaw)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--pose", help="name of a reference pose from the YAML")
    ap.add_argument("--q", default="", help="comma list joint=radians")
    ap.add_argument("--extra", action="append",
                    help="point mass 'KG@frame_of_interest' (repeatable)")
    args = ap.parse_args(argv)

    spec = lm.load_spec(args.config)
    extra = parse_extra(args.extra)
    poses = lm.reference_poses(spec)

    if args.pose or args.q:
        q = parse_q(args.q) if args.q else poses[args.pose]
        show_one(spec, args.pose or "custom", q, extra)
        return 0

    # summary table over every reference pose
    print("Centre of mass by reference pose  (physical links only; masses PROVISIONAL)")
    print(f"total model mass: {lm.total_mass(spec):.4f} kg"
          + (f"  + extra {extra}" if extra else ""))
    hdr = f"\n  {'pose':<20} {'COM x':>9} {'COM y':>9} {'COM z':>9}   {'depth below pelvis':>18}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for name, cfg in poses.items():
        _, com, _ = lm.center_of_mass(spec, cfg, extra)
        print(f"  {name:<20} {com[0]:>9.4f} {com[1]:>9.4f} {com[2]:>9.4f}   {-com[2]*1000:>15.1f} mm")
    print("\n(+X forward, +Y left, +Z up; origin at the pelvis)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
