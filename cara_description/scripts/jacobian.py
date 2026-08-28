#!/usr/bin/env python3
"""Foot-position Jacobian for the left leg, and its validation.

The Jacobian maps joint rates to foot-point velocity:

    xdot_foot = J(q) qdot ,      J(q) = d x_foot / d q   (3 x 6)

Computed two ways and cross-checked:
  * geometric / analytic :  column j = a_j x (p_foot - o_j)     (revolute)
  * numeric              :  central finite differences of FK

Validation performed:
  1. geometric vs numeric Jacobian:            max |J_geo - J_num|
  2. xdot ~ J qdot  vs central-difference FK along a random qdot
     (this is the  xdot_foot ~ J(q) qdot  check requested)

Also prints the manipulability  w = sqrt(det(J J^T))  as a scalar measure of
how freely the foot point can move in that pose (w -> 0 near a singularity).

Later this same J connects contact force to joint torque:  tau = J^T F.

Usage:
    python3 jacobian.py                       # all reference poses
    python3 jacobian.py --pose half_crouch    # one pose, full matrices
    python3 jacobian.py --q l_knee_pitch=0.9,l_hip_pitch=-0.5
"""

from __future__ import annotations

import argparse
import math
import random
import sys

import leg_model as lm

SOLE = "l_foot_sole_center"


def parse_q(text: str) -> dict:
    q = {}
    for tok in filter(None, (t.strip() for t in text.split(","))):
        name, _, val = tok.partition("=")
        q[name.strip()] = float(val)
    return q


def _mat_max_abs_diff(a, b) -> float:
    return max(abs(x - y) for ca, cb in zip(a, b) for x, y in zip(ca, cb))


def _JJt_det(cols) -> float:
    """det(J J^T) for the 3xN position Jacobian given as a list of 3-vector columns."""
    m = [[0.0] * 3 for _ in range(3)]
    for c in cols:
        for i in range(3):
            for j in range(3):
                m[i][j] += c[i] * c[j]
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _print_matrix(names, cols, label):
    print(f"  {label}  (rows = foot x/y/z  [m per rad],  columns = joints)")
    short = [n.replace("l_", "").replace("_pitch", "P").replace("_roll", "R").replace("_yaw", "Y")
             for n in names]
    print("        " + "".join(f"{s:>11}" for s in short))
    for r, axis in enumerate("xyz"):
        print(f"     {axis}  " + "".join(f"{c[r]:>11.5f}" for c in cols))


def report_pose(spec, label, q, full, rng):
    names, jg = lm.foot_jacobian_geometric(spec, q, SOLE)
    _, jn = lm.foot_jacobian_numeric(spec, q, 1e-6, SOLE)
    d_geo_num = _mat_max_abs_diff(jg, jn)

    # xdot ~ J qdot  vs  central-difference FK along that qdot
    qdot = [rng.uniform(-1.0, 1.0) for _ in names]
    v_pred = lm.jacobian_times(jg, qdot)
    h = 1e-6
    qp = dict(q); qm = dict(q)
    for n, qd in zip(names, qdot):
        qp[n] = qp.get(n, 0.0) + h * qd
        qm[n] = qm.get(n, 0.0) - h * qd
    pp = lm.foot_position(spec, qp, SOLE)
    pm = lm.foot_position(spec, qm, SOLE)
    v_fd = tuple((a - b) / (2 * h) for a, b in zip(pp, pm))
    d_vel = max(abs(a - b) for a, b in zip(v_pred, v_fd))

    w = math.sqrt(max(0.0, _JJt_det(jg)))

    print(f"\n=== pose '{label}'  q = {q if q else '(zero)'} ===")
    print(f"  foot point {SOLE} at ({', '.join(f'{c:+.4f}' for c in lm.foot_position(spec, q, SOLE))}) m")
    if full:
        _print_matrix(names, jg, "geometric J")
    print(f"  |J_geometric - J_numeric|_max        = {d_geo_num:.2e}   (expect ~1e-7)")
    print(f"  |J*qdot - dFK/dt|_max (random qdot)   = {d_vel:.2e}   (expect ~1e-7)")
    print(f"  manipulability  w = sqrt(det(J J^T))  = {w:.5e}")
    return d_geo_num, d_vel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--pose", help="single reference pose (default: all)")
    ap.add_argument("--q", default="", help="comma list joint=radians")
    args = ap.parse_args(argv)

    spec = lm.load_spec(args.config)
    rng = random.Random(20260828)
    poses = lm.reference_poses(spec)

    print(f"Foot-position Jacobian for point '{SOLE}'  (+X fwd, +Y left, +Z up)")

    if args.q:
        report_pose(spec, "custom", parse_q(args.q), full=True, rng=rng)
        return 0

    names = [args.pose] if args.pose else list(poses)
    worst_gn = worst_v = 0.0
    for name in names:
        dg, dv = report_pose(spec, name, poses[name], full=bool(args.pose), rng=rng)
        worst_gn = max(worst_gn, dg)
        worst_v = max(worst_v, dv)

    ok = worst_gn < 1e-4 and worst_v < 1e-4
    print("\n" + "=" * 60)
    print(f"worst |J_geo - J_num| = {worst_gn:.2e}   worst velocity mismatch = {worst_v:.2e}")
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
