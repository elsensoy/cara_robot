#!/usr/bin/env python3
"""Approximate gravitational joint torques for the left leg.

BASE-FIXED model: the pelvis is ground. For a pose q, this reports the holding
torque each joint must supply to balance gravity acting on the leg segments
DISTAL to it:

    tau_j(q) = dU/dq_j = sum_i m_i g [ a_j x (r_i - o_j) ]_z

Two important caveats:
  * This is the torque to hold the DANGLING leg in that pose. It does NOT
    include supporting Cara's body weight. Use --carry-fraction to add body
    weight as a point load at the foot (the stance-leg case).
  * Masses / COM are PROVISIONAL (config/left_leg.yaml -> dynamics).

Cross-checks printed:
  * analytic vs central-difference of the potential energy (should match);
  * for the carried load, tau = J^T F  vs the same load via dU/dq (identical).

Usage:
    python3 gravity_torques.py
    python3 gravity_torques.py --pose deep_crouch
    python3 gravity_torques.py --carry-fraction 0.5      # 50% of body mass on this foot
    python3 gravity_torques.py --carry-fraction 1.0 --pose half_crouch
"""

from __future__ import annotations

import argparse
import math
import sys

import leg_model as lm

SOLE = "l_foot_sole_center"


def _row(name, tau):
    return f"  {name:<16} " + "  ".join(f"{tau[j]:+8.4f}" for j in tau)


def analytic_single_link_check(spec, pose_name, q):
    """Sanity: the THIGH's own contribution to the hip_pitch hold torque should
    equal the textbook pendulum term  m g r sin(theta)  (r = COM distance from
    the axis, theta = hip_pitch), when hip_yaw = hip_roll = 0.

    Compared against that same contribution taken straight from the geometric
    solver (mass * g * dz_com/dq about the hip_pitch axis).
    """
    li = lm.link_inertials(spec)["l_thigh"]
    g = lm.analysis_gravity(spec)
    r = -li.com[2]
    theta = q.get("l_hip_pitch", 0.0)
    approx = li.mass * g * r * math.sin(theta)

    tf = lm.forward_kinematics(spec, q)
    R, p = tf["l_thigh"]
    com_world = lm.vec_add(lm.mat_vec(R, li.com), p)
    for name, axis_w, o_j in lm._world_joint_axes_origins(spec, q):
        if name == "l_hip_pitch":
            dz = lm.cross(axis_w, lm.vec_sub(com_world, o_j))[2]
            exact = li.mass * g * dz
            break
    print(f"\n  thigh-only hip_pitch torque @ {pose_name}: "
          f"pendulum m*g*r*sin(th) = {approx:+.4f}   solver contribution = {exact:+.4f}   "
          f"|diff| = {abs(approx - exact):.2e} N*m")


def report_pose(spec, pose_name, q, carry_mass):
    g = lm.analysis_gravity(spec)
    extra = [(carry_mass, SOLE)] if carry_mass else None

    tau = lm.gravity_joint_torques(spec, q, extra_masses=extra)
    tau_fd = lm.gravity_joint_torques_fd(spec, q, extra_masses=extra)
    max_mismatch = max(abs(tau[j] - tau_fd[j]) for j in tau)

    print(f"\n=== pose '{pose_name}'  q = {q if q else '(zero)'} ===")
    if carry_mass:
        print(f"  carrying {carry_mass:.3f} kg at {SOLE} "
              f"(weight {carry_mass * g:.2f} N down)")
    print(f"  {'joint':<16} " + "  ".join(f"{j.split('_',1)[1]:>8}" for j in tau))
    print(_row("hold torque N*m", tau))
    print(f"  (analytic vs finite-difference dU/dq: max |diff| = {max_mismatch:.2e} N*m)")

    if carry_mass:
        # tau = J^T F  for the carried weight only -> must equal the load's dU/dq part
        F = (0.0, 0.0, carry_mass * g)      # actuators push the load UP
        tau_jtf = lm.joint_torques_from_foot_force(spec, F, q, SOLE)
        tau_noload = lm.gravity_joint_torques(spec, q)
        tau_loadpart = {j: tau[j] - tau_noload[j] for j in tau}
        mm = max(abs(tau_jtf[j] - tau_loadpart[j]) for j in tau)
        print("  carried-load torque via J^T F vs via dU/dq: "
              f"max |diff| = {mm:.2e} N*m")

    peak = max(tau.items(), key=lambda kv: abs(kv[1]))
    print(f"  peak |hold torque|: {peak[1]:+.4f} N*m at {peak[0]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--pose", help="single reference pose (default: all)")
    ap.add_argument("--carry-fraction", type=float, default=0.0,
                    help="fraction of analysis.provisional_total_robot_mass to "
                         "place at the foot as body-weight support")
    args = ap.parse_args(argv)

    spec = lm.load_spec(args.config)
    poses = lm.reference_poses(spec)
    total_robot = float((spec.get("analysis", {}) or {}).get("provisional_total_robot_mass", 0.0))
    carry_mass = args.carry_fraction * total_robot

    print("Gravitational joint torques  (base = pelvis; masses PROVISIONAL)")
    print(f"gravity = {lm.analysis_gravity(spec)} m/s^2   "
          f"leg mass = {lm.total_mass(spec):.4f} kg")
    if args.carry_fraction:
        print(f"body-weight support: {args.carry_fraction:.2f} x {total_robot} kg "
              f"(PROVISIONAL total) = {carry_mass:.3f} kg at the foot")

    names = [args.pose] if args.pose else list(poses)
    for name in names:
        q = poses[name]
        report_pose(spec, name, q, carry_mass)
        if "l_hip_pitch" in q:
            analytic_single_link_check(spec, name, q)

    print("\nnote: torques are 'hold against gravity'. Sign follows the joint's")
    print("positive_rotation convention. Not a servo spec -- no servo is chosen yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
