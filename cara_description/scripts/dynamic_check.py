#!/usr/bin/env python3
"""Dynamic plausibility check for the single left leg under gravity + PD control.

Loads the DYNAMIC MJCF (generate_mjcf.py --dynamic: gravity on, PD <position>
actuators, foot <-> ground contact, pelvis welded to the world) and, for every
reference pose, commands the PD servos to that pose, lets it settle, then
reports:

  settle    max |achieved angle - commanded angle|            [rad]
  jitter    max |joint velocity| over the last second         [rad/s]
  peak tau  largest |actuator torque| (and whether it hit forcerange)
  vs grav   |actuator tau - analytic gravity hold torque|  (airborne poses)
  contact   # foot-floor contacts, min gap (<0 = penetration), total normal force
  FK err    max |MuJoCo body position - leg_model.forward_kinematics|

The question: does this leg behave physically plausibly -- settle without
jitter, hold poses with sane torques, and contact the floor without punching
through it?  Masses / gains are PROVISIONAL (config/left_leg.yaml).

Requires `mujoco` (which also brings numpy). Prints SKIPPED and exits 0 if
mujoco is unavailable.

Usage:
    python3 dynamic_check.py [path/to/left_leg.yaml]
    python3 dynamic_check.py --settle 8 --verbose
"""

from __future__ import annotations

import argparse
import os
import sys

import leg_model as lm

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MJCF = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf", "cara_left_leg_dynamic.xml"))
SOLE = "l_foot_sole_center"

# plausibility thresholds
SETTLE_TOL = 0.05      # rad, airborne pose-tracking error
JITTER_TOL = 0.05      # rad/s, residual joint speed => "settled, no jitter"
PENETRATION_TOL = -3e-3  # m, contact gap floor (more negative = bad)
FK_TOL = 1e-4          # m, MuJoCo vs analytic FK
GRAV_REL_TOL = 0.15    # airborne: |tau - tau_gravity| / max(|tau_gravity|, 0.02)


def _classify(contacts_n, saturated, settle_err, jitter, penetration, fk_err):
    notes = []
    fail = False
    if jitter > JITTER_TOL:
        notes.append("NOT SETTLED"); fail = True
    if penetration < PENETRATION_TOL:
        notes.append(f"PENETRATION {penetration*1e3:.1f}mm"); fail = True
    if fk_err > FK_TOL:
        notes.append(f"FK MISMATCH {fk_err:.1e}m"); fail = True
    if settle_err > SETTLE_TOL:
        if contacts_n > 0:
            notes.append("contact-limited (expected)")
        elif saturated:
            notes.append("torque-saturated (servo too weak)")
        else:
            notes.append(f"POOR TRACKING {settle_err:.3f}rad"); fail = True
    return ("FAIL" if fail else "PASS"), notes


def run(config, settle_t, verbose):
    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    import generate_mjcf

    spec = lm.load_spec(config)
    xml = generate_mjcf.build_mjcf(spec, dynamic=True)
    if os.path.exists(DEFAULT_MJCF):
        with open(DEFAULT_MJCF, encoding="utf-8") as fh:
            if fh.read() != xml:
                print(f"WARNING: {DEFAULT_MJCF} is stale -- run "
                      "`generate_mjcf.py --dynamic` (using a fresh render here)\n")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    n_steps = int(settle_t / dt)
    n_jitter = int(1.0 / dt)

    jnames = lm.joint_names(spec)
    physical = list(lm.link_inertials(spec))
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    foot_gid, floor_gid = gid("l_foot_collision"), gid("floor")
    leg_mass = sum(li.mass for n, li in lm.link_inertials(spec).items() if n != "pelvis")
    g = lm.analysis_gravity(spec)

    print(f"Dynamic plausibility check  ({model.nu} PD servos, g = {g} m/s^2, "
          f"settle {settle_t}s)")
    print(f"provisional leg mass below the pelvis: {leg_mass:.3f} kg  "
          f"(=> {leg_mass * g:.2f} N if fully floor-supported)\n")
    hdr = (f"  {'pose':<18} {'verdict':<6} {'settle':>7} {'jitter':>7} "
           f"{'peak|tau|':>10} {'vsGrav':>7} {'contact':>9} {'gap mm':>7} {'Fn N':>7} {'FKerr':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    any_fail = False
    for pose_name, cfg in lm.reference_poses(spec).items():
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, pose_name)
        mujoco.mj_resetDataKeyframe(model, data, kid)
        jitter = 0.0
        for step in range(n_steps):
            mujoco.mj_step(model, data)
            if step >= n_steps - n_jitter:
                jitter = max(jitter, float(np.abs(data.qvel).max()))
        mujoco.mj_forward(model, data)

        achieved = {j: float(data.qpos[int(model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])]) for j in jnames}
        target = cfg
        settle_err = max(abs(achieved[j] - target.get(j, 0.0)) for j in jnames)

        tau = {j: float(data.actuator_force[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)]) for j in jnames}
        frc = {model.actuator(j).name: float(model.actuator_forcerange[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)][1]) for j in jnames}
        peak_j = max(jnames, key=lambda j: abs(tau[j]))
        saturated = any(abs(tau[j]) >= frc[j] - 1e-6 for j in jnames)

        # foot <-> floor contact
        gaps, fn_total = [], 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if {c.geom1, c.geom2} == {foot_gid, floor_gid}:
                gaps.append(float(c.dist))
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f6)
                fn_total += float(f6[0])
        n_contact = len(gaps)
        gap = min(gaps) if gaps else float("nan")

        # FK consistency at the achieved configuration
        tf = lm.forward_kinematics(spec, achieved)
        fk_err = 0.0
        for name in physical:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            fk_err = max(fk_err, max(abs(a - b) for a, b in zip(data.xpos[bid], tf[name][1])))
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SOLE)
        fk_err = max(fk_err, max(abs(a - b) for a, b in
                                 zip(data.site_xpos[sid], lm.frame_world_position(spec, tf, SOLE))))

        # torque vs analytic gravity hold torque (only meaningful airborne)
        grav = lm.gravity_joint_torques(spec, achieved)
        if n_contact == 0:
            vs_grav = max(abs(tau[j] - grav[j]) for j in jnames)
            vs_grav_str = f"{vs_grav:.4f}"
        else:
            vs_grav = 0.0
            vs_grav_str = "  --"

        verdict, notes = _classify(n_contact, saturated, settle_err, jitter, gap, fk_err)
        if verdict == "FAIL":
            any_fail = True
        gap_str = f"{gap*1e3:+.2f}" if n_contact else "  --"
        fn_str = f"{fn_total:.2f}" if n_contact else "  --"
        print(f"  {pose_name:<18} {verdict:<6} {settle_err:>7.4f} {jitter:>7.4f} "
              f"{abs(tau[peak_j]):>7.3f}@{peak_j.split('_')[-1]:<3} {vs_grav_str:>7} "
              f"{n_contact:>9d} {gap_str:>7} {fn_str:>7} {fk_err:>8.1e}"
              + (f"   {'; '.join(notes)}" if notes else ""))
        if verbose:
            print(f"      target   {[round(target.get(j,0.0),3) for j in jnames]}")
            print(f"      achieved {[round(achieved[j],3) for j in jnames]}")
            print(f"      tau      {[round(tau[j],3) for j in jnames]}")
            if n_contact == 0:
                print(f"      grav tau {[round(grav[j],3) for j in jnames]}")

    print("\n" + "=" * 60)
    print("Notes:")
    print("  * Airborne poses: actuator torque matches the analytic gravity hold")
    print("    torque (gravity_torques.py) once settled -- 'vsGrav' is that gap.")
    print("  * Fixed-pelvis rig: with a straight leg the hinge constraints carry")
    print("    the weight up to the welded pelvis, so the floor barely loads")
    print("    (Fn ~ 0 at 'zero'). Real foot loading needs a floating/sliding")
    print("    pelvis -- that is the single-leg STANCE phase, not this one.")
    print("  * To load the servos now: raise analysis.ground.z_offset (e.g. 0.04)")
    print("    so the crouch poses plant on the floor, and re-run.")
    print("RESULT:", "FAIL" if any_fail else "PASS")
    return 1 if any_fail else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--settle", type=float, default=5.0, help="settle time per pose [s]")
    ap.add_argument("--verbose", action="store_true", help="print per-joint vectors")
    args = ap.parse_args(argv)
    return run(args.config, args.settle, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
