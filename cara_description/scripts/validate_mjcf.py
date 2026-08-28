#!/usr/bin/env python3
"""Check that YAML -> MJCF -> MuJoCo reproduces the already-validated kinematics.

For every reference pose in config/left_leg.yaml, this:
  1. sets the 6 joint angles in a freshly compiled MuJoCo model,
  2. runs mujoco.mj_kinematics,
  3. compares, in the world frame:
       * every physical body origin   (pelvis, l_thigh, l_shin, l_foot)
       * every physical body orientation (rotation matrix)
       * the l_foot_sole_center site
     against scripts/leg_model.forward_kinematics / frame_world_position.

Passes if every position error < 1e-9 m and every rotation error < 1e-9.

Requires `mujoco` (pip install mujoco). If it is not importable the script
prints SKIPPED and exits 0 so it does not break a checkout without MuJoCo.

Usage:
    python3 validate_mjcf.py [path/to/left_leg.yaml]
"""

from __future__ import annotations

import os
import sys

import leg_model as lm

POS_TOL = 1e-9
ROT_TOL = 1e-9
_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf"))
SOLE = "l_foot_sole_center"


def main(argv=None) -> int:
    args = list(argv or [])
    dynamic = "--dynamic" in args
    args = [a for a in args if a != "--dynamic"]
    config = args[0] if args else None
    try:
        import mujoco
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    import generate_mjcf

    spec = lm.load_spec(config)

    # Validate against a FRESH render of the current YAML; flag a stale file.
    xml = generate_mjcf.build_mjcf(spec, dynamic=dynamic)
    stem = spec["meta"]["name"]
    on_disk = os.path.join(_MJCF_DIR, stem + ("_dynamic.xml" if dynamic else ".xml"))
    print(f"({spec['meta']['name']}, {'dynamic' if dynamic else 'kinematic'} model)")
    if os.path.exists(on_disk):
        with open(on_disk, encoding="utf-8") as fh:
            if fh.read() != xml:
                print(f"WARNING: {on_disk} is stale -- run generate_mjcf.py"
                      f"{' --dynamic' if dynamic else ''} (validating a fresh render anyway)\n")

    import numpy as np

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    base_name = spec["frame_conventions"]["base_frame"]
    physical = list(lm.link_inertials(spec))
    soles = [foi["name"] for foi in spec.get("frames_of_interest", []) or []]
    jnames = lm.joint_names(spec)
    qadr = {j: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])
            for j in jnames}
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    pelvis_bid = bid(base_name)

    poses = lm.reference_poses(spec)
    worst_pos = worst_rot = 0.0
    failures = []

    print(f"YAML -> MJCF -> MuJoCo kinematic reproduction  ({len(poses)} reference poses)")
    print(f"  frames checked (pelvis-relative): {', '.join(physical + soles)}\n")

    for pose_name, cfg in poses.items():
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, pose_name)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(model, data, kid)
        else:
            mujoco.mj_resetData(model, data)
            for j, val in cfg.items():
                data.qpos[qadr[j]] = val
        mujoco.mj_kinematics(model, data)

        tf = lm.forward_kinematics(spec, cfg)
        R_pel = np.array(data.xmat[pelvis_bid]).reshape(3, 3)
        p_pel = np.array(data.xpos[pelvis_bid])
        fk_pel = np.array(tf[base_name][1])
        pose_pos = pose_rot = 0.0

        for name in physical:
            rel_mj = R_pel.T @ (np.array(data.xpos[bid(name)]) - p_pel)
            rel_fk = np.array(tf[name][1]) - fk_pel
            pose_pos = max(pose_pos, float(np.max(np.abs(rel_mj - rel_fk))))
            # orientation: MuJoCo body R in pelvis frame vs analytic R
            rel_R_mj = R_pel.T @ np.array(data.xmat[bid(name)]).reshape(3, 3)
            rel_R_fk = np.array(tf[base_name][0]).T @ np.array(tf[name][0])
            pose_rot = max(pose_rot, float(np.max(np.abs(rel_R_mj - rel_R_fk))))

        for s in soles:
            rel_mj = R_pel.T @ (np.array(data.site_xpos[sid(s)]) - p_pel)
            rel_fk = np.array(lm.frame_world_position(spec, tf, s)) - fk_pel
            pose_pos = max(pose_pos, float(np.max(np.abs(rel_mj - rel_fk))))

        ok = pose_pos < POS_TOL and pose_rot < ROT_TOL
        worst_pos = max(worst_pos, pose_pos)
        worst_rot = max(worst_rot, pose_rot)
        if not ok:
            failures.append(pose_name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {pose_name:<20} "
              f"max |dpos| = {pose_pos:.2e} m   max |dR| = {pose_rot:.2e}")

    print("\n" + "=" * 60)
    print(f"worst position error {worst_pos:.2e} m   worst rotation error {worst_rot:.2e}")
    if failures:
        print(f"RESULT: FAIL  ({len(failures)} pose(s): {', '.join(failures)})")
        return 1
    print(f"RESULT: PASS  (MJCF reproduces leg_model FK to < {POS_TOL:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
