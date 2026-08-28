#!/usr/bin/env python3
"""Load a generated MJCF programmatically and open the MuJoCo viewer.

The reproducible entry point -- the model is built from the generated MJCF
(which comes from config/left_leg.yaml), never hand-loaded through the
viewer's File > Open. Use File > Open only for a throwaway glance.

    python3 scripts/view_mujoco.py                       # kinematic model
    python3 scripts/view_mujoco.py --regen               # regenerate first
    python3 scripts/view_mujoco.py --pose deep_crouch
    python3 scripts/view_mujoco.py --dynamic --regen     # gravity + PD servos
    python3 scripts/view_mujoco.py --dynamic --pose knee_lift

With --dynamic the PD servos are commanded to --pose (or 'zero'); without it
the model is frozen kinematics and you drag joints by hand.

Requires `mujoco` (pip install mujoco).
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf"))
KINEMATIC_MJCF = os.path.join(_MJCF_DIR, "cara_left_leg.xml")
DYNAMIC_MJCF = os.path.join(_MJCF_DIR, "cara_left_leg_dynamic.xml")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dynamic", action="store_true",
                    help="use the gravity + actuators + contact model")
    ap.add_argument("--mjcf", default=None, help="explicit MJCF path (overrides --dynamic default)")
    ap.add_argument("--config", default=None, help="YAML to (re)generate from with --regen")
    ap.add_argument("--regen", action="store_true", help="regenerate the MJCF from YAML first")
    ap.add_argument("--pose", help="reference-pose name for the initial qpos (and PD target)")
    args = ap.parse_args(argv)

    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("error: mujoco is not installed  (pip install mujoco)", file=sys.stderr)
        return 2

    import leg_model as lm

    mjcf = args.mjcf or (DYNAMIC_MJCF if args.dynamic else KINEMATIC_MJCF)

    if args.regen:
        import generate_mjcf
        gen_args = ([args.config] if args.config else []) + ["-o", mjcf]
        if args.dynamic:
            gen_args.append("--dynamic")
        rc = generate_mjcf.main(gen_args)
        if rc != 0:
            return rc

    if not os.path.exists(mjcf):
        print(f"error: {mjcf} does not exist -- run generate_mjcf.py"
              f"{' --dynamic' if args.dynamic else ''} (or pass --regen)", file=sys.stderr)
        return 2

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)

    pose = args.pose or ("zero" if args.dynamic else None)
    if pose:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, pose)
        if kid < 0:
            print(f"error: no reference pose / keyframe named {pose!r}", file=sys.stderr)
            return 2
        mujoco.mj_resetDataKeyframe(model, data, kid)
        mujoco.mj_forward(model, data)
        print(f"initial pose '{pose}'"
              + ("  (PD servos commanded to it)" if args.dynamic else ""))

    print(f"loading {mjcf}  ({model.nbody - 1} bodies, {model.njnt} joints, "
          f"{model.nu} actuators)")
    if args.dynamic:
        print("dynamic model: gravity on, PD position servos hold the commanded pose.")
    else:
        print("kinematic model: gravity off; drag joints in the viewer.")
    mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
