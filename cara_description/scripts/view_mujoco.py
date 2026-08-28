#!/usr/bin/env python3
"""Load the generated MJCF programmatically and open the MuJoCo viewer.

This is the reproducible entry point -- the model is built from
mjcf/cara_left_leg.xml (which is generated from config/left_leg.yaml), never
hand-loaded through the viewer's File > Open. Use File > Open only for a
throwaway glance.

    python3 scripts/view_mujoco.py                 # current mjcf/cara_left_leg.xml
    python3 scripts/view_mujoco.py --regen         # regenerate from YAML first
    python3 scripts/view_mujoco.py --pose deep_crouch
    python3 scripts/view_mujoco.py --config config/variant_A.yaml --regen

Requires `mujoco` (pip install mujoco).
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MJCF = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf", "cara_left_leg.xml"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", default=DEFAULT_MJCF, help="path to the MJCF file")
    ap.add_argument("--config", default=None, help="YAML to (re)generate from with --regen")
    ap.add_argument("--regen", action="store_true", help="regenerate the MJCF from YAML first")
    ap.add_argument("--pose", help="reference-pose name to set as the initial qpos")
    args = ap.parse_args(argv)

    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("error: mujoco is not installed  (pip install mujoco)", file=sys.stderr)
        return 2

    import leg_model as lm

    if args.regen:
        import generate_mjcf
        rc = generate_mjcf.main(([args.config] if args.config else [])
                                + ["-o", args.mjcf])
        if rc != 0:
            return rc

    if not os.path.exists(args.mjcf):
        print(f"error: {args.mjcf} does not exist -- run generate_mjcf.py (or pass --regen)",
              file=sys.stderr)
        return 2

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    if args.pose:
        spec = lm.load_spec(args.config)
        cfg = lm.reference_poses(spec).get(args.pose)
        if cfg is None:
            print(f"error: no reference pose named {args.pose!r}", file=sys.stderr)
            return 2
        for j, val in cfg.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            data.qpos[int(model.jnt_qposadr[jid])] = val
        mujoco.mj_forward(model, data)
        print(f"initial pose '{args.pose}': {cfg}")

    print(f"loading {args.mjcf}  ({model.nbody - 1} bodies, {model.njnt} joints)")
    print("gravity is off in this model (kinematic inspection); drag joints in the viewer.")
    mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
