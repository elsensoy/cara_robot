#!/usr/bin/env python3
"""Generate a MuJoCo MJCF model from config/left_leg.yaml.

The YAML is the single source of truth -- the SAME file that feeds
generate_urdf.py:

                    +--> urdf/cara_left_leg.urdf
    left_leg.yaml --+--> mjcf/cara_left_leg.xml           (kinematic, default)
                    +--> mjcf/cara_left_leg_dynamic.xml   (--dynamic)

Never hand-tune the MJCF: edit the YAML and regenerate, or the descriptions
drift apart.

Preserved from the YAML, unchanged, in BOTH modes:
  * coordinate convention  +X forward, +Y left, +Z up, right-handed
  * every joint origin, axis and position limit
  * the coincident hip / ankle abstraction

Coincident abstraction in MJCF:
  MuJoCo requires positive mass on any body that carries a DOF, so the three
  virtual coupling links are NOT bodies (no fake epsilon inertia). The
  coincident joints are stacked on the physical body downstream --
  l_hip_yaw/roll/pitch as three <joint> on l_thigh, l_ankle_pitch/roll as two
  on l_foot. Mathematically identical to the URDF chain (validate_mjcf.py).

KINEMATIC mode (default):
  * gravity off, all geoms non-colliding, no actuators -- pure pose inspection.

DYNAMIC mode (--dynamic):
  * gravity on (analysis.gravity)
  * PD <position> actuators on all 6 joints (dynamics.actuators.control gains,
    forcerange = the provisional effort limit)
  * the foot collides with a ground plane at the zero-pose sole height
    (analysis.ground.friction); all other geoms stay non-colliding
  * <keyframe> for every analysis.reference_pose (qpos + matching ctrl target)

Usage:
    python3 generate_mjcf.py                       # mjcf/cara_left_leg.xml
    python3 generate_mjcf.py --dynamic             # mjcf/cara_left_leg_dynamic.xml
    python3 generate_mjcf.py [--dynamic] --check
    python3 generate_mjcf.py [--dynamic] --stdout
    python3 generate_mjcf.py --dynamic -o other.xml
"""

from __future__ import annotations

import argparse
import os
import sys

import leg_model as lm

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf"))
DEFAULT_OUT = os.path.join(_MJCF_DIR, "cara_left_leg.xml")
DEFAULT_OUT_DYNAMIC = os.path.join(_MJCF_DIR, "cara_left_leg_dynamic.xml")

LINK_RGBA = "0.75 0.78 0.82 1"
FLOOR_RGBA = "0.85 0.85 0.88 1"
JOINT_SITE_RGBA = "0.9 0.5 0.2 1"
SOLE_SITE_RGBA = "0.2 0.7 0.4 1"
FOOT_COLLISION_RGBA = "0.9 0.6 0.3 0.35"


def _fmt(x: float) -> str:
    if x == 0.0:            # normalise -0.0 -> 0
        return "0"
    return f"{x:.9g}"


def _xyz(v) -> str:
    return " ".join(_fmt(c) for c in v)


def _visual_geom(shape: tuple, com) -> str:
    kind, dims = shape
    if kind == "box":
        half = [d / 2.0 for d in dims]
        return f'<geom type="box" size="{_xyz(half)}" pos="{_xyz(com)}" rgba="{LINK_RGBA}"/>'
    if kind == "cylinder":
        radius, length = dims
        return (f'<geom type="cylinder" size="{_fmt(radius)} {_fmt(length / 2.0)}" '
                f'pos="{_xyz(com)}" rgba="{LINK_RGBA}"/>')
    raise ValueError(f"unhandled shape {shape!r}")


def _foot_collision_geom(shape: tuple, com, friction) -> str:
    kind, dims = shape
    if kind != "box":
        raise ValueError("expected a box foot for the collision geom")
    half = [d / 2.0 for d in dims]
    return (f'<geom name="l_foot_collision" type="box" size="{_xyz(half)}" pos="{_xyz(com)}" '
            f'contype="1" conaffinity="1" condim="3" friction="{_xyz(friction)}" '
            f'rgba="{FOOT_COLLISION_RGBA}" group="4"/>')


def _stacks(spec: dict):
    """Group the joint chain into (physical_child, body_pos, [(joint, anchor)]).

    Coincident joints sharing a physical child body are stacked on that body.
    For a stack with zero-config origins o_0..o_n the exact equivalence to the
    URDF chain  Trans(o_0) Rot(0) ... Trans(o_n) Rot(n)  is reproduced by
        body pos              = o_0 + o_1 + ... + o_n
        joint i <joint pos>   = -(o_{i+1} + ... + o_n)
    With the current coincident abstraction o_1..o_n are all zero (every anchor
    "0 0 0"); the formula also covers real inter-axis offsets added later.
    """
    inertials = lm.link_inertials(spec)
    physical = set(inertials)
    stacks = []
    pending: list[lm.JointModel] = []
    for jm in lm.build_chain(spec):
        pending.append(jm)
        if jm.child in physical:
            origins = [j.origin for j in pending]
            body_pos = (0.0, 0.0, 0.0)
            for o in origins:
                body_pos = lm.vec_add(body_pos, o)
            joints = []
            for i, j in enumerate(pending):
                tail = (0.0, 0.0, 0.0)
                for o in origins[i + 1:]:
                    tail = lm.vec_add(tail, o)
                joints.append((j, (-tail[0], -tail[1], -tail[2])))
            stacks.append((jm.child, body_pos, joints))
            pending = []
    if pending:
        raise ValueError(f"chain ends on virtual links: {[j.name for j in pending]}")
    return stacks


def build_mjcf(spec: dict, dynamic: bool = False) -> str:
    inertials = lm.link_inertials(spec)
    stacks = _stacks(spec)
    chain = lm.build_chain(spec)
    model_name = spec["meta"]["name"] + ("_dynamic" if dynamic else "")
    syms = lm.resolve_symbols(spec)
    g = lm.analysis_gravity(spec)
    foot_link = stacks[-1][0]
    ground = lm.ground_params(spec)
    friction = ground["friction"]
    control = lm.actuator_control(spec)

    floor_z = lm.foot_position(spec, {})[2] + ground["z_offset"]   # zero-pose sole height (+ offset)

    child_stack = {child: (pos, joints) for child, pos, joints in stacks}
    base = spec["frame_conventions"]["base_frame"]

    parent_of = {}
    prev_physical = base
    for child, _pos, _joints in stacks:
        parent_of[child] = prev_physical
        prev_physical = child
    children_of: dict[str, list[str]] = {}
    for child, parent in parent_of.items():
        children_of.setdefault(parent, []).append(child)

    out: list[str] = []
    w = out.append
    mode = "dynamic" if dynamic else "kinematic"

    w(f'<mujoco model="{model_name}">')
    w("  <!-- ================================================================")
    w("       GENERATED FILE. Do not edit by hand.")
    w("       source : cara_description/config/left_leg.yaml")
    w("       tool   : cara_description/scripts/generate_mjcf.py"
      + ("  --dynamic" if dynamic else ""))
    w(f"       regen  : python3 scripts/generate_mjcf.py{' --dynamic' if dynamic else ''}")
    w("")
    w(f"       mode: {mode.upper()}.  Same source of truth as the URDF.")
    if dynamic:
        w("       gravity ON; PD <position> actuators; foot <-> ground contact.")
        w("       Pelvis is welded to the world (fixed-base single-leg test rig).")
    else:
        w("       gravity OFF; geoms non-colliding; no actuators (pose inspection).")
    w("       Coincident hip/ankle joints are stacked on the physical body")
    w("       downstream (see the script docstring).")
    w("       Frame convention: +X forward, +Y left, +Z up, right-handed.")
    w("       ============================================================= -->")
    w('  <compiler angle="radian" autolimits="true"/>')
    if dynamic:
        w(f'  <option gravity="0 0 {_fmt(-g)}" timestep="0.002" integrator="implicitfast"/>')
    else:
        w(f'  <option gravity="0 0 0" timestep="0.002"/>   <!-- real g = {_fmt(g)}; off for inspection -->')
    w("")
    w("  <visual>")
    w('    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>')
    w('    <rgba haze="0.15 0.25 0.35 1"/>')
    w('    <global azimuth="130" elevation="-20"/>')
    w("  </visual>")
    w("")
    w("  <default>")
    w('    <geom contype="0" conaffinity="0" group="2"/>   <!-- link geoms: visual only -->')
    w('    <site type="sphere" size="0.006" group="3"/>')
    w("  </default>")
    w("")
    w("  <worldbody>")
    w('    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.5 0.5 0.5"/>')
    if dynamic:
        w(f'    <geom name="floor" type="plane" size="0.6 0.6 0.02" pos="0 0 {_fmt(floor_z)}" '
          f'contype="1" conaffinity="1" condim="3" friction="{_xyz(friction)}" '
          f'rgba="{FLOOR_RGBA}" group="1"/>')
    else:
        w(f'    <geom name="floor" type="plane" size="0.6 0.6 0.02" pos="0 0 {_fmt(floor_z)}" '
          f'rgba="{FLOOR_RGBA}" group="1"/>')
    w("")

    def emit_body(name: str, indent: str, parent_pos):
        li = inertials[name]
        body_pos, joints = child_stack.get(name, (parent_pos, []))
        w(f'{indent}<body name="{name}" pos="{_xyz(body_pos)}">')
        for jm, anchor in joints:
            w(f'{indent}  <joint name="{jm.name}" type="hinge" pos="{_xyz(anchor)}" '
              f'axis="{_xyz(jm.axis)}" range="{_fmt(jm.lower)} {_fmt(jm.upper)}"/>')
        w(f'{indent}  <inertial pos="{_xyz(li.com)}" mass="{_fmt(li.mass)}" '
          f'diaginertia="{_xyz(li.inertia_diag)}"/>')
        w(f'{indent}  {_visual_geom(li.shape, li.com)}')
        if dynamic and name == foot_link:
            w(f'{indent}  {_foot_collision_geom(li.shape, li.com, friction)}')
        w(f'{indent}  <site name="{name}_frame" pos="0 0 0" rgba="{JOINT_SITE_RGBA}"/>')
        for foi in spec.get("frames_of_interest", []) or []:
            if foi["link"] == name:
                p = lm.resolve_vec3(foi["xyz_expr"], syms)
                w(f'{indent}  <site name="{foi["name"]}" pos="{_xyz(p)}" rgba="{SOLE_SITE_RGBA}"/>')
        for ch in children_of.get(name, []):
            emit_body(ch, indent + "  ", None)
        w(f'{indent}</body>')

    emit_body(base, "    ", (0.0, 0.0, 0.0))
    w("  </worldbody>")
    w("")

    # ---- keyframes (reference poses) ------------------------------------
    poses = lm.reference_poses(spec)
    if poses:
        w("  <keyframe>")
        for pname, cfg in poses.items():
            qp = lm.pose_qpos(spec, cfg)
            line = f'    <key name="{pname}" qpos="{_xyz(qp)}"'
            if dynamic:
                line += f' ctrl="{_xyz(qp)}"'
            w(line + "/>")
        w("  </keyframe>")
        w("")

    # ---- actuators (dynamic only) -------------------------------------
    if dynamic:
        w("  <actuator>")
        for jm in chain:
            c = control[jm.name]
            w(f'    <position name="{jm.name}" joint="{jm.name}" '
              f'kp="{_fmt(c["kp"])}" dampratio="{_fmt(c["dampratio"])}" '
              f'ctrlrange="{_fmt(jm.lower)} {_fmt(jm.upper)}" '
              f'forcerange="{_fmt(-jm.effort)} {_fmt(jm.effort)}"/>')
        w("  </actuator>")
        w("")

    w("</mujoco>")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None, help="path to left_leg.yaml")
    ap.add_argument("-o", "--output", default=None, help="output MJCF path")
    ap.add_argument("--dynamic", action="store_true",
                    help="emit the gravity + actuators + contact model")
    ap.add_argument("--stdout", action="store_true", help="print to stdout, do not write")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the on-disk MJCF differs from a fresh render")
    args = ap.parse_args(argv)

    out = args.output or (DEFAULT_OUT_DYNAMIC if args.dynamic else DEFAULT_OUT)
    spec = lm.load_spec(args.config)
    text = build_mjcf(spec, dynamic=args.dynamic)

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.check:
        if not os.path.exists(out):
            print(f"STALE: {out} does not exist")
            return 1
        with open(out, "r", encoding="utf-8") as fh:
            if fh.read() != text:
                print(f"STALE: {out} is out of date -- run generate_mjcf.py"
                      f"{' --dynamic' if args.dynamic else ''}")
                return 1
        print(f"OK: {out} matches config")
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {out}  ({len(text.splitlines())} lines, {'dynamic' if args.dynamic else 'kinematic'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
