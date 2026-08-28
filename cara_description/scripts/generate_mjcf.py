#!/usr/bin/env python3
"""Generate mjcf/cara_left_leg.xml from config/left_leg.yaml.

The YAML is the single source of truth -- the SAME file that feeds
generate_urdf.py:

                    +--> urdf/cara_left_leg.urdf
    left_leg.yaml --+
                    +--> mjcf/cara_left_leg.xml

Never hand-tune the MJCF: edit the YAML and regenerate, or URDF and MJCF drift.

What is preserved from the YAML, unchanged:
  * coordinate convention  +X forward, +Y left, +Z up, right-handed
  * every joint origin, axis and position limit
  * the coincident hip / ankle abstraction

How the coincident abstraction is represented in MJCF:
  MuJoCo requires positive mass on any body that carries a DOF, so the three
  virtual coupling links are NOT emitted as bodies (that would force fake
  epsilon inertia). Instead the coincident joints are stacked on the physical
  body downstream -- l_hip_yaw/roll/pitch as three <joint> on l_thigh,
  l_ankle_pitch/roll as two <joint> on l_foot. This is the idiomatic MuJoCo
  form and is mathematically identical to the URDF chain (validated by
  scripts/validate_mjcf.py).

Other choices (kinematics-only model, no dynamics/RL yet):
  * option gravity = 0 0 0  -- so the viewer shows the pose you set, the leg
    does not sag. Flip this on when dynamics work begins.
  * all link geoms are primitive and NON-colliding (contype=conaffinity=0),
    for inspection only. Shapes/sizes come straight from dynamics.links.
  * a visual ground plane at the zero-pose sole height, also non-colliding.
  * no actuators.

Usage:
    python3 generate_mjcf.py                 # write ../mjcf/cara_left_leg.xml
    python3 generate_mjcf.py --stdout
    python3 generate_mjcf.py --check         # exit 1 if the file is stale
    python3 generate_mjcf.py -o other.xml
"""

from __future__ import annotations

import argparse
import os
import sys

import leg_model as lm

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf", "cara_left_leg.xml"))

LINK_RGBA = "0.75 0.78 0.82 1"
FLOOR_RGBA = "0.85 0.85 0.88 1"
JOINT_SITE_RGBA = "0.9 0.5 0.2 1"
SOLE_SITE_RGBA = "0.2 0.7 0.4 1"
ZERO_TOL = 1e-12


def _fmt(x: float) -> str:
    if x == 0.0:            # normalise -0.0 -> 0
        return "0"
    return f"{x:.9g}"


def _xyz(v) -> str:
    return " ".join(_fmt(c) for c in v)


def _geom_xml(shape: tuple, com) -> str:
    kind, dims = shape
    if kind == "box":
        half = [d / 2.0 for d in dims]
        return (f'<geom type="box" size="{_xyz(half)}" pos="{_xyz(com)}" '
                f'rgba="{LINK_RGBA}"/>')
    if kind == "cylinder":
        radius, length = dims
        return (f'<geom type="cylinder" size="{_fmt(radius)} {_fmt(length / 2.0)}" '
                f'pos="{_xyz(com)}" rgba="{LINK_RGBA}"/>')
    raise ValueError(f"unhandled shape {shape!r}")


def _stacks(spec: dict):
    """Group the joint chain into (physical_child, body_pos, [(joint, anchor)]).

    Consecutive joints that share a physical child body (the coincident hip /
    ankle joints) are stacked on that body. For a stack with zero-config
    origins o_0..o_n the exact equivalence to the URDF chain
        Trans(o_0) Rot(0) Trans(o_1) Rot(1) ... Trans(o_n) Rot(n)
    is reproduced in MJCF by
        body pos   = o_0 + o_1 + ... + o_n
        joint i anchor (<joint pos=...>) = -(o_{i+1} + ... + o_n)
    With the current coincident abstraction o_1..o_n are all zero, so every
    anchor is "0 0 0"; the formula also handles real inter-axis offsets if they
    are added later.
    """
    inertials = lm.link_inertials(spec)
    physical = set(inertials)  # pelvis, l_thigh, l_shin, l_foot
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


def build_mjcf(spec: dict) -> str:
    inertials = lm.link_inertials(spec)
    stacks = _stacks(spec)
    model_name = spec["meta"]["name"]
    syms = lm.resolve_symbols(spec)
    g = lm.analysis_gravity(spec)

    # zero-pose sole height, for the visual floor
    sole0 = lm.foot_position(spec, {})
    floor_z = sole0[2]

    child_stack = {child: (pos, joints) for child, pos, joints in stacks}
    base = spec["frame_conventions"]["base_frame"]

    # child link -> its stack's parent physical link (chain order)
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

    w(f'<mujoco model="{model_name}">')
    w("  <!-- ================================================================")
    w("       GENERATED FILE. Do not edit by hand.")
    w("       source : cara_description/config/left_leg.yaml")
    w("       tool   : cara_description/scripts/generate_mjcf.py")
    w("       regen  : python3 scripts/generate_mjcf.py")
    w("")
    w("       Same source of truth as urdf/cara_left_leg.urdf.")
    w("       Kinematics-only model: gravity is off, geoms do not collide,")
    w("       there are no actuators. Coincident hip/ankle joints are stacked")
    w("       on the physical body downstream (see the script docstring).")
    w("       Frame convention: +X forward, +Y left, +Z up, right-handed.")
    w("       ============================================================= -->")
    w('  <compiler angle="radian" autolimits="true"/>')
    w(f'  <option gravity="0 0 0" timestep="0.002"/>   <!-- real g = {_fmt(g)}; off for kinematic inspection -->')
    w("")
    w("  <visual>")
    w('    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>')
    w('    <rgba haze="0.15 0.25 0.35 1"/>')
    w('    <global azimuth="130" elevation="-20"/>')
    w("  </visual>")
    w("")
    w("  <default>")
    w('    <geom contype="0" conaffinity="0" group="2"/>')
    w('    <site type="sphere" size="0.006" group="3"/>')
    w("  </default>")
    w("")
    w("  <worldbody>")
    w('    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.5 0.5 0.5"/>')
    w(f'    <geom name="floor" type="plane" size="0.6 0.6 0.02" pos="0 0 {_fmt(floor_z)}" '
      f'rgba="{FLOOR_RGBA}" group="1"/>')
    w("")

    def emit_body(name: str, indent: str, parent_pos):
        li = inertials[name]
        if name in child_stack:
            body_pos, joints = child_stack[name]
        else:
            body_pos, joints = parent_pos, []
        w(f'{indent}<body name="{name}" pos="{_xyz(body_pos)}">')
        for jm, anchor in joints:
            w(f'{indent}  <joint name="{jm.name}" type="hinge" pos="{_xyz(anchor)}" '
              f'axis="{_xyz(jm.axis)}" range="{_fmt(jm.lower)} {_fmt(jm.upper)}"/>')
        w(f'{indent}  <inertial pos="{_xyz(li.com)}" mass="{_fmt(li.mass)}" '
          f'diaginertia="{_xyz(li.inertia_diag)}"/>')
        w(f'{indent}  {_geom_xml(li.shape, li.com)}')
        # a small site at this body's own frame origin (= the joint centre it carries)
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
    w("</mujoco>")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None, help="path to left_leg.yaml")
    ap.add_argument("-o", "--output", default=DEFAULT_OUT, help="output MJCF path")
    ap.add_argument("--stdout", action="store_true", help="print to stdout, do not write")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the on-disk MJCF differs from a fresh render")
    args = ap.parse_args(argv)

    spec = lm.load_spec(args.config)
    text = build_mjcf(spec)

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.check:
        if not os.path.exists(args.output):
            print(f"STALE: {args.output} does not exist")
            return 1
        with open(args.output, "r", encoding="utf-8") as fh:
            current = fh.read()
        if current != text:
            print(f"STALE: {args.output} is out of date -- run generate_mjcf.py")
            return 1
        print(f"OK: {args.output} matches config")
        return 0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {args.output}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
