#!/usr/bin/env python3
"""Generate urdf/cara_left_leg.urdf from config/left_leg.yaml.

The YAML is the single source of truth. This script is the only thing that
writes the URDF -- never hand-edit the generated file.

What is authoritative vs placeholder in the output:
  * AUTHORITATIVE : the kinematics -- joint origins, axes, limits, tree.
  * PLACEHOLDER   : link mass / COM / inertia come from the `dynamics:` block
    (provisional, method-tagged); joint effort / velocity come from
    `dynamics.actuators` and are TBD.

Physical links get <inertial>/<visual>/<collision>. The virtual coupling
links (l_hip_yaw_link, l_hip_roll_link, l_ankle_link) are emitted as BARE
frames -- no inertial, no visual, no collision -- because they are massless
mathematical abstractions, not pieces of plastic.

Usage:
    python3 generate_urdf.py                 # write ../urdf/cara_left_leg.urdf
    python3 generate_urdf.py --stdout        # print instead of writing
    python3 generate_urdf.py --check         # exit 1 if the file is stale
    python3 generate_urdf.py -o other.urdf   # write somewhere else
"""

from __future__ import annotations

import argparse
import os
import sys

import leg_model as lm

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, os.pardir, "urdf", "cara_left_leg.urdf"))


def _fmt(x: float) -> str:
    return f"{x:.9g}"


def _xyz(v) -> str:
    return " ".join(_fmt(c) for c in v)


def _comment(indent: str, text: str) -> str:
    """An XML comment line. Collapses '--' in the body ('--' is illegal in comments)."""
    body = text
    while "--" in body:
        body = body.replace("--", "-")
    return f"{indent}<!-- {body.strip()} -->"


def _geometry_xml(shape: tuple) -> str:
    kind, dims = shape
    if kind == "box":
        return f'<box size="{_xyz(dims)}"/>'
    if kind == "cylinder":
        radius, length = dims
        return f'<cylinder radius="{_fmt(radius)}" length="{_fmt(length)}"/>'
    raise ValueError(f"unhandled shape {shape!r}")


def build_urdf(spec: dict) -> str:
    inertials = lm.link_inertials(spec)          # physical links only
    chain = lm.build_chain(spec)
    robot_name = spec["meta"]["name"]

    out: list[str] = []
    w = out.append

    w('<?xml version="1.0"?>')
    w("<!-- ===================================================================")
    w("     GENERATED FILE. Do not edit by hand.")
    w("     source : cara_description/config/left_leg.yaml")
    w("     tool   : cara_description/scripts/generate_urdf.py")
    w("     regen  : python3 scripts/generate_urdf.py")
    w("")
    w("     AUTHORITATIVE : joint origins / axes / limits / tree (kinematics).")
    w("     PLACEHOLDER   : link mass / COM / inertia (dynamics: block,")
    w("                     method-tagged); joint effort / velocity are TBD.")
    w("     Virtual coupling links are bare frames on purpose (massless).")
    w("     Frame convention: +X forward, +Y left, +Z up, right-handed.")
    w("     ================================================================ -->")
    w(f'<robot name="{robot_name}">')
    w("")

    # ---- links -------------------------------------------------------------
    for link in spec["links"]:
        name = link["name"]
        role = link.get("role", "?")

        if name not in inertials:
            w(f'  <link name="{name}">')
            w(_comment("    ", f"role: {role}. Virtual coincident-axis coupling link: "
                               f"massless abstraction, intentionally no inertial/visual/collision."))
            w("  </link>")
            w("")
            continue

        li = inertials[name]
        ixx, iyy, izz = li.inertia_diag
        w(f'  <link name="{name}">')
        w(_comment("    ", f"role: {role}. Mass/COM/inertia PROVISIONAL "
                           f"(method: {li.method}). TODO: replace with CAD."))
        w("    <inertial>")
        w(f'      <origin xyz="{_xyz(li.com)}" rpy="0 0 0"/>')
        w(f'      <mass value="{_fmt(li.mass)}"/>')
        w(f'      <inertia ixx="{_fmt(ixx)}" ixy="0" ixz="0" '
          f'iyy="{_fmt(iyy)}" iyz="0" izz="{_fmt(izz)}"/>')
        w("    </inertial>")
        w("    <visual>")
        w(f'      <origin xyz="{_xyz(li.com)}" rpy="0 0 0"/>')
        w(f"      <geometry>{_geometry_xml(li.shape)}</geometry>")
        w("    </visual>")
        w("    <collision>")
        w(f'      <origin xyz="{_xyz(li.com)}" rpy="0 0 0"/>')
        w(f"      <geometry>{_geometry_xml(li.shape)}</geometry>")
        w("    </collision>")
        w("  </link>")
        w("")

    # ---- joints ----------------------------------------------------------
    for jm in chain:
        w(f'  <joint name="{jm.name}" type="{jm.jtype}">')
        w(_comment("    ", f"purpose: {jm.purpose}"))
        w(_comment("    ", f"+angle: {jm.positive_rotation}"))
        w(f'    <parent link="{jm.parent}"/>')
        w(f'    <child link="{jm.child}"/>')
        w(f'    <origin xyz="{_xyz(jm.origin)}" rpy="0 0 0"/>')
        w(f'    <axis xyz="{_xyz(jm.axis)}"/>')
        w(f'    <limit lower="{_fmt(jm.lower)}" upper="{_fmt(jm.upper)}" '
          f'effort="{_fmt(jm.effort)}" velocity="{_fmt(jm.velocity)}"/>'
          '  <!-- effort/velocity: TBD placeholder -->')
        w('    <dynamics damping="0.0" friction="0.0"/>  <!-- TODO: measured -->')
        w("  </joint>")
        w("")

    w("</robot>")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default=None, help="path to a description YAML")
    ap.add_argument("-o", "--output", default=None, help="output URDF path")
    ap.add_argument("--stdout", action="store_true", help="print to stdout, do not write")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the on-disk URDF differs from a fresh render")
    args = ap.parse_args(argv)

    spec = lm.load_spec(args.config)
    text = build_urdf(spec)
    args.output = args.output or os.path.normpath(
        os.path.join(_HERE, os.pardir, "urdf", spec["meta"]["name"] + ".urdf"))

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
            print(f"STALE: {args.output} is out of date -- run generate_urdf.py")
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
