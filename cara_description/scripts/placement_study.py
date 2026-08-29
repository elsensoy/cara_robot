#!/usr/bin/env python3
"""U3 -- where should Cara's heavy electronics go?

For each named electronics layout (`electronics.layouts` in the YAML, each a
{jetson: <mount>, battery: <mount>} pair) this steps MuJoCo and reports the
whole-body picture:

    total COM (x, z) and COM height above the floor
    standing pelvis tilt and support-polygon margin
    peak hip / knee / ankle actuator torque (worst over the standing poses)
    quasi-static weight-shift limit

so the packaging trade-off ("lower is more stable, but is it reachable?") can be
read off directly.  It does NOT pick a layout.

Requires `mujoco`.  Prints SKIPPED / exits 0 without it.

Usage:
    python3 placement_study.py config/cara_full_body.yaml
    python3 placement_study.py config/cara_full_body.yaml \
        --layouts both_pelvis_low,both_high --baseline baselines/lower_body_standing.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import leg_model as lm
import subsystem_sweep as sub


def run(config, layout_names, baseline_path):
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    base_spec = lm.load_spec(config)
    all_layouts = lm.electronics_layouts(base_spec)
    if not all_layouts:
        print("no electronics.layouts in this spec"); return 1
    names = layout_names or list(all_layouts)
    ws = (base_spec.get("analysis", {}) or {}).get("weight_shift", {}) or {}
    base_pose = ws.get("base_pose", "stand_nominal")
    stand_poses = (base_spec.get("analysis", {}) or {}).get("standing_poses") or [base_pose]
    mounts = (base_spec.get("electronics", {}) or {}).get("mounts", {}) or {}

    e = base_spec.get("electronics", {})
    jm = float(e.get("jetson", {}).get("mass", 0.0))
    bm = float(e.get("battery", {}).get("mass", 0.0))
    print(f"Electronics placement study  ({base_spec['meta']['name']};  "
          f"Jetson {jm:.2f} kg + battery {bm:.2f} kg = {jm+bm:.2f} kg to place)")
    print(f"standing at '{base_pose}'; peak torque worst over {stand_poses}\n")
    print("  mount presets (link, z offset in that link frame):")
    for k, v in mounts.items():
        print(f"    {k:<12} {v.get('link'):<7} z={v.get('z', 0.0):+.3f}")

    hdr = (f"\n  {'layout':<24} {'jetson':<11} {'battery':<11} "
           f"{'COM h':>7} {'COMz_pel':>9} {'tilt°':>6} {'margin':>7} "
           f"{'hip τ':>7} {'knee τ':>7} {'ankle τ':>8} {'shift lim':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))

    rows = []
    for name in names:
        layout = all_layouts[name]
        spec = lm.load_spec(config)
        lm.apply_electronics_layout(spec, layout)
        r = sub.measure(spec, stand_poses, base_pose, ws.get("accept", {}))
        rows.append((name, layout, r))
        print(f"  {name:<24} {layout['jetson']:<11} {layout['battery']:<11} "
              f"{r['com_h']:>7.3f} {r['com_z_pel']:>+9.1f} {r['tilt']:>6.2f} {r['margin']:>7.1f} "
              f"{r['hip']:>7.3f} {r['knee']:>7.3f} {r['ankle']:>8.3f} {r['shift_limit']:>9.3f}")

    if len(rows) >= 2:
        lo = min(rows, key=lambda x: x[2]["com_z_pel"])
        hi = max(rows, key=lambda x: x[2]["com_z_pel"])
        print(f"\n  lowest COM : '{lo[0]}'  (COM {lo[2]['com_z_pel']:+.1f} mm vs pelvis, "
              f"shift limit {lo[2]['shift_limit']:.3f} m, worst knee τ {lo[2]['knee']:.3f} N·m)")
        print(f"  highest COM: '{hi[0]}'  (COM {hi[2]['com_z_pel']:+.1f} mm vs pelvis, "
              f"shift limit {hi[2]['shift_limit']:.3f} m, worst knee τ {hi[2]['knee']:.3f} N·m)")
        print(f"  spread     : COM {hi[2]['com_z_pel']-lo[2]['com_z_pel']:+.1f} mm, "
              f"knee τ {hi[2]['knee']-lo[2]['knee']:+.3f} N·m, "
              f"shift limit {hi[2]['shift_limit']-lo[2]['shift_limit']:+.3f} m")

    if baseline_path and os.path.exists(baseline_path):
        b = json.load(open(baseline_path, encoding="utf-8"))
        bp = b.get("poses", {}).get(base_pose, {})
        if bp:
            print(f"\n  lower-body baseline ({b.get('total_mass',0):.2f} kg): "
                  f"tilt {bp.get('tilt_deg',0):.2f} deg, margin {bp.get('com_margin_mm',0):.1f} mm, "
                  f"peak torque {bp.get('peak_torque_Nm',0):.3f} N·m")

    print("\nNo layout is chosen here -- this is a design-analysis table.")
    print("(provisional masses / mount offsets / PD gains; all TODO: measured/CAD)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--layouts", default=None, help="comma-separated subset of electronics.layouts")
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args(argv)
    return run(args.config, args.layouts.split(",") if args.layouts else None, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
