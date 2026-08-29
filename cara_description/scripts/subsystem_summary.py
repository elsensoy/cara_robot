#!/usr/bin/env python3
"""U6 -- full-body standing + weight-shift regression, per-subsystem summary.

Roadmap Phase U6: once torso + head + electronics + passive arms + ears exist,
re-run the whole standing / weight-shift suite on the complete model and
"save a summary table that makes the effect of each added upper-body subsystem
explicit."

This builds the full body up ONE subsystem at a time -- by pruning the composed
`cara_full_body` spec down to each stage -- and runs the same MuJoCo measurement
(`subsystem_sweep.measure`) at every stage:

    lower body  ->  + torso  ->  + head/neck  ->  + electronics  ->  + arms  ->  + ears

Per stage it reports whole-body mass, COM height vs the pelvis, worst-pose
standing tilt / support margin / hip-knee-ankle peak torque, the whole-body
inertia tensor about the COM, and -- from the quasi-static lateral probe --
the largest controlled shift amplitude with the loaded / unloaded foot force,
pelvis roll and slip at that amplitude.

The "lower body" stage is cross-checked against `config/cara_lower_body.yaml`
loaded directly: the pruned full body must reproduce it.

Nothing is auto-tuned; this is a design-analysis report, not a controller.

Requires `mujoco`.  Prints SKIPPED / exits 0 without it.

Usage:
    python3 subsystem_summary.py                       # table to stdout
    python3 subsystem_summary.py --md docs/subsystem_summary.md
"""

from __future__ import annotations

import argparse
import copy
import datetime
import os
import sys

import leg_model as lm

_HERE = os.path.dirname(os.path.abspath(__file__))
FULL_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_full_body.yaml"))
LOWER_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_lower_body.yaml"))

# (label, links added by this subsystem, joints added by this subsystem)
STAGES = [
    ("lower body",    [], []),
    ("+ torso",       ["torso"], ["base_to_torso"]),
    ("+ head/neck",   ["neck_yaw_link", "neck_roll_link", "head"],
                      ["neck_yaw", "neck_roll", "neck_pitch"]),
    ("+ electronics", ["jetson", "battery"], ["mount_jetson", "mount_battery"]),
    ("+ arms",        ["l_arm", "r_arm"], ["l_shoulder", "r_shoulder"]),
    ("+ ears",        ["l_ear", "r_ear", "l_ear_servo", "r_ear_servo"],
                      ["l_ear_joint", "r_ear_joint", "l_ear_servo_mount", "r_ear_servo_mount"]),
]


def pruned_spec(base_spec: dict, upto: int) -> dict:
    """`base_spec` with every subsystem AFTER index `upto` removed."""
    drop_links, drop_joints = set(), set()
    for _, ls, js in STAGES[upto + 1:]:
        drop_links.update(ls)
        drop_joints.update(js)
    spec = copy.deepcopy(base_spec)
    spec["links"] = [l for l in spec.get("links", []) if l["name"] not in drop_links]
    spec["joints"] = [j for j in spec.get("joints", []) if j["name"] not in drop_joints]
    dl = (spec.get("dynamics", {}) or {}).get("links", {}) or {}
    for n in list(dl):
        if n in drop_links:
            del dl[n]
    return spec


def _rows(argv_md=None):
    import subsystem_sweep as sub

    base = lm.load_spec(FULL_CONFIG)
    ws = (base.get("analysis", {}) or {}).get("weight_shift", {}) or {}
    base_pose = ws.get("base_pose", "stand_nominal")
    accept = ws.get("accept", {})
    stand_poses = (base.get("analysis", {}) or {}).get("standing_poses") or [base_pose]

    out = []
    for i, (label, _, _) in enumerate(STAGES):
        spec = pruned_spec(base, i)
        r = sub.measure(spec, stand_poses, base_pose, accept)
        out.append((label, r))
    return base_pose, stand_poses, out


def _fmt_table(base_pose, stand_poses, rows):
    L = []
    L.append(f"Full-body subsystem summary (U6)   model cara_full_body")
    L.append(f"each row = lower body + the subsystems listed, cumulative")
    L.append(f"standing metrics = worst over {stand_poses}; base pose = {base_pose}")
    L.append(f"weight shift = quasi-static lateral COM probe, values captured AT the shift limit")
    L.append("")
    hdr = (f"  {'stage':<14} {'kg':>6} {'COMz_pel':>9} {'tilt°':>6} {'margin':>7} "
           f"{'hipτ':>6} {'kneeτ':>6} {'anklτ':>6} {'Ixx@COM':>9} {'Izz@COM':>9} "
           f"{'shift':>6} {'Fn_load':>8} {'Fn_unld':>8} {'roll°':>6} {'slip':>6}")
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    for label, r in rows:
        s = r["shift"]
        fl = f"{s['fn_loaded']:>8.1f}" if s["fn_loaded"] is not None else f"{'--':>8}"
        fu = f"{s['fn_unloaded']:>8.1f}" if s["fn_unloaded"] is not None else f"{'--':>8}"
        rd = f"{s['roll_deg']:>6.2f}" if s["roll_deg"] is not None else f"{'--':>6}"
        sl = f"{s['slip_mm']:>6.2f}" if s["slip_mm"] is not None else f"{'--':>6}"
        L.append(f"  {label:<14} {r['m_total']:>6.2f} {r['com_z_pel']:>+9.1f} {r['tilt']:>6.2f} "
                 f"{r['margin']:>7.1f} {r['hip']:>6.2f} {r['knee']:>6.2f} {r['ankle']:>6.2f} "
                 f"{r['Ixx']:>9.5f} {r['Izz']:>9.5f} {r['shift_limit']:>6.3f} {fl} {fu} {rd} {sl}")

    L.append("")
    L.append("  marginal contribution of each subsystem (Δ vs the row above):")
    for k in range(1, len(rows)):
        (_, a), (label, b) = rows[k - 1], rows[k]
        L.append(f"    {label:<14} {b['m_total']-a['m_total']:+.2f} kg   "
                 f"COM {b['com_z_pel']-a['com_z_pel']:+.1f} mm   "
                 f"knee τ {b['knee']-a['knee']:+.2f}   "
                 f"Ixx {b['Ixx']-a['Ixx']:+.5f}   Izz {b['Izz']-a['Izz']:+.5f} kg·m²   "
                 f"shift {b['shift_limit']-a['shift_limit']:+.3f} m")

    (_, lo), (_, hi) = rows[0], rows[-1]
    L.append("")
    L.append(f"  lower body -> full body:  mass {lo['m_total']:.2f} -> {hi['m_total']:.2f} kg   "
             f"COM {lo['com_z_pel']:+.0f} -> {hi['com_z_pel']:+.0f} mm (pelvis frame)   "
             f"weight-shift limit {lo['shift_limit']:.3f} -> {hi['shift_limit']:.3f} m")
    return "\n".join(L)


def run(md_path: str | None) -> int:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    base_pose, stand_poses, rows = _rows()

    # cross-check: the pruned "lower body" stage vs the real cara_lower_body spec
    import subsystem_sweep as sub
    lb = lm.load_spec(LOWER_CONFIG)
    lws = (lb.get("analysis", {}) or {}).get("weight_shift", {}) or {}
    lb_r = sub.measure(lb, (lb.get("analysis", {}) or {}).get("standing_poses") or [base_pose],
                       lws.get("base_pose", base_pose), lws.get("accept", {}))
    pruned_r = rows[0][1]
    dm = abs(lb_r["m_total"] - pruned_r["m_total"])
    dtilt = abs(lb_r["tilt"] - pruned_r["tilt"])

    table = _fmt_table(base_pose, stand_poses, rows)
    print(table)
    print(f"\n  cross-check: pruned 'lower body' vs config/cara_lower_body.yaml  "
          f"-> Δmass {dm*1e3:.1f} g, Δtilt {dtilt:.3f}°  "
          f"({'OK' if dm < 1e-6 and dtilt < 0.05 else 'MISMATCH'})")
    print("\n(provisional masses / mount offsets / PD gains; all TODO: measured/CAD)")

    if md_path:
        stamp = datetime.date.today().isoformat()
        body = ["<!-- GENERATED by scripts/subsystem_summary.py -- do not edit by hand -->",
                f"# U6 -- full-body subsystem summary  ({stamp})", "",
                "Regenerate: `python3 scripts/subsystem_summary.py --md docs/subsystem_summary.md`",
                "", "```", table, "```", ""]
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(body))
        print(f"\nsaved -> {md_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default=None, help="also write the summary table here (markdown)")
    args = ap.parse_args(argv)
    return run(args.md)


if __name__ == "__main__":
    sys.exit(main())
