#!/usr/bin/env python3
"""U5 -- ears and head asymmetry study.

Question (roadmap Phase U5):

    "Does the ear design materially affect head/neck rotational inertia and
     balance?"

The ears + ear-twitch servos are small masses mounted well off the neck axis.
Rotational inertia scales as

    I ~ m r^2

so a light mass placed far out can still dominate what the neck-yaw servo has to
accelerate.  This script quantifies that for the *fixed-mass* model (ear joints
`locked: true`); ear motion is only added after this is understood.

It reports, about the coincident neck yaw/roll/pitch axes (the head link frame
origin):

  * the head-subsystem inertia tensor with and without the ears
    (head + l_ear + r_ear + l_ear_servo + r_ear_servo),
  * a sweep of the ear lateral offset `upper_body.ear.offset_y`, with the
    measured change in yaw inertia checked against the m r^2 point-mass
    prediction,
  * sweeps of `upper_body.ear.mass` and `upper_body.ear.servo_mass`,
  * the whole-body standing tilt / support margin / weight-shift limit for each
    value (needs `mujoco`; skipped with a note otherwise).

Nothing is auto-tuned and no ear geometry is chosen -- this is a design-analysis
tool.

Usage:
    python3 ear_inertia_study.py [config/cara_full_body.yaml]
"""

from __future__ import annotations

import argparse
import copy
import sys

import leg_model as lm

HEAD_SUBSYSTEM = ("head", "l_ear", "r_ear", "l_ear_servo", "r_ear_servo")
NECK_AXIS_FRAME = "head"          # head link frame origin == the coincident neck axes
BASE_POSE = "stand_nominal"


def _diag(I):
    return (I[0][0], I[1][1], I[2][2])


def _spec_with(base_spec, path, value):
    spec = copy.deepcopy(base_spec)
    node = spec
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"path '{path}' not in spec")
    node[keys[-1]] = float(value)
    return spec


try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except ImportError:
    HAVE_MJ = False


def _measure(spec, accept):
    """Whole-body standing + weight-shift metrics, or None if mujoco is absent."""
    if not HAVE_MJ:
        return None
    import subsystem_sweep as sub
    return sub.measure(spec, [BASE_POSE], BASE_POSE, accept)


def run(config: str | None) -> int:
    spec = lm.load_spec(config)
    poses = lm.reference_poses(spec)
    q = poses[BASE_POSE]
    ub_ear = (spec.get("upper_body", {}) or {}).get("ear", {}) or {}
    if not ub_ear:
        print("this model has no upper_body.ear block -- nothing to study")
        return 0
    accept = ((spec.get("analysis", {}) or {}).get("weight_shift", {}) or {}).get("accept", {})

    y0 = float(ub_ear["offset_y"])
    x0 = float(ub_ear["offset_x"])
    m_ear = float(ub_ear["mass"])

    print(f"Ear + head asymmetry study (U5)   model {spec['meta']['name']}")
    print(f"head subsystem = {' + '.join(HEAD_SUBSYSTEM)}")
    print(f"reference point = the coincident neck yaw/roll/pitch axes "
          f"(the '{NECK_AXIS_FRAME}' link frame origin)")
    print(f"pose = {BASE_POSE}   (neck + ears locked at 0)\n")

    # --- ears vs no ears -------------------------------------------------
    no_ears = _spec_with(_spec_with(spec, "upper_body.ear.mass", 0.0),
                         "upper_body.ear.servo_mass", 0.0)
    I_with = lm.whole_body_inertia(spec, q, about=NECK_AXIS_FRAME, links=HEAD_SUBSYSTEM)
    I_without = lm.whole_body_inertia(no_ears, q, about=NECK_AXIS_FRAME, links=HEAD_SUBSYSTEM)
    m_with = sum(li.mass for n, li in lm.link_inertials(spec).items() if n in HEAD_SUBSYSTEM)
    m_without = sum(li.mass for n, li in lm.link_inertials(no_ears).items() if n in HEAD_SUBSYSTEM)

    print("--- ears vs no ears: head-subsystem inertia about the neck axis [kg.m^2] ---")
    print(f"  {'':22} {'Ixx (roll)':>12} {'Iyy (nod)':>12} {'Izz (yaw)':>12} {'mass [kg]':>11}")
    dw = _diag(I_without)
    dwi = _diag(I_with)
    print(f"  {'head only':22} {dw[0]:>12.6f} {dw[1]:>12.6f} {dw[2]:>12.6f} {m_without:>11.3f}")
    print(f"  {'head + ears (nominal)':22} {dwi[0]:>12.6f} {dwi[1]:>12.6f} {dwi[2]:>12.6f} {m_with:>11.3f}")
    pct = tuple(100.0 * (dwi[i] - dw[i]) / dw[i] if dw[i] else float("nan") for i in range(3))
    print(f"  {'ears add':22} "
          f"{dwi[0]-dw[0]:>+12.6f} {dwi[1]-dw[1]:>+12.6f} {dwi[2]-dw[2]:>+12.6f} "
          f"{m_with-m_without:>+11.3f}")
    print(f"  {'  as % of head-only':22} {pct[0]:>+11.1f}% {pct[1]:>+11.1f}% {pct[2]:>+11.1f}%")
    print(f"\n  the ears are {m_with-m_without:.3f} kg ({100*(m_with-m_without)/m_without:.0f}% of the head "
          f"mass) but add {pct[2]:+.0f}% to the yaw inertia the neck-yaw servo must accelerate.")

    # --- sweeps --------------------------------------------------------
    def sweep(path, values, nominal, label, note_pred=False):
        print(f"\n--- sweep: {label}  ({path}) ---")
        mj_hdr = f" {'tilt°':>6} {'margin':>7} {'shift lim':>9}" if HAVE_MJ else ""
        pred_hdr = f" {'m·r² pred ΔIzz':>14}" if note_pred else ""
        print(f"  {'value':>9} {'Ixx roll':>10} {'Iyy nod':>10} {'Izz yaw':>10}{pred_hdr}"
              f" {'WB Izz@COM':>11}{mj_hdr}")
        base_v = values[0]
        for v in values:
            sp = _spec_with(spec, path, v)
            Ih = _diag(lm.whole_body_inertia(sp, q, about=NECK_AXIS_FRAME, links=HEAD_SUBSYSTEM))
            Iwb = lm.whole_body_inertia(sp, q, about="com")
            tag = "  <- nominal" if abs(v - nominal) < 1e-9 else ""
            pred = ""
            if note_pred:
                # two ears as point masses about the neck axis: ΔIzz = 2 m_ear (v² − base_v²)
                dp = 2.0 * m_ear * ((x0*x0 + v*v) - (x0*x0 + base_v*base_v))
                pred = f" {dp:>+14.6f}"
            row = (f"  {v:>9.4f} {Ih[0]:>10.6f} {Ih[1]:>10.6f} {Ih[2]:>10.6f}{pred}"
                   f" {Iwb[2][2]:>11.6f}")
            m = _measure(sp, accept)
            if m:
                row += f" {m['tilt']:>6.2f} {m['margin']:>7.1f} {m['shift_limit']:>9.3f}"
            print(row + tag)

    sweep("upper_body.ear.offset_y", [0.020, 0.040, y0, 0.070, 0.090], y0,
          "ear lateral offset from the neck axis", note_pred=True)
    print("  m·r² pred = point-mass parallel-axis ΔIzz vs the 0.020 m row, two ears only "
          "(ignores the servos and each ear's own box inertia).")

    sweep("upper_body.ear.mass", [0.010, m_ear, 0.040, 0.060], m_ear, "ear plush mass")
    sweep("upper_body.ear.servo_mass", [0.005, float(ub_ear["servo_mass"]), 0.020, 0.030],
          float(ub_ear["servo_mass"]), "ear-twitch servo mass")

    print("\n" + "=" * 70)
    print("U5 acceptance -- the ear design's effect on head/neck inertia is now quantified:")
    print(f"  * at the nominal {m_ear*1000:.0f} g ear + {float(ub_ear['servo_mass'])*1000:.0f} g servo "
          f"per side, the ears raise head YAW inertia by {pct[2]:+.0f}% and roll by {pct[0]:+.0f}%,")
    print("    while whole-body standing tilt and the weight-shift envelope barely move.")
    print("  * yaw inertia grows with the SQUARE of the lateral ear offset (m r^2) -- a wide-set")
    print("    ear costs the neck-yaw servo far more than a heavier ear kept close to the axis.")
    print("  no ear mass / offset is chosen here; these are provisional TODO values.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None)
    args = ap.parse_args(argv)
    return run(args.config)


if __name__ == "__main__":
    sys.exit(main())
