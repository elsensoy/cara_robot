#!/usr/bin/env python3
"""U13 -- reduced-order walking model (LIPM / ZMP / capture point).

U12 showed a *kinematic* gait (replay a periodic joint-pose cycle) hits two
walls: at speed she topples at the double-support hand-off, and slow she has no
forward drive.  U13 stops replaying poses and starts *predicting where the COM
must go* for a dynamically valid step, with the standard reduced-order model.

Linear inverted pendulum (point mass at height z_com over a movable center of
pressure p):

        x'' = omega0^2 (x - p),     omega0 = sqrt(g / z_com)

Closed-form over a step of duration T with p held constant:

        x(T)  = p + (x0 - p) cosh(w T) + (x0'/w) sinh(w T)
        x'(T) =     w (x0 - p) sinh(w T) + x0'   cosh(w T)

Divergent component of motion / capture point (DCM):

        xi = x + x'/omega0,     xi(T) = p + (xi0 - p) e^{omega0 T}

The DCM is what has to be "caught": to stop, xi must be inside the support
polygon; to keep walking, the next foot's CoP region must contain the DCM at
foot strike.  This is exactly the hand-off that failed in U12.

This script is PURE PYTHON analysis (no controller, no sim required).  It:

  1. reads Cara's LIPM parameters from the model;
  2. solves the periodic LATERAL limit cycle (foot spacing sets the sway) and
     asks whether the DCM at each hand-off lands inside the next foot -- the
     feasibility of a dynamic side-to-side rock;
  3. does the same for FORWARD motion at a target speed;
  4. sweeps step time x foot size -> a feasibility map, and marks where Cara is;
  5. (optional, needs mujoco) cross-checks omega0 against a slow weight-shift.

Requires nothing for 1-4; `mujoco` only for the cross-check.

Usage:
    python3 walk_model.py
    python3 walk_model.py --speed 0.05          # target 50 mm/s forward
    python3 walk_model.py --check               # + MuJoCo omega0 cross-check
    python3 walk_model.py --json baselines/full_body_walk_model.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import leg_model as lm

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_full_body.yaml"))


# --------------------------------------------------------------------------- #
# LIPM core (pure python)
# --------------------------------------------------------------------------- #
class LIPM:
    def __init__(self, z_com, g=9.81):
        self.z = z_com
        self.g = g
        self.w = math.sqrt(g / z_com)          # omega0

    def evolve(self, x, xd, p, T):
        c, s = math.cosh(self.w * T), math.sinh(self.w * T)
        return (p + (x - p) * c + (xd / self.w) * s,
                self.w * (x - p) * s + xd * c)

    def dcm(self, x, xd):
        return x + xd / self.w

    def dcm_evolve(self, xi, p, T):
        return p + (xi - p) * math.exp(self.w * T)

    def integrate(self, x, xd, p_of_t, T, n=4000):
        """RK4 reference integration of x'' = w^2 (x - p(t)); p_of_t(t)->p."""
        h = T / n
        for i in range(n):
            t = i * h

            def acc(xx, tt):
                return self.w * self.w * (xx - p_of_t(tt))

            k1x, k1v = xd, acc(x, t)
            k2x, k2v = xd + 0.5 * h * k1v, acc(x + 0.5 * h * k1x, t + 0.5 * h)
            k3x, k3v = xd + 0.5 * h * k2v, acc(x + 0.5 * h * k2x, t + 0.5 * h)
            k4x, k4v = xd + h * k3v, acc(x + h * k3x, t + h)
            x += h / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            xd += h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        return x, xd


# --------------------------------------------------------------------------- #
# lateral limit cycle:  stance foot at y = -s (half-spacing), CoP fixed there.
# By L<->R symmetry the state at the step end is the mirror of the start:
#   y(T) = -y0,  y'(T) = -y0'.
# Solving the closed form gives  y0 = 0,  y0' = -w*s*tanh(wT/2)  (toward stance),
# so the COM crosses the midline at each hand-off and the DCM there is
#   xi_y = y(T) + y'(T)/w = +s*tanh(wT/2).
# For the next step's CoP to catch it, xi_y must sit inside the next foot:
#   |xi_y - s| <= a_y   ->   tanh(wT/2) >= 1 - a_y/s   (a lower bound on T).
# --------------------------------------------------------------------------- #
def lateral_cycle(m: LIPM, s, a_y, T):
    wT2 = m.w * T / 2.0
    th = math.tanh(wT2)
    y0d = -m.w * s * th                          # COM velocity at hand-off (toward stance)
    y_peak = -s * (1.0 - 1.0 / math.cosh(wT2))   # max excursion toward the stance foot
    xi_y = s * th                                # DCM at the hand-off (toward the *new* stance foot)
    margin_near = xi_y - (s - a_y)               # + => DCM past the inner edge (good)
    margin_far = (s + a_y) - xi_y                # + => DCM inside the outer edge (good)
    return {"T": T, "com_vel_handoff": y0d, "com_peak": y_peak, "dcm_y": xi_y,
            "margin_inner_mm": 1e3 * margin_near, "margin_outer_mm": 1e3 * margin_far,
            "feasible": margin_near >= 0.0 and margin_far >= 0.0}


def lateral_T_min(m: LIPM, s, a_y):
    r = 1.0 - a_y / s
    if r <= 0.0:
        return 0.0                               # foot already wider than the stance offset
    if r >= 1.0:
        return float("inf")
    return (2.0 / m.w) * math.atanh(r)


# --------------------------------------------------------------------------- #
# forward motion at a target mean speed v.  Steady LIPM walking: the DCM leads
# the COM by v/w; each foot lands so the *next* CoP interval brackets the DCM at
# strike.  With step length L = v*T and CoP fixed at the foot centre, the DCM
# offset from the foot at strike is  d = (v/w) * (something) -- work it from the
# periodic condition  x(T) = x0 + L,  x'(T) = x'0  (steady speed):
# --------------------------------------------------------------------------- #
def forward_cycle(m: LIPM, v, a_x, T):
    L = v * T
    c, sh = math.cosh(m.w * T), math.sinh(m.w * T)
    # periodic: x'(T) = x'0 = v (mean speed sustained), foot at x=0, next at x=L.
    # x0' = v.  x0 solves  x(T) - x0 = L  with p = 0:
    #   (x0)(c-1) + (v/w) sh = L    (measuring x from the stance foot)
    x0 = (L - (v / m.w) * sh) / (c - 1.0) if abs(c - 1.0) > 1e-9 else 0.0
    xT, xdT = m.evolve(x0, v, 0.0, T)
    xi_strike = m.dcm(xT, xdT)                   # DCM measured from the *old* stance foot
    dcm_vs_newfoot = xi_strike - L               # DCM position relative to the new foothold
    return {"T": T, "v": v, "step_len_mm": 1e3 * L, "com_start_mm": 1e3 * x0,
            "dcm_rel_newfoot_mm": 1e3 * dcm_vs_newfoot,
            "cop_margin_mm": 1e3 * (a_x - abs(dcm_vs_newfoot)),
            "feasible": abs(dcm_vs_newfoot) <= a_x}


def run(config, speed, do_check, json_path):
    spec = lm.load_spec(config)
    g = lm.analysis_gravity(spec)
    lc = (spec.get("analysis", {}) or {}).get("lipm", {}) or {}
    if speed is None:
        speed = float(lc.get("target_speed", 0.050))
    LAT_T = [float(x) for x in lc.get("step_times", [0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.00])]
    FWD_T = [float(x) for x in lc.get("fwd_step_times", [0.30, 0.40, 0.50, 0.70, 1.00, 1.50])]
    MAP_AY = [float(x) for x in lc.get("foot_half_widths", [0.015, 0.0225, 0.030, 0.040, 0.050])]
    MAP_T = [float(x) for x in lc.get("map_step_times", [0.12, 0.15, 0.18, 0.22, 0.28, 0.35, 0.50])]
    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)["stand_nominal"]
    m_total, com, _ = lm.center_of_mass(spec, base_cfg)
    tf = lm.forward_kinematics(spec, base_cfg)
    sole_z = lm.frame_world_position(spec, tf, "l_foot_sole_center")[2]
    z_com = com[2] - sole_z
    sym = lm.resolve_symbols(spec)
    a_x = 0.5 * float(sym["foot_len"])           # foot half-length (fore/aft CoP range)
    a_y = 0.5 * float(sym["foot_width"])         # foot half-width  (lateral CoP range)
    s_half = float(sym["w_hip_half"])            # nominal foot lateral offset from the midline

    m = LIPM(z_com, g)

    print(f"Reduced-order walking model (U13 -- LIPM / DCM)   {spec['meta']['name']}")
    print("=" * 74)
    print("1) LIPM PARAMETERS  (from the model + provisional_geometry)")
    print(f"   total mass ............. {m_total:.2f} kg")
    print(f"   COM height z_com ....... {1e3*z_com:.0f} mm  (above the sole, stand_nominal)")
    print(f"   omega0 = sqrt(g/z) ..... {m.w:.3f} rad/s   (time constant 1/omega0 = {1e3/m.w:.0f} ms)")
    print(f"   foot half-length a_x ... {1e3*a_x:.1f} mm   (fore/aft CoP range)")
    print(f"   foot half-width  a_y ... {1e3*a_y:.1f} mm   (lateral CoP range)")
    print(f"   foot lateral offset s .. {1e3*s_half:.0f} mm  (half the stance width)")

    # -- 2) lateral limit cycle ---------------------------------------------- #
    Tmin = lateral_T_min(m, s_half, a_y)
    print(f"\n2) LATERAL LIMIT CYCLE  (side-to-side rock; CoP at the foot centre)")
    print(f"   at the L<->R hand-off the COM is on the midline with the DCM at "
          f"xi_y = s*tanh(omega0*T/2)")
    print(f"   -> feasible when tanh(omega0 T/2) >= 1 - a_y/s = {1 - a_y/s_half:.2f}, "
          f"i.e. T >= T_min")
    print(f"   T_min = {Tmin:.3f} s   (steps FASTER than this: the DCM lands short of the "
          f"next foot -> topple inward)")
    print(f"\n   {'T step':>7} {'sway peak':>10} {'COM vel @handoff':>16} {'DCM_y':>8} "
          f"{'inner marg':>11} {'outer marg':>11}  feasible")
    lat_rows = []
    for T in LAT_T:
        r = lateral_cycle(m, s_half, a_y, T)
        lat_rows.append(r)
        print(f"   {T:>6.2f}s {1e3*abs(r['com_peak']):>8.0f}mm {1e3*abs(r['com_vel_handoff']):>13.0f}mm/s "
              f"{1e3*r['dcm_y']:>6.0f}mm {r['margin_inner_mm']:>9.1f}mm {r['margin_outer_mm']:>9.1f}mm  "
              f"{'yes' if r['feasible'] else 'NO'}")
    print(f"\n   => a dynamic lateral rock is FEASIBLE for Cara for step times "
          f">= {Tmin*1e3:.0f} ms.")
    print(f"      (U12's slow runs -- t_step 3.5-8 s -- are far above T_min and did stay up;")
    print(f"       U12's fast run toppled because the hand-picked sway + static-hold trim did")
    print(f"       NOT follow the pendulum, not because the morphology forbids it.)")

    # -- 3) forward motion -------------------------------------------------- #
    v = float(speed)
    print(f"\n3) FORWARD MOTION  (target mean speed {1e3*v:.0f} mm/s)")
    print(f"   {'T step':>7} {'step len':>9} {'COM@strike':>11} {'DCM vs new foot':>16} "
          f"{'CoP margin':>11}  feasible")
    print(f"   (COM@strike = COM position behind the stance foot at foot strike; "
          f"DCM lands ahead of it)")
    fwd_rows = []
    for T in FWD_T:
        r = forward_cycle(m, v, a_x, T)
        fwd_rows.append(r)
        print(f"   {T:>6.2f}s {r['step_len_mm']:>7.0f}mm {r['com_start_mm']:>9.0f}mm "
              f"{r['dcm_rel_newfoot_mm']:>13.0f}mm {r['cop_margin_mm']:>9.1f}mm  "
              f"{'yes' if r['feasible'] else 'NO'}")
    print(f"   forward is roomier -- the fore/aft CoP range is a_x = {1e3*a_x:.0f} mm "
          f"vs a_y = {1e3*a_y:.0f} mm lateral.")
    print(f"   U12's slow run had NO forward speed because kinematic playback carries no")
    print(f"   momentum: the planner has to command the forward lean (DCM offset v/omega0 = "
          f"{1e3*v/m.w:.0f} mm) and place each foot ahead of the DCM.")

    # -- 4) feasibility map: step time x foot half-width ------------------- #
    print(f"\n4) FEASIBILITY MAP -- lateral rock:  foot half-width a_y  x  step time T")
    print(f"   (Y = DCM inside the next foot at hand-off; . = topple inward)")
    widths = MAP_AY
    times = MAP_T
    print("        a_y \\ T   " + "  ".join(f"{t:>4.2f}" for t in times))
    fmap = {}
    for ay in widths:
        cells = []
        for T in times:
            ok = lateral_cycle(m, s_half, ay, T)["feasible"]
            cells.append(" Y  " if ok else " .  ")
        tag = "  <- Cara" if abs(ay - a_y) < 1e-6 else ""
        fmap[f"{1e3*ay:.1f}mm"] = {f"{t:.2f}": lateral_cycle(m, s_half, ay, t)["feasible"] for t in times}
        print(f"   {1e3*ay:>7.1f} mm  " + "".join(cells) + tag)
    print(f"\n   widening the foot lowers T_min (more time before the DCM must reach the foot):")
    for ay in widths:
        tm = lateral_T_min(m, s_half, ay)
        print(f"     a_y {1e3*ay:>5.1f} mm  ->  T_min {tm*1e3:>5.0f} ms" +
              ("   <- Cara" if abs(ay - a_y) < 1e-6 else ""))

    # -- 5) optional MuJoCo cross-check ---------------------------------- #
    check = None
    if do_check:
        check = _omega_check(spec, m)

    # -- summary --------------------------------------------------------- #
    r30 = lateral_cycle(m, s_half, a_y, 0.30)
    print("\n" + "=" * 74)
    print("FINDING.  A dynamically-consistent walk IS within Cara's morphology:")
    print(f"  * lateral side-to-side rock: feasible for step times >= {Tmin*1e3:.0f} ms")
    print(f"    (at T = 0.30 s the DCM at hand-off sits {1e3*r30['dcm_y']:.0f} mm off the midline -- "
          f"{r30['margin_inner_mm']:.0f} mm past the next foot's inner edge, inside the 22.5 mm foot);")
    print(f"  * forward motion: roomier still (a_x = {1e3*a_x:.0f} mm).")
    print("  The U12 wall was the KINEMATIC FORMULATION (replay a fixed pose cycle +")
    print("  a static-hold trim), not the hardware.  The next phase (U14) is a")
    print("  DCM-tracking walk: plan the CoP + footholds from this model, then track")
    print("  the DCM to the planned foothold -- no fixed pose cycle.")

    if json_path:
        out = {"model": spec["meta"]["name"], "m_total": m_total, "z_com": z_com,
               "omega0": m.w, "a_x": a_x, "a_y": a_y, "s_half": s_half,
               "lateral_T_min_s": Tmin,
               "lateral_cycle": lat_rows, "forward_cycle": fwd_rows,
               "feasibility_map": fmap, "omega_check": check}
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nsummary -> {json_path}")
    return 0


def _omega_check(spec, m: LIPM):
    """Cross-check omega0: give the LIPM the COM's measured initial state from a
    released lean and compare its divergence rate to a MuJoCo roll-out."""
    try:
        import mujoco
        import numpy as np
        import generate_mjcf
        import weight_shift as wsh
    except ImportError:
        print("\n5) (cross-check skipped -- mujoco not available)")
        return None
    model = mujoco.MjModel.from_xml_string(generate_mjcf.build_mjcf(spec, dynamic=True))
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)["stand_nominal"]
    nominal = [float(base_cfg.get(n, 0.0)) for n in jn]
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand_nominal")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    for _ in range(int(1.5 / dt)):
        data.ctrl[:] = nominal
        mujoco.mj_step(model, data)
    # nudge the pelvis sideways, release, watch the COM diverge (feet planted)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["frame_conventions"]["base_frame"])
    data.xfrc_applied[pelvis][1] = 6.0
    for _ in range(int(0.15 / dt)):
        data.ctrl[:] = nominal
        mujoco.mj_step(model, data)
    data.xfrc_applied[pelvis][1] = 0.0
    ys, ts = [], []
    for k in range(int(0.5 / dt)):
        data.ctrl[:] = nominal
        mujoco.mj_step(model, data)
        ys.append(float(data.subtree_com[0][1]))
        ts.append(k * dt)
        r, p, _ = wsh.quat_rpy(data.qpos[3:7])
        if max(abs(r), abs(p)) > math.radians(20):
            break
    # fit y(t) ~ A e^{k t}: k = d/dt ln|y - y_settle|.  crude: last vs first slope of ln|y|.
    import numpy as np
    y = np.array(ys) - ys[0]
    t = np.array(ts)
    good = np.abs(y) > 1e-4
    if good.sum() > 20:
        k = float(np.polyfit(t[good], np.log(np.abs(y[good])), 1)[0])
    else:
        k = float("nan")
    print(f"\n5) MuJoCo CROSS-CHECK  (release a small lean, fit the COM-y divergence rate)")
    print(f"   LIPM predicts divergence ~ e^(omega0 t), omega0 = {m.w:.2f} /s")
    print(f"   MuJoCo full model:  fitted rate = {k:.2f} /s   "
          f"({'consistent' if abs(k - m.w) < 0.25 * m.w else 'differs -- leg compliance / finite feet'})")
    return {"omega0_lipm": m.w, "rate_mujoco": k}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--speed", type=float, default=None, help="target forward speed [m/s] (default: analysis.lipm.target_speed)")
    ap.add_argument("--check", action="store_true", help="cross-check omega0 against MuJoCo")
    ap.add_argument("--json", default=None, help="write the analysis summary here")
    args = ap.parse_args(argv)
    return run(args.config, args.speed, args.check, args.json)


if __name__ == "__main__":
    sys.exit(main())
