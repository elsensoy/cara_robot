#!/usr/bin/env python3
"""Forward-kinematics sanity checks for Cara's left leg.

This is NOT inverse kinematics and NOT a dynamics check. It only confirms
that sweeping each joint angle moves the foot in a physically reasonable
direction, and that the leg never stretches beyond its bone lengths.

Checks:
  A. zero pose        -- foot sole lands exactly under the hip at the summed
                         link length (foot_x_off, w_hip_half, -(L_thigh+L_shin+h_ankle))
  B. knee_pitch  +    -- shin swings rearward: sole moves -X and rises (+Z);
                         the knee joint centre itself does not move
  C. hip_pitch   +    -- whole leg swings rearward: sole X decreases
                         monotonically; the hip centre does not move
  D. hip_roll    +    -- thigh abducts: sole moves outward (+Y) monotonically
  E. hip_yaw     +    -- foot heading rotates: the foot-frame +X axis gains a
                         +Y component monotonically
  F. ankle_pitch +    -- toe drops: the foot-frame +Z axis gains a +X
                         component monotonically; the ankle centre is fixed
  G. reach bound      -- for many random poses in-limits, the hip->ankle
                         distance never exceeds L_thigh + L_shin, with
                         equality only when the knee is at 0

Exit code 0 = all checks passed, 1 = one or more failed.

Usage:
    python3 fk_sanity_check.py [path/to/left_leg.yaml]
"""

from __future__ import annotations

import math
import random
import sys

import leg_model as lm

POS_TOL = 1e-9          # metres, for exact-geometry assertions
MONO_TOL = 1e-6         # metres / unit, slack for "monotonic" comparisons
REACH_TOL = 1e-9        # metres, reach-bound slack
N_RANDOM = 2000
SWEEP_N = 25


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.n = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        self.n += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  --  {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n == 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _is_increasing(xs: list[float], tol: float) -> bool:
    return all(b - a > -tol for a, b in zip(xs, xs[1:])) and xs[-1] - xs[0] > tol


def _is_decreasing(xs: list[float], tol: float) -> bool:
    return all(a - b > -tol for a, b in zip(xs, xs[1:])) and xs[0] - xs[-1] > tol


def _sole(spec, q):
    return lm.frame_world_position(spec, lm.forward_kinematics(spec, q), "l_foot_sole_center")


def _col(mat, j):
    return (mat[0][j], mat[1][j], mat[2][j])


def run(path: str | None) -> int:
    spec = lm.load_spec(path)
    syms = lm.resolve_symbols(spec)
    chain = {jm.name: jm for jm in lm.build_chain(spec)}
    rep = Report()

    Lt, Ls, ha = syms["L_thigh"], syms["L_shin"], syms["h_ankle"]
    wh, fx = syms["w_hip_half"], syms["foot_x_off"]

    # ----- A. zero pose ---------------------------------------------------- #
    print("== A. zero pose ==")
    sole0 = _sole(spec, {})
    expected = (fx + syms["x_hip"], wh, syms["z_hip"] - (Lt + Ls + ha))
    err = lm.vec_norm(lm.vec_sub(sole0, expected))
    rep.check(err <= POS_TOL, "foot sole at hip-x, hip-y, summed leg length below hip",
              detail=f"sole={tuple(round(c, 6) for c in sole0)} expected={tuple(round(c, 6) for c in expected)}")

    hip_c0 = lm.link_origin(lm.forward_kinematics(spec, {}), "l_hip_yaw_link")
    knee_c0 = lm.link_origin(lm.forward_kinematics(spec, {}), "l_shin")
    ankle_c0 = lm.link_origin(lm.forward_kinematics(spec, {}), "l_ankle_link")

    # ----- B. knee_pitch + ---------------------------------------------- #
    print("\n== B. knee_pitch positive -> knee flexion ==")
    kj = chain["l_knee_pitch"]
    qs = _linspace(0.0, min(kj.upper, 1.2), SWEEP_N)
    xs, sole_z, ankle_z, knee_moves = [], [], [], []
    for a in qs:
        tf = lm.forward_kinematics(spec, {"l_knee_pitch": a})
        s = lm.frame_world_position(spec, tf, "l_foot_sole_center")
        xs.append(s[0])
        sole_z.append(s[2])
        ankle_z.append(lm.link_origin(tf, "l_ankle_link")[2])
        knee_moves.append(lm.vec_norm(lm.vec_sub(lm.link_origin(tf, "l_shin"), knee_c0)))
    rep.check(_is_decreasing(xs, MONO_TOL), "sole moves rearward (-X) as knee flexes",
              detail=f"x: {xs[0]:.4f} -> {xs[-1]:.4f}")
    # The ankle centre is exactly below the knee axis, so it lifts monotonically.
    # The sole carries a small toe (+X) offset, so it dips a fraction of a mm
    # before lifting -- only require a net rise there.
    rep.check(_is_increasing(ankle_z, MONO_TOL), "ankle centre rises (+Z) monotonically as knee flexes",
              detail=f"z: {ankle_z[0]:.4f} -> {ankle_z[-1]:.4f}")
    rep.check(sole_z[-1] - sole_z[0] > MONO_TOL, "sole nets a rise (+Z) as knee flexes",
              detail=f"z: {sole_z[0]:.4f} -> {sole_z[-1]:.4f}")
    rep.check(max(knee_moves) <= POS_TOL, "knee joint centre stays fixed during knee sweep",
              detail=f"max drift {max(knee_moves):.2e} m")

    # ----- C. hip_pitch + ---------------------------------------------- #
    print("\n== C. hip_pitch positive -> leg swings rearward ==")
    hp = chain["l_hip_pitch"]
    qs = _linspace(0.0, min(hp.upper, 1.0), SWEEP_N)
    xs, hip_moves = [], []
    for a in qs:
        tf = lm.forward_kinematics(spec, {"l_hip_pitch": a})
        xs.append(lm.frame_world_position(spec, tf, "l_foot_sole_center")[0])
        hip_moves.append(lm.vec_norm(lm.vec_sub(lm.link_origin(tf, "l_hip_yaw_link"), hip_c0)))
    rep.check(_is_decreasing(xs, MONO_TOL), "sole X decreases monotonically as hip extends",
              detail=f"x: {xs[0]:.4f} -> {xs[-1]:.4f}")
    rep.check(max(hip_moves) <= POS_TOL, "hip centre stays fixed during hip-pitch sweep",
              detail=f"max drift {max(hip_moves):.2e} m")

    # ----- D. hip_roll + ---------------------------------------------- #
    print("\n== D. hip_roll positive -> abduction (+Y) ==")
    hr = chain["l_hip_roll"]
    qs = _linspace(0.0, min(hr.upper, 0.5), SWEEP_N)
    ys = [_sole(spec, {"l_hip_roll": a})[1] for a in qs]
    rep.check(_is_increasing(ys, MONO_TOL), "sole moves outward (+Y) monotonically as hip abducts",
              detail=f"y: {ys[0]:.4f} -> {ys[-1]:.4f}")

    # ----- E. hip_yaw + ---------------------------------------------- #
    print("\n== E. hip_yaw positive -> foot heading rotates toward +Y ==")
    hy = chain["l_hip_yaw"]
    qs = _linspace(0.0, min(hy.upper, 0.6), SWEEP_N)
    fwd_y = []
    for a in qs:
        tf = lm.forward_kinematics(spec, {"l_hip_yaw": a})
        r, _ = tf["l_foot"]
        fwd_y.append(_col(r, 0)[1])   # +Y component of the foot-frame +X axis
    rep.check(_is_increasing(fwd_y, MONO_TOL),
              "foot forward-axis gains +Y component monotonically as leg yaws",
              detail=f"fwd.y: {fwd_y[0]:.4f} -> {fwd_y[-1]:.4f}")

    # ----- F. ankle_pitch + ------------------------------------------ #
    print("\n== F. ankle_pitch positive -> toe drops (plantarflexion) ==")
    ap = chain["l_ankle_pitch"]
    qs = _linspace(0.0, min(ap.upper, 0.6), SWEEP_N)
    up_x, ankle_moves = [], []
    for a in qs:
        tf = lm.forward_kinematics(spec, {"l_ankle_pitch": a})
        r, _ = tf["l_foot"]
        up_x.append(_col(r, 2)[0])   # +X component of the foot-frame +Z axis
        ankle_moves.append(lm.vec_norm(lm.vec_sub(lm.link_origin(tf, "l_ankle_link"), ankle_c0)))
    rep.check(_is_increasing(up_x, MONO_TOL),
              "foot up-axis tips forward (+X) monotonically as ankle plantarflexes",
              detail=f"up.x: {up_x[0]:.4f} -> {up_x[-1]:.4f}")
    rep.check(max(ankle_moves) <= POS_TOL, "ankle joint centre stays fixed during ankle sweep",
              detail=f"max drift {max(ankle_moves):.2e} m")

    # ----- G. reach bound ------------------------------------------------- #
    print("\n== G. reach bound: hip->ankle distance <= L_thigh + L_shin ==")
    rng = random.Random(20260828)
    max_reach = 0.0
    worst = None
    bound = Lt + Ls
    knee0_reach = []
    for _ in range(N_RANDOM):
        q = {name: rng.uniform(jm.lower, jm.upper) for name, jm in chain.items()}
        tf = lm.forward_kinematics(spec, q)
        d = lm.vec_norm(lm.vec_sub(lm.link_origin(tf, "l_ankle_link"),
                                   lm.link_origin(tf, "l_hip_yaw_link")))
        if d > max_reach:
            max_reach, worst = d, q
    rep.check(max_reach <= bound + REACH_TOL,
              "no sampled pose stretches the leg past bone length",
              detail=f"max reach {max_reach:.6f} m <= {bound:.6f} m")

    for _ in range(200):
        q = {name: rng.uniform(jm.lower, jm.upper) for name, jm in chain.items()}
        q["l_knee_pitch"] = 0.0
        tf = lm.forward_kinematics(spec, q)
        knee0_reach.append(lm.vec_norm(lm.vec_sub(lm.link_origin(tf, "l_ankle_link"),
                                                  lm.link_origin(tf, "l_hip_yaw_link"))))
    rep.check(all(abs(d - bound) <= 1e-9 for d in knee0_reach),
              "with knee at 0 the leg is exactly fully extended",
              detail=f"reach in [{min(knee0_reach):.6f}, {max(knee0_reach):.6f}] m")

    # ----- summary + a small reference table ----------------------------- #
    print("\n== reference: sole position at a few named poses ==")
    poses = dict(lm.reference_poses(spec))
    poses.setdefault("knee 60 deg", {"l_knee_pitch": math.radians(60)})
    poses.setdefault("hip_pitch -30 deg (forward)", {"l_hip_pitch": math.radians(-30)})
    for label, q in poses.items():
        s = _sole(spec, q)
        print(f"    {label:<24} sole = ({s[0]:+.4f}, {s[1]:+.4f}, {s[2]:+.4f}) m")

    print("\n" + "=" * 60)
    if rep.failures:
        print(f"RESULT: FAIL  ({len(rep.failures)}/{rep.n} checks failed)")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print(f"RESULT: PASS  ({rep.n}/{rep.n} checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else None))
