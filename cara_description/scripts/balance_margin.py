#!/usr/bin/env python3
"""U9 diagnostic -- *why* is the single-support balance envelope small?

U9 (`scripts/single_support.py`) is MET: Cara holds one foot for 5 s and recovers
a small lateral push (~1 N*100ms toward the swing foot, ~3 N toward the stance
foot).  The envelope is small.  This script explains the cause with numbers
instead of asserting it, and it does NOT change the controller or the model.

It gets into the same held single-support state, then reports:

  1. the held state -- stance Fz, COM height, inverted-pendulum omega, and where
     the center of pressure (CoP) actually sits inside the little foot;
  2. the STATIC CoP moment budget -- Fz * foot-half-width is all the restoring
     moment the ankle can make before the foot rolls onto its edge, and how much
     of it is already spent just holding the pose;
  3. a first-order CAPTURE-POINT prediction of the largest lateral impulse she
     can catch in each direction (J_max = margin * m * omega);
  4. a VALIDATION sweep -- the real controller, fine push steps -- and how well
     the geometric prediction matches the simulated fall threshold;
  5. a foot-half-width SENSITIVITY table (analytic) -- how much wider feet buy;
  6. the ACTUATOR headroom in the held pose (the swing hip_roll bottleneck).

Everything here is measurement + first-order mechanics.  The capture-point
numbers ignore the ankle actively pulling back during the pulse and the 2.5 s
recovery window, so treat them as "can she even catch it", not a tight bound --
step 4 is the real check.  All masses / gains / friction / foot size are
PROVISIONAL.

Requires `mujoco` (brings numpy).  Prints SKIPPED / exits 0 without it.

Usage:
    python3 balance_margin.py                       # full body, stance r_ (left foot balances)
    python3 balance_margin.py --stance l_
    python3 balance_margin.py config/cara_lower_body.yaml
    python3 balance_margin.py --json baselines/full_body_balance_margin.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import leg_model as lm
import weight_shift as wsh

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf"))
DEFAULT_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_full_body.yaml"))

SIDE = {"l_": +1.0, "r_": -1.0}
OTHER = {"l_": "r_", "r_": "l_"}


def run(config, stance_sel, json_path):
    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("SKIPPED: mujoco is not installed (pip install mujoco)")
        return 0

    import generate_mjcf

    spec = lm.load_spec(config)
    model_name = spec["meta"]["name"]
    xml = generate_mjcf.build_mjcf(spec, dynamic=True)
    on_disk = os.path.join(_MJCF_DIR, model_name + "_dynamic.xml")
    if os.path.exists(on_disk):
        with open(on_disk, encoding="utf-8") as fh:
            if fh.read() != xml:
                print(f"WARNING: {on_disk} is stale -- fresh render used here\n")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    g = lm.analysis_gravity(spec)

    ss = (spec.get("analysis", {}) or {}).get("single_support", {}) or {}
    if not ss:
        print("this config has no analysis.single_support block")
        return 2
    base_pose = ss.get("base_pose", "stand_nominal")
    com_target = float(ss.get("com_target", 0.028))
    lift_h = float(ss.get("lift_height", 0.008))
    ramp = float(ss.get("ramp_seconds", 4.0))
    settle = float(ss.get("settle_seconds", 1.5))
    CG = float(ss.get("clearance_gain", 0.012))
    bal = ss.get("balance", {}) or {}
    KPA = float(bal.get("kp_ankle_roll", 50.0))
    KDA = float(bal.get("kd_ankle_roll", 10.0))
    KPH = float(bal.get("kp_hip_roll", 15.0))
    PUSH_DUR = float((ss.get("disturbance", {}) or {}).get("push_duration", 0.10))
    RECOVER_TILT = math.radians(float((ss.get("accept", {}) or {}).get("recover_tilt_deg", 25.0)))
    RECOVER_S = float((ss.get("accept", {}) or {}).get("recover_seconds", 2.5))

    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)[base_pose]
    nominal_ctrl = [float(base_cfg.get(n, 0.0)) for n in jn]

    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose)
    foot_gid = {"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")}
    floor_gid = gid("floor")
    pelvis_bid = bid(spec["frame_conventions"]["base_frame"])
    forcerng = np.array([model.actuator_forcerange[aid(n)][1] for n in jn])
    m_total = float(sum(model.body_mass))
    total_weight = m_total * g

    table = wsh.build_ik_table(spec, base_cfg, com_target + 0.03, 61)
    sagittal = {s: [s + "hip_pitch", s + "knee_pitch", s + "ankle_pitch"] for s in ("l_", "r_")}

    # --- contact helpers ---------------------------------------------------- #
    def stance_contacts(fg):
        """[(x, y, fz), ...] world contact points carrying vertical load."""
        out = []
        for i in range(data.ncon):
            c = data.contact[i]
            if not (floor_gid in (c.geom1, c.geom2) and fg in (c.geom1, c.geom2)):
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f6)
            fr = c.frame
            fz = fr[2] * f6[0] + fr[5] * f6[1] + fr[8] * f6[2]
            out.append((float(c.pos[0]), float(c.pos[1]), float(fz)))
        return out

    def foot_normal_force(fg):
        return sum(fz for _, _, fz in stance_contacts(fg))

    # --- controller (mirrors single_support.py Balancer) ------------------- #
    def build_swing_table(unld, com_cfg, n=25):
        free = sagittal[unld]
        q0 = {**base_cfg, **com_cfg}
        tf0 = lm.forward_kinematics(spec, q0)
        sw = lm.frame_world_position(spec, tf0, unld + "foot_sole_center")
        rot = tf0[unld + "foot"][0]
        rows, q = [], dict(q0)
        for i in range(n):
            c = 0.030 * i / (n - 1)
            sol, _r = lm.leg_ik(spec, unld, unld + "foot_sole_center",
                                (sw[0], sw[1], sw[2] + c), rot, q,
                                free_joints=free, task_rows=[2, 4], iters=200)
            q = {**q, **sol}
            rows.append((c, [sol[j] for j in free]))
        return rows

    def swing_lookup(rows, c):
        if c <= rows[0][0]:
            return rows[0][1]
        if c >= rows[-1][0]:
            return rows[-1][1]
        for i in range(len(rows) - 1):
            if rows[i][0] <= c <= rows[i + 1][0]:
                a = (c - rows[i][0]) / (rows[i + 1][0] - rows[i][0])
                return [rows[i][1][k] * (1 - a) + rows[i + 1][1][k] * a
                        for k in range(len(rows[i][1]))]
        return rows[-1][1]

    class Balancer:
        def __init__(self, stance):
            self.stance = stance
            self.unld = OTHER[stance]
            self.a_roll = stance + "ankle_roll"
            self.h_roll = stance + "hip_roll"
            self.sf_sign = -SIDE[stance]
            _, qt_com = wsh.table_lookup(table, SIDE[stance] * com_target)
            self.com_cfg = {jn[i]: qt_com[i] for i in range(len(jn))}
            self.rows = build_swing_table(self.unld, self.com_cfg)
            self.cmd = dict(self.com_cfg)
            self.ccmd = 0.0
            self.ref_ey = None
            self.prev_comy = 0.0
            self.swing_z0 = 0.0

        def enter(self):
            mujoco.mj_resetDataKeyframe(model, data, kid)
            for _ in range(int(settle / dt)):
                data.ctrl[:] = nominal_ctrl
                mujoco.mj_step(model, data)
            self.swing_z0 = float(data.geom_xpos[foot_gid[self.unld]][2])
            for s in range(int(ramp / dt)):
                _, qt = wsh.table_lookup(table, SIDE[self.stance] * com_target
                                         * wsh.smoothstep(s * dt / ramp))
                data.ctrl[:] = qt
                mujoco.mj_step(model, data)
            self.prev_comy = float(data.subtree_com[0][1])
            for s in range(int(ramp / dt)):
                self.step(lift_h * wsh.smoothstep(s * dt / ramp))

        def step(self, clear_target=None, push=0.0):
            if clear_target is None:
                clear_target = lift_h
            wc = float(data.geom_xpos[foot_gid[self.unld]][2]) - self.swing_z0
            self.ccmd = min(0.030, max(0.0, self.ccmd + CG * (clear_target - wc)))
            for n, v in zip(sagittal[self.unld], swing_lookup(self.rows, self.ccmd)):
                self.cmd[n] = v
            com = data.subtree_com[0]
            sf = data.geom_xpos[foot_gid[self.stance]]
            ey = float(com[1] - sf[1])
            if self.ref_ey is None:
                self.ref_ey = ey
            drift = ey - self.ref_ey
            vy = (float(com[1]) - self.prev_comy) / dt
            self.prev_comy = float(com[1])
            c2 = dict(self.cmd)
            c2[self.a_roll] = self.com_cfg[self.a_roll] + self.sf_sign * (KPA * drift + KDA * vy)
            c2[self.h_roll] = self.com_cfg[self.h_roll] + self.sf_sign * KPH * drift
            data.ctrl[:] = [c2[n] for n in jn]
            data.xfrc_applied[pelvis_bid][1] = push
            mujoco.mj_step(model, data)
            data.xfrc_applied[pelvis_bid][1] = 0.0
            roll, pitch, _ = wsh.quat_rpy(data.qpos[3:7])
            tau = np.array([data.actuator_force[aid(n)] for n in jn])
            return {"drift": drift, "clear": wc, "tilt": max(abs(roll), abs(pitch)),
                    "swing_fz": foot_normal_force(foot_gid[self.unld]),
                    "tau_frac": float(np.max(np.abs(tau) / forcerng))}

    # --- held-state measurement ------------------------------------------- #
    def held_state(stance):
        b = Balancer(stance)
        b.enter()
        fg = foot_gid[stance]
        acc = {"fz": [], "copx": [], "copy": [], "comx": [], "comy": [], "comz": [],
               "vy": [], "tau": np.zeros(len(jn))}
        prev_com = np.array(data.subtree_com[0])
        for k in range(int(2.0 / dt)):
            b.step()
            com = np.array(data.subtree_com[0])
            if k < int(1.0 / dt):                 # let it settle first
                prev_com = com
                continue
            cons = stance_contacts(fg)
            fz = sum(c[2] for c in cons)
            if fz < 1.0:
                continue
            acc["fz"].append(fz)
            acc["copx"].append(sum(c[0] * c[2] for c in cons) / fz)
            acc["copy"].append(sum(c[1] * c[2] for c in cons) / fz)
            acc["comx"].append(float(com[0])); acc["comy"].append(float(com[1]))
            acc["comz"].append(float(com[2]))
            acc["vy"].append(float((com[1] - prev_com[1]) / dt))
            acc["tau"] = np.maximum(acc["tau"], np.abs([data.actuator_force[aid(n)] for n in jn]))
            prev_com = com

        s = model.geom_size[fg]                   # box half-extents (x fwd, y lat, z)
        p = data.geom_xpos[fg]
        sole_z = float(p[2] - s[2])
        mean = lambda k: float(np.mean(acc[k]))
        return {
            "stance": stance,
            "Fz": mean("fz"),
            "cop_x": mean("copx"), "cop_y": mean("copy"),
            "com_x": mean("comx"), "com_y": mean("comy"), "com_z": mean("comz"),
            "com_vy": mean("vy"),
            "foot_cx": float(p[0]), "foot_cy": float(p[1]),
            "half_x": float(s[0]), "half_y": float(s[1]),
            "com_height": mean("comz") - sole_z,
            "tau": acc["tau"],
        }

    # ==================================================================== #
    print(f"U9 diagnostic -- single-support balance margin   {model_name}   "
          f"{m_total:.2f} kg ({total_weight:.1f} N)")
    print(f"controller: COM-y feedback kp/kd ankle {KPA:.0f}/{KDA:.0f}, kp hip {KPH:.0f}  "
          f"(unchanged from analysis.single_support)")

    stances = ("l_", "r_") if stance_sel is None else (stance_sel,)
    hs = {st: held_state(st) for st in stances}
    st = stances[-1] if stance_sel else "r_"
    h = hs[st]
    to_swing = -SIDE[st]                          # +y unit direction toward the swing foot

    # this diagnostic only makes sense from a real single-support hold.  The U9
    # controller is full-body tuned; the lower-body model never gets there (its
    # COM stays far outside the stance foot).  Bail with a clear message.
    lat_off = to_swing * (h["com_y"] - h["foot_cy"])
    if h["Fz"] < 0.8 * total_weight or abs(lat_off) > h["half_y"]:
        print(f"\nNOT A VALID SINGLE-SUPPORT HOLD for {model_name}: stance carries only "
              f"{100*h['Fz']/total_weight:.0f}% of body weight and the COM projects "
              f"{1e3*lat_off:+.0f} mm from the foot centre (foot half-width {1e3*h['half_y']:.1f} mm).")
        print("U9's balance controller is tuned for the full body (`cara_full_body.yaml`); "
              "the lower-body model does not reach single support with it.  Nothing to diagnose here.")
        return 0

    omega = math.sqrt(g / h["com_height"])
    tau_p = 1.0 / omega

    # lateral edges of the stance foot, and how far the COM / CoP sit from each
    y_edge_swing = h["foot_cy"] + to_swing * h["half_y"]
    y_edge_stance = h["foot_cy"] - to_swing * h["half_y"]
    com_margin_swing = to_swing * (y_edge_swing - h["com_y"])
    com_margin_stance = -to_swing * (y_edge_stance - h["com_y"])
    cop_margin_swing = to_swing * (y_edge_swing - h["cop_y"])
    cop_margin_stance = -to_swing * (y_edge_stance - h["cop_y"])
    cop_moment_budget = h["Fz"] * h["half_y"]
    cop_moment_spent = h["Fz"] * abs(h["cop_y"] - h["foot_cy"])

    print("\n1) HELD SINGLE-SUPPORT STATE"
          + ("  (both sides -- expect a match)" if len(stances) > 1 else f"  (stance {st[:-1]})"))
    if len(stances) > 1:
        for s2 in stances:
            hh = hs[s2]
            print(f"   stance {s2[:-1]}:  Fz {hh['Fz']:.1f} N ({100*hh['Fz']/total_weight:.0f}% wt)   "
                  f"COM height {1e3*hh['com_height']:.0f} mm   "
                  f"CoP {1e3*(hh['cop_y']-hh['foot_cy']):+.1f} mm from foot centre (lateral)")
    print(f"   stance Fz .................. {h['Fz']:.1f} N   ({100*h['Fz']/total_weight:.0f}% of body weight on one foot)")
    print(f"   COM height above the sole .. {1e3*h['com_height']:.0f} mm")
    print(f"   inverted-pendulum omega .... {omega:.2f} rad/s   (time constant {1e3*tau_p:.0f} ms)")
    print(f"   foot sole half-width (lat).. {1e3*h['half_y']:.1f} mm   (half-length fwd {1e3*h['half_x']:.1f} mm)")
    print(f"   residual COM-y speed ....... {1e3*abs(h['com_vy']):.2f} mm/s   (settled)")
    print(f"   stance foot centre ........ {1e3*to_swing*h['foot_cy']:+.1f} mm from the body midline "
          f"(toward stance)")
    print(f"   COM lateral position ...... {1e3*to_swing*(h['com_y']-h['foot_cy']):+.1f} mm from the foot centre "
          f"toward the swing side  (COM sits {1e3*to_swing*h['com_y']:+.1f} mm off the midline; "
          f"the foot centre is further out)")
    print(f"   CoP lateral position ...... {1e3*to_swing*(h['cop_y']-h['foot_cy']):+.1f} mm from the foot centre "
          f"toward the swing side")

    print("\n2) STATIC CoP MOMENT BUDGET (ankle roll)")
    print(f"   max restoring moment  Fz * half-width = {h['Fz']:.1f} * {1e3*h['half_y']:.1f}mm = {cop_moment_budget:.2f} N*m")
    print(f"   spent holding the pose ............... {cop_moment_spent:.2f} N*m "
          f"({100*cop_moment_spent/cop_moment_budget:.0f}% of the budget)")
    print(f"   lateral room left toward the SWING foot .. {1e3*cop_margin_swing:.1f} mm  "
          f"(CoP)   {1e3*com_margin_swing:.1f} mm (COM projection)")
    print(f"   lateral room left toward the STANCE foot . {1e3*cop_margin_stance:.1f} mm  "
          f"(CoP)   {1e3*com_margin_stance:.1f} mm (COM projection)")
    if com_margin_swing < com_margin_stance:
        print(f"   -> the swing leg parks the COM near the INNER (swing-side) edge: only "
              f"{1e3*com_margin_swing:.1f} mm of lateral margin that way.")

    print("\n3) CAPTURE-POINT PREDICTION  (J_max = margin * m * omega,  first-order)")
    def predict(margin):
        j = margin * m_total * omega
        return j, j / PUSH_DUR
    j_sw, f_sw = predict(com_margin_swing)
    j_st, f_st = predict(com_margin_stance)
    print(f"   toward the SWING foot  ... J_max {j_sw:.3f} N*s  ->  {f_sw:.1f} N over {1e3*PUSH_DUR:.0f} ms")
    print(f"   toward the STANCE foot  .. J_max {j_st:.3f} N*s  ->  {f_st:.1f} N over {1e3*PUSH_DUR:.0f} ms")

    # --- 4) validation sweep -------------------------------------------- #
    def push_test(b, magnitude, direction):
        for _ in range(int(1.0 / dt)):
            b.step()
        for _ in range(int(PUSH_DUR / dt)):
            b.step(push=magnitude * direction * (-SIDE[b.stance]))
        pk = 0.0
        why = ""
        for _ in range(int(RECOVER_S / dt)):
            m = b.step()
            pk = max(pk, m["tilt"])
            if not why and m["tilt"] > RECOVER_TILT:
                why = "toppled"
            if not why and m["swing_fz"] > 3.0:
                why = "swing foot down"
        return (why == ""), math.degrees(pk), why

    print("\n4) VALIDATION -- real controller, fine push steps "
          f"(x {1e3*PUSH_DUR:.0f} ms), stance {st[:-1]}")
    sweeps = {"SWING": ([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], +1.0),
              "STANCE": ([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], -1.0)}
    sim_cliff = {}
    sim_why = {}
    for label, (mags, sgn) in sweeps.items():
        b = Balancer(st)
        b.enter()
        last_ok = 0.0
        fail_why = ""
        for mag in mags:
            ok, pk, why = push_test(b, mag, sgn)
            pk_s = f"{pk:>5.1f}°" if pk < 90 else "tumbled"
            print(f"   {mag:>4.2f} N toward {label:<6} -> peak tilt {pk_s:>7}   "
                  f"{'recovered' if ok else 'FELL (' + why + ')'}")
            if ok:
                last_ok = mag
            else:
                fail_why = why
                break
        sim_cliff[label] = last_ok
        sim_why[label] = fail_why

    pred = {"SWING": f_sw, "STANCE": f_st}
    print("\n   prediction vs simulation:")
    for label in ("SWING", "STANCE"):
        p, s2 = pred[label], sim_cliff[label]
        rel = (s2 - p) / p * 100.0 if p else 0.0
        print(f"     toward {label:<6}: predicted {p:>4.1f} N,  simulated ~{s2:>4.1f} N   ({rel:+.0f}%)"
              + (f"   [fell: {sim_why[label]}]" if sim_why[label] else ""))
    print("   toward SWING the wall is geometric -- the CoP hits the inner foot edge; the")
    print("   first-order estimate is ~1.5-2x high because it ignores the ankle pulling")
    print("   back during the pulse.  toward STANCE she has ~38 mm of sole but falls far")
    print("   below that: the failure there is the RECOVERY overshoot swinging her back")
    print("   past the inner edge / dropping the lifted foot -- a controller limit, not")
    print("   the CoP wall.  Either way the binding constraint is the 6.5 mm swing-side gap.")

    # --- 5) foot half-width sensitivity (analytic) --------------------- #
    print("\n5) FOOT HALF-WIDTH SENSITIVITY  (analytic first-order -- widen the sole, "
          "hold everything else)")
    print(f"   {'half-width':>11} {'swing margin':>13} {'J_max swing':>12} {'~force x100ms':>14}")
    base_hy = h["half_y"]
    for hy in (0.015, 0.0225, 0.030, 0.040, 0.050, 0.060):
        msw = com_margin_swing + (hy - base_hy)
        if msw <= 0:
            print(f"   {1e3*hy:>9.1f} mm {1e3*msw:>11.1f} mm {'--':>12} {'(COM outside)':>14}")
            continue
        j = msw * m_total * omega
        tag = "  <- current" if abs(hy - base_hy) < 1e-6 else ""
        print(f"   {1e3*hy:>9.1f} mm {1e3*msw:>11.1f} mm {j:>10.3f} N*s {j/PUSH_DUR:>12.1f} N{tag}")

    # --- 6) actuator headroom in the held pose ------------------------ #
    print("\n6) ACTUATOR HEADROOM IN THE HELD POSE  (separate limit -- servo sizing, not balance)")
    order = np.argsort(-(h["tau"] / forcerng))
    for i in order[:4]:
        print(f"   {jn[i]:>14}: {h['tau'][i]:>5.2f} / {forcerng[i]:.1f} N*m   "
              f"({100*h['tau'][i]/forcerng[i]:.0f}%)")

    # --- summary ----------------------------------------------------- #
    print("\n" + "=" * 74)
    print("CAUSE.  In single support the whole body pivots about one small ankle.  The")
    print(f"restoring moment is capped at Fz*half-width = {cop_moment_budget:.2f} N*m; past that the")
    print("foot rolls onto its edge and no gain helps.  The swing leg's mass parks the")
    print(f"COM ~{1e3*abs(h['com_y']-h['foot_cy']):.0f} mm toward the INNER edge, leaving only "
          f"~{1e3*com_margin_swing:.0f} mm of lateral")
    print(f"margin that way -> a ~{sim_cliff['SWING']:.1f} N*100ms pulse is all she can catch toward the")
    print(f"swing foot ({sim_cliff['STANCE']:.1f} N toward the stance foot, where there is more sole).")
    print("Fixes are structural, not control: a wider/longer foot (step 5), tucking the")
    print("lifted foot toward the midline, an arm/trunk angular-momentum strategy, or a")
    print("protective step (U10).")

    if json_path:
        out = {"model": model_name, "total_mass_kg": m_total, "stance": st,
               "held": {k: (v if not hasattr(v, "tolist") else v.tolist())
                        for k, v in h.items() if k != "tau"},
               "omega": omega, "cop_moment_budget_Nm": cop_moment_budget,
               "com_margin_swing_m": com_margin_swing, "com_margin_stance_m": com_margin_stance,
               "predicted_force_swing_N": f_sw, "predicted_force_stance_N": f_st,
               "simulated_cliff_swing_N": sim_cliff["SWING"],
               "simulated_cliff_stance_N": sim_cliff["STANCE"],
               "push_duration_s": PUSH_DUR}
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nsummary -> {json_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--stance", default="r_", choices=("l_", "r_", "both"),
                    help="which foot is the stance (default r_ -> left foot balances; "
                         "'both' measures the held state on each side)")
    ap.add_argument("--json", default=None, help="write the diagnostic summary here")
    args = ap.parse_args(argv)
    return run(args.config, None if args.stance == "both" else args.stance, args.json)


if __name__ == "__main__":
    sys.exit(main())
