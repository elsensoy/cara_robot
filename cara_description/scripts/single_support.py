#!/usr/bin/env python3
"""U9 -- single-support balance.

U8 got Cara onto one foot for ~1.5 s with a minimal pelvis-roll trim.  U9
replaces that trim with a proper **COM-feedback balance controller** and asks:

  1. can she hold single support *indefinitely* (tested to `hold_seconds`, 5 s)
     with the whole-body COM held over the stance foot?
  2. how big a lateral disturbance can she reject without the free foot touching
     down or toppling?

The controller (transparent, hand-tuned provisional gains in
`analysis.single_support.balance`): a PD on the whole-body COM-y drift relative
to the stance foot, trimming the stance `ankle_roll` target, plus a P term on
the stance `hip_roll` (ankle + hip strategy).  The swing-foot clearance stays on
the U8 closed loop.  No stepping, no RL, no gait.

Disturbances are scripted lateral force pulses on the pelvis
(`analysis.single_support.disturbance`), swept in magnitude, both directions.

Failure cases are reported, not hidden.

Requires `mujoco` (brings numpy).  Prints SKIPPED / exits 0 without it.

Usage:
    python3 single_support.py                       # full body, both sides
    python3 single_support.py config/cara_lower_body.yaml
    python3 single_support.py --view --stance r_    # watch the left foot balance
    python3 single_support.py --json baselines/full_body_single_support.json
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


def run(config, view, view_stance, json_path, baseline_path):
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
                print(f"WARNING: {on_disk} is stale -- run "
                      f"`generate_mjcf.py --dynamic {config or ''}` (fresh render used here)\n")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    ss = (spec.get("analysis", {}) or {}).get("single_support", {}) or {}
    if not ss:
        print("this config has no analysis.single_support block")
        return 2
    base_pose = ss.get("base_pose", "stand_nominal")
    com_target = float(ss.get("com_target", 0.028))
    lift_h = float(ss.get("lift_height", 0.008))
    ramp = float(ss.get("ramp_seconds", 4.0))
    hold = float(ss.get("hold_seconds", 5.0))
    settle = float(ss.get("settle_seconds", 1.5))
    CG = float(ss.get("clearance_gain", 0.012))
    bal = ss.get("balance", {}) or {}
    KPA = float(bal.get("kp_ankle_roll", 50.0))
    KDA = float(bal.get("kd_ankle_roll", 10.0))
    KPH = float(bal.get("kp_hip_roll", 15.0))
    dist = ss.get("disturbance", {}) or {}
    PUSH_DUR = float(dist.get("push_duration", 0.10))
    SWEEP_SWING = [float(x) for x in dist.get("toward_swing", [1.0, 2.0, 3.0, 4.0])]
    SWEEP_STANCE = [float(x) for x in dist.get("toward_stance", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
    acc = ss.get("accept", {}) or {}
    MAX_DRIFT = float(acc.get("max_com_drift", 0.020))
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 8.0)))
    MIN_CLEAR = float(acc.get("min_swing_clearance", 0.003))
    MAX_TQ = float(acc.get("max_torque_frac", 1.0))
    RECOVER_TILT = math.radians(float(acc.get("recover_tilt_deg", 25.0)))
    RECOVER_S = float(acc.get("recover_seconds", 2.5))

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
    total_weight = float(sum(model.body_mass)) * lm.analysis_gravity(spec)

    table = wsh.build_ik_table(spec, base_cfg, com_target + 0.03, 61)
    sagittal = {s: [s + "hip_pitch", s + "knee_pitch", s + "ankle_pitch"] for s in ("l_", "r_")}

    def foot_normal_force(fg):
        fz = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if floor_gid in (c.geom1, c.geom2) and fg in (c.geom1, c.geom2):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f6)
                fr = c.frame
                fz += fr[2] * f6[0] + fr[5] * f6[1] + fr[8] * f6[2]
        return fz

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
        """One single-support episode: enter, then step() with COM feedback."""

        def __init__(self, stance):
            self.stance = stance
            self.unld = OTHER[stance]
            self.a_roll = stance + "ankle_roll"
            self.h_roll = stance + "hip_roll"
            self.sf_sign = -SIDE[stance]        # ankle_roll axis flips with the mirror
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
            for s in range(int(ramp / dt)):        # ramp COM onto the stance foot
                _, qt = wsh.table_lookup(table, SIDE[self.stance] * com_target
                                         * wsh.smoothstep(s * dt / ramp))
                data.ctrl[:] = qt
                mujoco.mj_step(model, data)
            self.prev_comy = float(data.subtree_com[0][1])
            for s in range(int(ramp / dt)):        # raise the free foot to lift_h
                self.step(lift_h * wsh.smoothstep(s * dt / ramp))

        def step(self, clear_target=lift_h, push=0.0):
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
                    "tq_frac": float(np.max(np.abs(tau) / forcerng))}

    # ------------------------------------------------------------------ #
    def hold_test(b):
        drift = tilt = tq = 0.0
        clear_min = 1e9
        for _ in range(int(hold / dt)):
            m = b.step()
            drift = max(drift, abs(m["drift"]))
            tilt = max(tilt, m["tilt"])
            tq = max(tq, m["tq_frac"])
            clear_min = min(clear_min, m["clear"])
        ok = (drift < MAX_DRIFT and tilt < MAX_TILT and clear_min > MIN_CLEAR
              and tq <= MAX_TQ and b.step()["swing_fz"] < 2.0)
        return ok, {"com_drift_mm": drift * 1e3, "tilt_deg": math.degrees(tilt),
                    "clear_mm": clear_min * 1e3, "torque_pct": tq * 100.0}

    def push_test(b, magnitude, direction):
        for _ in range(int(1.0 / dt)):        # re-settle
            b.step()
        for _ in range(int(PUSH_DUR / dt)):   # the pulse (direction: +1 swing, -1 stance)
            b.step(push=magnitude * direction * (-SIDE[b.stance]))
        pk_tilt = 0.0
        fell = False
        for _ in range(int(RECOVER_S / dt)):
            m = b.step()
            pk_tilt = max(pk_tilt, m["tilt"])
            if m["tilt"] > RECOVER_TILT or m["swing_fz"] > 3.0:
                fell = True
        return (not fell), math.degrees(pk_tilt)

    # ================================================================== #
    if view:
        import time
        try:
            import mujoco.viewer
        except ImportError:
            print("error: mujoco.viewer unavailable (needs a display)", file=sys.stderr)
            return 2
        floor_z = float(model.geom_pos[floor_gid][2])
        b = Balancer(view_stance)
        push_mag = 1.0 * (-SIDE[view_stance])       # gentle, alternating, recoverable
        print(f"\nviewer: {view_stance[:-1]} foot balances; a gentle ~1 N lateral pulse every ~4 s "
              f"(alternating). green = stance foot centre, orange = COM. close the window to stop.")
        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running():
                b = Balancer(view_stance)
                b.enter()
                k = 0
                while v.is_running():
                    push = 0.0
                    phase = k % int(4.0 / dt)
                    if phase < int(PUSH_DUR / dt):
                        push = push_mag * (1.0 if (k // int(4.0 / dt)) % 2 == 0 else -1.0)
                    m = b.step(push=push)
                    k += 1
                    com = data.subtree_com[0]
                    sf = data.geom_xpos[foot_gid[view_stance]]
                    v.user_scn.ngeom = 0
                    for pos, rgba in (((float(sf[0]), float(sf[1]), floor_z + 0.001), (0.2, 0.8, 0.3, 1)),
                                      ((float(com[0]), float(com[1]), floor_z + 0.002), (0.95, 0.55, 0.15, 1))):
                        g = v.user_scn.geoms[v.user_scn.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                                            list(pos), np.eye(3).flatten(), list(rgba))
                        v.user_scn.ngeom += 1
                    v.sync()
                    time.sleep(dt)
                    if m["tilt"] > math.radians(35) or m["swing_fz"] > 5.0:
                        break                       # fell -- restart the loop
        return 0

    print(f"Single-support balance (U9)  base '{base_pose}'  {model_name}")
    print(f"{sum(model.body_mass):.2f} kg ({total_weight:.1f} N)  |  COM-feedback controller "
          f"kp/kd ankle {KPA:.0f}/{KDA:.0f}, kp hip {KPH:.0f}  |  hold {hold:.0f} s, then lateral pushes")

    results = {"model": model_name, "total_mass_kg": float(sum(model.body_mass)),
               "hold_seconds": hold, "balance_gains": [KPA, KDA, KPH], "sides": {}}
    hold_ok_all = True
    env_ok_all = True

    for stance in ("l_", "r_"):
        b = Balancer(stance)
        b.enter()
        h_ok, h = hold_test(b)
        print(f"\nstance {stance[:-1]}:  {hold:.0f}s hold -> COM drift {h['com_drift_mm']:.1f} mm, "
              f"tilt {h['tilt_deg']:.2f}°, free foot {h['clear_mm']:.1f} mm clear, "
              f"torque {h['torque_pct']:.0f}%  [{'OK' if h_ok else 'FAIL'}]")
        hold_ok_all &= h_ok

        swing_max = 0.0
        for mag in SWEEP_SWING:
            ok, pk = push_test(b, mag, +1.0)
            print(f"   push {mag:.1f} N / {PUSH_DUR*1e3:.0f} ms toward SWING  -> peak tilt {pk:.1f}°  "
                  f"{'recovered' if ok else 'FELL'}")
            if ok:
                swing_max = mag
            else:
                break
        stance_max = 0.0
        b2 = Balancer(stance)
        b2.enter()
        hold_test(b2)
        for mag in SWEEP_STANCE:
            ok, pk = push_test(b2, mag, -1.0)
            print(f"   push {mag:.1f} N / {PUSH_DUR*1e3:.0f} ms toward STANCE -> peak tilt {pk:.1f}°  "
                  f"{'recovered' if ok else 'FELL'}")
            if ok:
                stance_max = mag
            else:
                break

        env_ok = swing_max > 0.0 and stance_max > 0.0
        env_ok_all &= env_ok
        results["sides"][stance] = {
            "hold_ok": h_ok, **h,
            "recover_toward_swing_N": swing_max, "recover_toward_stance_N": stance_max,
            "push_duration_s": PUSH_DUR}
        print(f"   => recoverable lateral impulse: ~{swing_max:.1f} N toward swing, "
              f"~{stance_max:.1f} N toward stance  (x {PUSH_DUR*1e3:.0f} ms)")

    both = hold_ok_all and env_ok_all
    results["milestone_met"] = both

    if (results["sides"]["l_"].get("com_drift_mm") is not None
            and abs(results["sides"]["l_"]["com_drift_mm"]
                    - results["sides"]["r_"]["com_drift_mm"]) < 0.3):
        print("\n(the two stance sides match -- Cara is sagittally symmetric, as expected)")

    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsummary -> {json_path}")
    if baseline_path and os.path.exists(baseline_path):
        base = json.load(open(baseline_path, encoding="utf-8"))
        for s in ("l_", "r_"):
            b0, c0 = base.get("sides", {}).get(s, {}), results["sides"][s]
            if "com_drift_mm" in b0:
                print(f"vs baseline  stance {s[:-1]}: drift {c0['com_drift_mm']:.1f} mm "
                      f"({c0['com_drift_mm']-b0['com_drift_mm']:+.1f}),  swing-push "
                      f"{c0['recover_toward_swing_N']:.1f} N ({c0['recover_toward_swing_N']-b0.get('recover_toward_swing_N',0):+.1f})")

    print("\n" + "=" * 70)
    if both:
        print(f"MILESTONE MET: Cara balances on one foot for {hold:.0f}s with the COM held over "
              f"the stance foot, and rejects a small lateral disturbance, both sides.")
    else:
        print("MILESTONE NOT MET: " + ("the extended hold fails; " if not hold_ok_all else "")
              + ("no lateral disturbance is recovered; " if not env_ok_all else "")
              + "see the rows above.")
    print("  ankle + hip COM feedback on the position PD; no stepping, no RL. "
          "(provisional masses / gains / friction / foot size)")
    return 0 if both else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--view", action="store_true",
                    help="watch the balance + periodic lateral pulses in the MuJoCo viewer")
    ap.add_argument("--stance", default="r_", choices=("l_", "r_"),
                    help="--view: which foot is the stance (default r_ -> left foot balances)")
    ap.add_argument("--json", default=None, help="write the run summary here")
    ap.add_argument("--baseline", default=None, help="compare against this summary JSON")
    args = ap.parse_args(argv)
    return run(args.config, args.view, args.stance, args.json, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
