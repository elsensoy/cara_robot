#!/usr/bin/env python3
"""Quasi-static bilateral weight-shift experiment on the lower-body model.

Milestone question: *can Cara deliberately transfer her weight from one leg to
the other while remaining in controlled double support?*

Neither foot is lifted; there is no RL and no gain tuning.

Pipeline (all transparent):

    com_y_desired(t)            smooth lateral COM target, centre -> foot -> centre
        │  (1-D table inversion, built from the IK below)
        ▼
    pelvis lateral shift        the pelvis displacement that puts the COM there
        │  frontal-plane IK per leg: free = {hip_roll, ankle_roll},
        │  constrain foot y-position + foot roll (soles stay flat & planted)
        ▼
    q_target  (12 joints)  ──►  the EXISTING PD <position> servos  ──►  MuJoCo

Logged per step (also written to CSV with --csv):
  desired & measured COM (x,y,z); COM margin vs the FULL support polygon;
  COM margin vs EACH foot's own polygon; pelvis roll & pitch;
  left/right vertical contact force; per-foot horizontal slip;
  q, qdot, actuator torque (12 each); torque-saturation count.

Validates that shifting toward a foot raises that foot's normal force and
unloads the other, both soles staying planted -- then sweeps the COM-target
magnitude to find how far the shift can go before support margin, contact,
pelvis orientation, slip or actuator limits fail.  Failure cases are reported,
not hidden.

Requires `mujoco` (brings numpy). Prints SKIPPED / exits 0 without it.

Usage:
    python3 weight_shift.py                       # demo run + sweep (headless, logs)
    python3 weight_shift.py --amplitude 0.04 --csv shift.csv
    python3 weight_shift.py --no-sweep
    python3 weight_shift.py --view               # watch the COM-shift loop in the viewer
    python3 weight_shift.py --view --amplitude 0.05   # watch it topple at the limit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import leg_model as lm

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJCF_DIR = os.path.normpath(os.path.join(_HERE, os.pardir, "mjcf"))
DEFAULT_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "cara_lower_body.yaml"))


def smoothstep(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def quat_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def make_trajectory(A, ramp, hold, extra_centre):
    """Waypoints (t_end, com_y_target). Smoothstep between them. Starts at 0."""
    wp = [(0.0, 0.0)]

    def add(dt, v):
        wp.append((wp[-1][0] + dt, v))

    add(extra_centre, 0.0)
    add(ramp, +A); add(hold, +A)
    add(ramp, 0.0); add(hold, 0.0)
    add(ramp, -A); add(hold, -A)
    add(ramp, 0.0); add(extra_centre, 0.0)
    total = wp[-1][0]

    def value(t):
        pt, pv = wp[0]
        for te, v in wp[1:]:
            if t <= te:
                if te <= pt or v == pv:
                    return v
                return pv + (v - pv) * smoothstep((t - pt) / (te - pt))
            pt, pv = te, v
        return wp[-1][1]

    # measurement windows: the final second of the +A and -A holds
    plus_te = wp[3][0]; minus_te = wp[7][0]
    windows = {"+A": (plus_te - 1.0, plus_te), "-A": (minus_te - 1.0, minus_te)}
    return value, total, windows


# --------------------------------------------------------------------------- #
# Feedforward task-space layer: desired COM  ->  joint targets
# --------------------------------------------------------------------------- #
def build_ik_table(spec, base_cfg, py_max, n):
    """[(pelvis_shift, predicted world COM y, q_target[12], ik_residual)]."""
    nfp = lm.nominal_foot_poses(spec, base_cfg)
    jn = lm.actuated_joint_names(spec)
    rows = []
    for i in range(n):
        py = -py_max + 2 * py_max * i / (n - 1)
        q = {k: float(base_cfg.get(k, 0.0)) for k in jn}
        res = 0.0
        for pfx, (sole0, rot0, _ank) in nfp.items():
            target = (sole0[0], sole0[1] - py, sole0[2])
            sol, r = lm.leg_ik(spec, pfx, pfx + "foot_sole_center", target, rot0, q,
                               free_joints=[pfx + "hip_roll", pfx + "ankle_roll"],
                               task_rows=[1, 3])
            q.update(sol)
            res = max(res, r)
        pf_com_y = lm.center_of_mass(spec, q)[1][1]
        rows.append((py, pf_com_y + py, [q[k] for k in jn], res))
    return rows


def table_lookup(table, com_des):
    cs = [r[1] for r in table]           # predicted world COM y, monotone
    if com_des <= cs[0]:
        return table[0][0], table[0][2]
    if com_des >= cs[-1]:
        return table[-1][0], table[-1][2]
    for i in range(len(cs) - 1):
        if cs[i] <= com_des <= cs[i + 1]:
            a = (com_des - cs[i]) / (cs[i + 1] - cs[i])
            py = table[i][0] * (1 - a) + table[i + 1][0] * a
            q = [table[i][2][k] * (1 - a) + table[i + 1][2][k] * a
                 for k in range(len(table[i][2]))]
            return py, q
    return table[-1][0], table[-1][2]


# --------------------------------------------------------------------------- #
def run(config, amplitude, do_sweep, csv_path, verbose, view=False,
        json_path=None, baseline_path=None):
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

    ws = (spec.get("analysis", {}) or {}).get("weight_shift", {}) or {}
    base_pose = ws.get("base_pose", "stand_nominal")
    A = float(amplitude if amplitude is not None else ws.get("amplitude", 0.03))
    ramp = float(ws.get("ramp_seconds", 3.0))
    hold = float(ws.get("hold_seconds", 3.0))
    settle = float(ws.get("settle_seconds", 1.5))
    sweep_vals = [float(x) for x in ws.get("sweep", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])]
    acc = ws.get("accept", {}) or {}
    MIN_MARGIN = float(acc.get("min_support_margin", 0.005))
    MIN_OPP_FRAC = float(acc.get("min_opposite_load_frac", 0.05))
    MAX_TILT = math.radians(float(acc.get("max_pelvis_tilt_deg", 6.0)))
    MAX_SLIP = float(acc.get("max_foot_slip", 0.003))
    MAX_TQ_FRAC = float(acc.get("max_torque_frac", 1.0))

    jn = lm.actuated_joint_names(spec)
    base_cfg = lm.reference_poses(spec)[base_pose]
    nominal_ctrl = [float(base_cfg.get(n, 0.0)) for n in jn]

    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    pelvis_bid = bid(spec["frame_conventions"]["base_frame"])
    foot_gid = {"l_": gid("l_foot_collision"), "r_": gid("r_foot_collision")}
    floor_gid = gid("floor")
    forcerng = np.array([model.actuator_forcerange[aid(n)][1] for n in jn])
    total_weight = float(sum(model.body_mass)) * lm.analysis_gravity(spec)

    py_max = max(sweep_vals + [A]) + 0.03
    table = build_ik_table(spec, base_cfg, py_max, 61)
    ik_res = max(r[3] for r in table)

    # --- per-step helpers ------------------------------------------------- #
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

    def foot_contacts(fg):
        return sum(1 for i in range(data.ncon)
                   if {data.contact[i].geom1, data.contact[i].geom2} == {fg, floor_gid})

    def foot_polygon(fg):
        s = model.geom_size[fg]
        p = data.geom_xpos[fg]
        rmat = data.geom_xmat[fg].reshape(3, 3)
        pts = []
        for ex in (-1, 1):
            for ey in (-1, 1):
                w = p + rmat @ np.array([ex * s[0], ey * s[1], -s[2]])
                pts.append((float(w[0]), float(w[1])))
        return lm.convex_hull_2d(pts)

    def all_foot_contact_points():
        pts = []
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {c.geom1, c.geom2}
            if floor_gid in pair and (pair & set(foot_gid.values())):
                pts.append((float(c.pos[0]), float(c.pos[1])))
        return pts

    # --- one episode ---------------------------------------------------- #
    def episode(traj, total):
        mujoco.mj_resetDataKeyframe(model, data,
                                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose))
        for _ in range(int(settle / dt)):
            data.ctrl[:] = nominal_ctrl
            mujoco.mj_step(model, data)
        foot0 = {p: np.array(data.geom_xpos[foot_gid[p]][:2]) for p in ("l_", "r_")}

        log = {k: [] for k in
               ("t", "com_des", "com_ff", "com_x", "com_y", "com_z",
                "m_full", "m_left", "m_right", "roll", "pitch",
                "Fn_l", "Fn_r", "slip_l", "slip_r", "nc_l", "nc_r", "sat")}
        Q, QD, TAU = [], [], []
        for step in range(int(total / dt)):
            t = step * dt
            cy = traj(t)
            py, qt = table_lookup(table, cy)
            data.ctrl[:] = qt
            mujoco.mj_step(model, data)

            com = data.subtree_com[0]
            tau = np.array([data.actuator_force[aid(n)] for n in jn])
            roll, pitch, _ = quat_rpy(data.qpos[3:7])
            full_poly = lm.convex_hull_2d(all_foot_contact_points())
            log["t"].append(t); log["com_des"].append(cy)
            log["com_ff"].append(cy)   # table is inverted so ff target == desired
            log["com_x"].append(float(com[0])); log["com_y"].append(float(com[1]))
            log["com_z"].append(float(com[2]))
            log["m_full"].append(lm.polygon_signed_margin(full_poly, (float(com[0]), float(com[1]))))
            log["m_left"].append(lm.polygon_signed_margin(foot_polygon(foot_gid["l_"]),
                                                          (float(com[0]), float(com[1]))))
            log["m_right"].append(lm.polygon_signed_margin(foot_polygon(foot_gid["r_"]),
                                                           (float(com[0]), float(com[1]))))
            log["roll"].append(roll); log["pitch"].append(pitch)
            log["Fn_l"].append(foot_normal_force(foot_gid["l_"]))
            log["Fn_r"].append(foot_normal_force(foot_gid["r_"]))
            log["slip_l"].append(float(np.linalg.norm(data.geom_xpos[foot_gid["l_"]][:2] - foot0["l_"])))
            log["slip_r"].append(float(np.linalg.norm(data.geom_xpos[foot_gid["r_"]][:2] - foot0["r_"])))
            log["nc_l"].append(foot_contacts(foot_gid["l_"]))
            log["nc_r"].append(foot_contacts(foot_gid["r_"]))
            log["sat"].append(int(np.sum(np.abs(tau) >= forcerng - 1e-6)))
            Q.append([float(data.qpos[7 + i]) for i in range(len(jn))])
            QD.append([float(data.qvel[6 + i]) for i in range(len(jn))])
            TAU.append([float(x) for x in tau])
        log = {k: np.array(v) for k, v in log.items()}
        log["q"] = np.array(Q); log["qd"] = np.array(QD); log["tau"] = np.array(TAU)
        return log

    def window_mean(log, key, t0, t1):
        m = (log["t"] >= t0) & (log["t"] <= t1)
        return float(np.mean(log[key][m]))

    def window_worst(log, key, t0, t1, kind="min"):
        m = (log["t"] >= t0) & (log["t"] <= t1)
        return float((np.min if kind == "min" else np.max)(log[key][m]))

    # ================================================================== #
    # 0. optional: watch the trajectory in the MuJoCo viewer
    # ================================================================== #
    print(f"Quasi-static weight shift  (base '{base_pose}', COM target ±{A:.3f} m, "
          f"ramp {ramp:.0f}s / hold {hold:.0f}s)")
    print(f"lower body {sum(model.body_mass):.2f} kg  |  IK table residual max {ik_res:.1e} "
          f"(frontal-plane, feet flat + planted)")

    traj, total, windows = make_trajectory(A, ramp, hold, extra_centre=1.0)

    if view:
        import time
        try:
            import mujoco.viewer
        except ImportError:
            print("error: mujoco.viewer unavailable (needs a display)", file=sys.stderr)
            return 2
        floor_z = float(model.geom_pos[floor_gid][2])
        mujoco.mj_resetDataKeyframe(model, data,
                                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, base_pose))
        print(f"\nviewer: green dot = desired COM (ground), orange dot = measured COM. "
              f"±{A:.3f} m loop; Ctrl-C or close the window to stop.")
        with mujoco.viewer.launch_passive(model, data) as v:
            for _ in range(int(settle / dt)):
                data.ctrl[:] = nominal_ctrl
                mujoco.mj_step(model, data)
            nsteps = int(total / dt)
            while v.is_running():
                wall0 = time.time()
                for step in range(nsteps):
                    cy = traj(step * dt)
                    _, qt = table_lookup(table, cy)
                    data.ctrl[:] = qt
                    mujoco.mj_step(model, data)
                    com = data.subtree_com[0]
                    v.user_scn.ngeom = 0
                    for pos, rgba in (((float(com[0]), cy, floor_z + 0.001), (0.2, 0.8, 0.3, 1)),
                                      ((float(com[0]), float(com[1]), floor_z + 0.002), (0.95, 0.55, 0.15, 1))):
                        g = v.user_scn.geoms[v.user_scn.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                                            list(pos), np.eye(3).flatten(), list(rgba))
                        v.user_scn.ngeom += 1
                    v.sync()
                    if not v.is_running():
                        break
                    dtw = time.time() - wall0
                    target = (step + 1) * dt
                    if target > dtw:
                        time.sleep(target - dtw)
        return 0

    log = episode(traj, total)

    base_t = (0.0, min(0.8, windows["+A"][0]))
    Fn_l0 = window_mean(log, "Fn_l", *base_t)
    Fn_r0 = window_mean(log, "Fn_r", *base_t)
    print(f"\ncentred baseline: Fn_left {Fn_l0:6.2f} N   Fn_right {Fn_r0:6.2f} N   "
          f"(total weight {total_weight:.2f} N)")

    rows = [("centred", *base_t)] + [(k, *v) for k, v in windows.items()]
    hdr = (f"  {'window':<9} {'COM y des':>9} {'COM y meas':>10} {'m_full':>8} "
           f"{'m_left':>8} {'m_right':>8} {'roll°':>6} {'pitch°':>7} "
           f"{'Fn_L':>7} {'Fn_R':>7} {'slip mm':>8} {'peak|τ|':>9} {'sat':>4}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, t0, t1 in rows:
        m = (log["t"] >= t0) & (log["t"] <= t1)
        peak_tau = float(np.max(np.abs(log["tau"][m])))
        print(f"  {name:<9} {window_mean(log,'com_des',t0,t1):>9.4f} "
              f"{window_mean(log,'com_y',t0,t1):>10.4f} "
              f"{window_worst(log,'m_full',t0,t1):>8.4f} "
              f"{window_mean(log,'m_left',t0,t1):>8.4f} "
              f"{window_mean(log,'m_right',t0,t1):>8.4f} "
              f"{math.degrees(window_worst(log,'roll',t0,t1,'max')):>6.2f} "
              f"{math.degrees(window_worst(log,'pitch',t0,t1,'max')):>7.2f} "
              f"{window_mean(log,'Fn_l',t0,t1):>7.2f} {window_mean(log,'Fn_r',t0,t1):>7.2f} "
              f"{1e3*window_worst(log,'slip_l',t0,t1,'max'):>8.2f} "
              f"{peak_tau:>9.3f} {int(window_worst(log,'sat',t0,t1,'max')):>4}")

    # --- validation of the demo run ----------------------------------- #
    print("\nvalidation (demonstration run):")
    checks = []

    def chk(ok, label, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  --  {detail}" if detail else ""))

    Fn_l_plus = window_mean(log, "Fn_l", *windows["+A"])
    Fn_r_plus = window_mean(log, "Fn_r", *windows["+A"])
    Fn_l_minus = window_mean(log, "Fn_l", *windows["-A"])
    Fn_r_minus = window_mean(log, "Fn_r", *windows["-A"])
    chk(Fn_l_plus > Fn_l0 and Fn_r_plus < Fn_r0,
        "shift toward +y (left) loads the LEFT foot, unloads the RIGHT",
        f"Fn_L {Fn_l0:.1f}->{Fn_l_plus:.1f}  Fn_R {Fn_r0:.1f}->{Fn_r_plus:.1f} N")
    chk(Fn_r_minus > Fn_r0 and Fn_l_minus < Fn_l0,
        "shift toward -y (right) loads the RIGHT foot, unloads the LEFT",
        f"Fn_R {Fn_r0:.1f}->{Fn_r_minus:.1f}  Fn_L {Fn_l0:.1f}->{Fn_l_minus:.1f} N")
    min_Fn = float(min(log["Fn_l"].min(), log["Fn_r"].min()))
    chk(min_Fn > MIN_OPP_FRAC * total_weight,
        "both soles stay planted (min foot normal force above threshold)",
        f"min Fn {min_Fn:.2f} N  ({100*min_Fn/total_weight:.1f}% of weight)")
    chk(int(log["nc_l"].min()) >= 3 and int(log["nc_r"].min()) >= 3,
        "each sole keeps >=3 contact corners throughout",
        f"min corners L{int(log['nc_l'].min())} R{int(log['nc_r'].min())}")
    max_slip = float(max(log["slip_l"].max(), log["slip_r"].max()))
    chk(max_slip < MAX_SLIP, "feet do not slip", f"max slip {1e3*max_slip:.2f} mm")
    max_tilt = float(max(np.abs(log["roll"]).max(), np.abs(log["pitch"]).max()))
    chk(max_tilt < MAX_TILT, "pelvis stays near level",
        f"max |roll/pitch| {math.degrees(max_tilt):.2f} deg")
    chk(float(log["m_full"].min()) > MIN_MARGIN,
        "COM stays inside the full support polygon with margin",
        f"min margin {1e3*float(log['m_full'].min()):.1f} mm")
    peak_frac = float(np.max(np.abs(log["tau"]) / forcerng))
    chk(peak_frac <= MAX_TQ_FRAC and int(log["sat"].max()) == 0,
        "no actuator torque saturation",
        f"peak torque {100*peak_frac:.0f}% of limit")
    demo_ok = all(checks)

    results = {
        "model": model_name, "total_mass_kg": float(sum(model.body_mass)),
        "total_weight_N": total_weight, "amplitude_m": A,
        "centred_Fn_N": [Fn_l0, Fn_r0], "demo_checks_passed": [int(sum(checks)), len(checks)],
        "plusA": {"com_y_des": A, "com_y_meas": window_mean(log, "com_y", *windows["+A"]),
                  "Fn_left": Fn_l_plus, "Fn_right": Fn_r_plus,
                  "m_full_mm": 1e3 * window_worst(log, "m_full", *windows["+A"]),
                  "m_loaded_mm": 1e3 * window_mean(log, "m_left", *windows["+A"]),
                  "pelvis_roll_deg": math.degrees(window_worst(log, "roll", *windows["+A"], "max")),
                  "slip_mm": 1e3 * max_slip, "peak_torque_frac": peak_frac},
    }

    if verbose:
        print("\nper-joint at the +A hold (mean over the last second):")
        m = (log["t"] >= windows["+A"][0]) & (log["t"] <= windows["+A"][1])
        for i, n in enumerate(jn):
            print(f"    {n:<14} q={float(np.mean(log['q'][m, i])):+.3f}  "
                  f"qd={float(np.mean(np.abs(log['qd'][m, i]))):.4f}  "
                  f"tau={float(np.mean(log['tau'][m, i])):+.3f} / {forcerng[i]:.1f} N*m")

    if csv_path:
        _write_csv(csv_path, log, jn)
        print(f"\nfull timeseries -> {csv_path}")

    # ================================================================== #
    # 2. sweep of COM-target magnitude
    # ================================================================== #
    sweep_limit = 0.0
    if do_sweep:
        print(f"\nSweep of lateral COM-target magnitude  (fail = breaks any acceptance "
              f"criterion at the hold):")
        sh = (f"  {'A (m)':>6} {'COMy@+A':>8} {'ΔFn_L':>7} {'ΔFn_R':>7} {'opp load%':>9} "
              f"{'m_full mm':>10} {'tilt°':>6} {'slip mm':>8} {'τ%lim':>6} {'planted':>8}  verdict")
        print(sh)
        print("  " + "-" * (len(sh) - 2))
        for Av in sweep_vals:
            tj, tot, wnd = make_trajectory(Av, ramp, hold, extra_centre=0.6)
            lg = episode(tj, tot)
            b = (0.0, min(0.5, wnd["+A"][0]))
            fnl0 = window_mean(lg, "Fn_l", *b); fnr0 = window_mean(lg, "Fn_r", *b)
            fnl = window_mean(lg, "Fn_l", *wnd["+A"]); fnr = window_mean(lg, "Fn_r", *wnd["+A"])
            comy = window_mean(lg, "com_y", *wnd["+A"])
            opp_frac = 100.0 * min(lg["Fn_l"].min(), lg["Fn_r"].min()) / total_weight
            mfull = max(-999.0, 1e3 * float(lg["m_full"].min()))
            tilt = math.degrees(float(max(np.abs(lg["roll"]).max(), np.abs(lg["pitch"]).max())))
            slip = 1e3 * float(max(lg["slip_l"].max(), lg["slip_r"].max()))
            tqf = 100.0 * float(np.max(np.abs(lg["tau"]) / forcerng))
            planted = bool(lg["nc_l"].min() >= 3 and lg["nc_r"].min() >= 3
                           and min(lg["Fn_l"].min(), lg["Fn_r"].min()) > MIN_OPP_FRAC * total_weight)
            transfer_ok = fnl > fnl0 and fnr < fnr0
            ok = (planted and transfer_ok and mfull > 1e3 * MIN_MARGIN
                  and opp_frac > 100 * MIN_OPP_FRAC and tilt < math.degrees(MAX_TILT)
                  and slip < 1e3 * MAX_SLIP and tqf <= 100 * MAX_TQ_FRAC)
            if ok:
                sweep_limit = max(sweep_limit, Av)
            fail_bits = []
            if not planted: fail_bits.append("unplanted")
            if not transfer_ok: fail_bits.append("no-transfer")
            if mfull <= 1e3 * MIN_MARGIN: fail_bits.append("margin")
            if opp_frac <= 100 * MIN_OPP_FRAC: fail_bits.append("opp-load")
            if tilt >= math.degrees(MAX_TILT): fail_bits.append("tilt")
            if slip >= 1e3 * MAX_SLIP: fail_bits.append("slip")
            if tqf > 100 * MAX_TQ_FRAC: fail_bits.append("torque")
            print(f"  {Av:>6.3f} {comy:>8.4f} {fnl-fnl0:>+7.2f} {fnr-fnr0:>+7.2f} {opp_frac:>8.1f}% "
                  f"{mfull:>10.1f} {tilt:>6.2f} {slip:>8.2f} {tqf:>6.0f} {str(planted):>8}  "
                  f"{'PASS' if ok else 'FAIL: ' + ','.join(fail_bits)}")
        print(f"\n  largest COM target that stays in controlled double support: "
              f"{sweep_limit:.3f} m"
              + (f"  (~{100*sweep_limit/0.05:.0f}% of the hip half-width)" if sweep_limit else ""))
        results["sweep_limit_m"] = sweep_limit

    if baseline_path:
        base = json.load(open(baseline_path, encoding="utf-8"))
        print(f"\nDelta vs baseline '{base.get('model', '?')}' "
              f"({base.get('total_mass_kg', 0):.2f} kg -> {results['total_mass_kg']:.2f} kg):")
        b, c = base.get("plusA", {}), results["plusA"]
        for k, lbl, p in (("Fn_left", "Fn loaded", 2), ("Fn_right", "Fn unloaded", 2),
                          ("m_full_mm", "COM margin mm", 1), ("pelvis_roll_deg", "pelvis roll deg", 2),
                          ("slip_mm", "slip mm", 2)):
            if k in b:
                print(f"  {lbl:<16} {c[k]:.{p}f} ({c[k]-b[k]:+.{p}f})")
        if "sweep_limit_m" in base and "sweep_limit_m" in results:
            print(f"  {'shift limit m':<16} {results['sweep_limit_m']:.3f} "
                  f"({results['sweep_limit_m']-base['sweep_limit_m']:+.3f})")

    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsummary -> {json_path}")

    # ================================================================== #
    print("\n" + "=" * 66)
    answer = demo_ok and (not do_sweep or sweep_limit >= 0.02)
    print(f"MILESTONE {'MET' if answer else 'NOT MET'}: "
          f"Cara {'can' if answer else 'cannot yet'} deliberately transfer weight "
          f"L<->R in controlled double support.")
    if answer:
        line = f"  demonstrated at ±{A:.3f} m COM target"
        if do_sweep:
            line += (f"; quasi-static limit ~{sweep_limit:.3f} m before "
                     "support/contact/tilt/torque criteria fail")
        print(line + ".")
    print("  (provisional masses / PD gains / friction; no foot lifting, no RL)")
    return 0 if answer else 1


def _write_csv(path, log, jn):
    cols = ["t", "com_des_y", "com_ff_y", "com_x", "com_y", "com_z",
            "margin_full", "margin_left", "margin_right", "pelvis_roll", "pelvis_pitch",
            "Fn_left", "Fn_right", "slip_left", "slip_right", "n_contact_left",
            "n_contact_right", "sat_count"]
    cols += [f"q_{n}" for n in jn] + [f"qd_{n}" for n in jn] + [f"tau_{n}" for n in jn]
    key = ["t", "com_des", "com_ff", "com_x", "com_y", "com_z", "m_full", "m_left",
           "m_right", "roll", "pitch", "Fn_l", "Fn_r", "slip_l", "slip_r", "nc_l", "nc_r", "sat"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for i in range(len(log["t"])):
            vals = [log[k][i] for k in key]
            vals += list(log["q"][i]) + list(log["qd"][i]) + list(log["tau"][i])
            fh.write(",".join(f"{v:.6g}" for v in vals) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--amplitude", type=float, default=None, help="COM target for the demo run [m]")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--csv", default=None, help="write the demo-run timeseries here")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--view", action="store_true",
                    help="watch the COM-shift trajectory loop in the MuJoCo viewer (needs a display)")
    ap.add_argument("--json", default=None, help="write the run summary here")
    ap.add_argument("--baseline", default=None, help="print deltas vs this summary JSON")
    args = ap.parse_args(argv)
    return run(args.config, args.amplitude, not args.no_sweep, args.csv, args.verbose,
               args.view, args.json, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
