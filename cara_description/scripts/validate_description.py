#!/usr/bin/env python3
"""Structural validation for config/left_leg.yaml.

Checks (all must pass):
  1. every joint axis is a unit vector
  2. every joint parent/child link exists in `links`
  3. every joint lower limit is strictly less than its upper limit
  4. every PHYSICAL link has positive mass; virtual coupling links carry
     no mass / COM / inertia at all
  5. every required parameter / key is defined
  6. the joints form a single-rooted tree (one root, no cycles, fully
     connected, each non-root link is the child of exactly one joint)
  7. joints are listed parent-before-child (so FK can sweep them once)
  8. every origin_expr / com / inertia / frame expression evaluates cleanly
  9. forward kinematics at the zero pose produces finite numbers
 10. dynamics: inertia diagonals are positive and obey the triangle
     inequality; dynamics.links matches `links` exactly; is_physical agrees
     with each link's role
 11. analysis: gravity > 0; every reference-pose joint exists and is in range;
     ground.friction / z_offset well-formed
 12. dynamics.actuators.control: kp > 0, dampratio >= 0, overrides name real joints

Exit code 0 = all checks passed, 1 = one or more failures.

Usage:
    python3 validate_description.py [path/to/left_leg.yaml]
"""

from __future__ import annotations

import math
import sys

import leg_model as lm

AXIS_UNIT_TOL = 1e-9
PHYSICAL_ROLES = {"base", "segment"}
VIRTUAL_ROLES = {"virtual_coupling"}


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        self.checks += 1
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {label}"
        if detail:
            line += f"  --  {detail}"
        print(line)
        if not ok:
            self.failures.append(label)


def _require_keys(rep: Report, section: str, obj: dict, keys) -> None:
    for k in keys:
        rep.check(k in obj, f"{section}: key '{k}' present")


def _flat_num(node, prefix, out) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _flat_num(v, f"{prefix}{k}." if prefix else f"{k}.", out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out[prefix[:-1]] = float(node)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(path: str | None) -> int:
    spec = lm.load_spec(path)
    rep = Report()

    print("== 5. required parameters / keys ==")
    _require_keys(rep, "top-level", spec,
                  ["meta", "frame_conventions", "provisional_geometry",
                   "links", "joints", "frames_of_interest", "dynamics", "analysis"])
    fc = spec.get("frame_conventions", {})
    _require_keys(rep, "frame_conventions", fc,
                  ["base_frame", "handedness", "axes", "zero_pose", "sign_convention"])
    pg = spec.get("provisional_geometry", {})
    for sym in lm.REQUIRED_SYMBOLS:
        rep.check(_num(pg.get(sym)), f"provisional_geometry: symbol '{sym}' is numeric")

    dyn = spec.get("dynamics", {}) or {}
    _require_keys(rep, "dynamics", dyn, ["links", "actuators"])
    act = dyn.get("actuators", {}) or {}
    _require_keys(rep, "dynamics.actuators", act, ["defaults"])
    _require_keys(rep, "dynamics.actuators.defaults", act.get("defaults", {}) or {},
                  ["effort", "velocity"])
    ana = spec.get("analysis", {}) or {}
    _require_keys(rep, "analysis", ana, ["gravity", "reference_poses"])

    links = spec.get("links", []) or []
    joints = spec.get("joints", []) or []
    link_names = [l["name"] for l in links]
    roles = {l["name"]: l.get("role") for l in links}
    rep.check(len(link_names) == len(set(link_names)), "links: names are unique",
              detail=f"{len(link_names)} links")
    for j in joints:
        fixed = j.get("type") == "fixed" or j.get("locked", False)
        keys = (("name", "type", "parent", "child", "origin_expr")
                if fixed else lm.REQUIRED_JOINT_KEYS)
        _require_keys(rep, f"joint '{j.get('name', '?')}'", j, keys)
    for foi in spec.get("frames_of_interest", []) or []:
        _require_keys(rep, f"frame '{foi.get('name', '?')}'", foi, ["name", "link", "xyz_expr"])

    # ---- 8 + build the resolved model -----------------------------------
    print("\n== 8. origin / com / inertia / frame expressions evaluate ==")
    chain = inertials = None
    try:
        chain = lm.build_chain(spec)
        for jm in chain:
            rep.check(all(math.isfinite(c) for c in jm.origin),
                      f"joint '{jm.name}': origin evaluates finite",
                      detail=f"origin={tuple(round(c, 6) for c in jm.origin)}")
        inertials = lm.link_inertials(spec)
        for name, li in inertials.items():
            rep.check(all(math.isfinite(c) for c in li.com)
                      and all(math.isfinite(c) for c in li.inertia_diag),
                      f"link '{name}': com + inertia evaluate finite",
                      detail=f"com={tuple(round(c, 6) for c in li.com)} "
                             f"I={tuple(round(c, 8) for c in li.inertia_diag)} ({li.method})")
        syms = lm.resolve_symbols(spec)
        for foi in spec.get("frames_of_interest", []) or []:
            v = lm.resolve_vec3(foi["xyz_expr"], syms)
            rep.check(all(math.isfinite(c) for c in v),
                      f"frame '{foi['name']}': expression evaluates",
                      detail=f"xyz={tuple(round(c, 6) for c in v)}")
    except Exception as exc:  # noqa: BLE001
        rep.check(False, "expressions evaluate without error", detail=repr(exc))

    if chain is None or inertials is None:
        return _summary(rep)

    print("\n== 1. joint axes are unit vectors  (actuated joints only) ==")
    for jm in chain:
        if jm.fixed:
            continue
        n = jm.axis_norm
        rep.check(abs(n - 1.0) <= AXIS_UNIT_TOL,
                  f"joint '{jm.name}': |axis| == 1", detail=f"|axis|={n:.15g}")

    print("\n== 2. parent / child links exist ==")
    known = set(link_names)
    for jm in chain:
        rep.check(jm.parent in known, f"joint '{jm.name}': parent '{jm.parent}' is a known link")
        rep.check(jm.child in known, f"joint '{jm.name}': child '{jm.child}' is a known link")

    print("\n== 3. lower limit < upper limit  (actuated joints only) ==")
    for jm in chain:
        if jm.fixed:
            continue
        rep.check(jm.lower < jm.upper, f"joint '{jm.name}': lower < upper",
                  detail=f"[{jm.lower}, {jm.upper}] rad")

    fixed_joints = [jm for jm in chain if jm.fixed]
    if fixed_joints:
        print("\n== 3b. fixed / locked joints ==")
        for jm in fixed_joints:
            rep.check(all(math.isfinite(c) for c in jm.origin),
                      f"fixed joint '{jm.name}': origin finite",
                      detail=f"{tuple(round(c, 5) for c in jm.origin)}")

    print("\n== 4 + 10. dynamics: physical vs virtual links ==")
    dyn_links = dyn.get("links", {}) or {}
    rep.check(set(dyn_links) == set(link_names),
              "dynamics.links has exactly one entry per link",
              detail=f"missing={sorted(set(link_names) - set(dyn_links))} "
                     f"extra={sorted(set(dyn_links) - set(link_names))}")
    for name in link_names:
        d = dyn_links.get(name, {}) or {}
        is_phys = bool(d.get("is_physical", False))
        role = roles.get(name)
        if role in PHYSICAL_ROLES:
            rep.check(is_phys, f"link '{name}' (role {role}): is_physical is true")
            m_resolved = inertials.get(name).mass if name in inertials else None
            rep.check(m_resolved is not None and m_resolved > 0.0,
                      f"link '{name}': mass > 0",
                      detail=f"{d.get('mass')} -> {m_resolved} kg")
            rep.check(isinstance(d.get("com"), list) and len(d["com"]) == 3,
                      f"link '{name}': com is a 3-vector")
            rep.check(isinstance(d.get("inertia"), dict) and "method" in d["inertia"],
                      f"link '{name}': inertia block with a method")
        elif role in VIRTUAL_ROLES:
            rep.check(not is_phys, f"link '{name}' (role {role}): is_physical is false")
            for forbidden in ("mass", "com", "inertia"):
                rep.check(forbidden not in d,
                          f"virtual link '{name}': carries no '{forbidden}'")
            rep.check(name not in inertials,
                      f"virtual link '{name}': excluded from inertial model")
        else:
            rep.check(False, f"link '{name}': unknown role {role!r}")

    print("\n== 10. inertia diagonals valid ==")
    for name, li in inertials.items():
        ixx, iyy, izz = li.inertia_diag
        rep.check(ixx > 0 and iyy > 0 and izz > 0,
                  f"link '{name}': inertia diagonal strictly positive",
                  detail=f"({ixx:.3e}, {iyy:.3e}, {izz:.3e})")
        tri = (ixx + iyy >= izz - 1e-12 and iyy + izz >= ixx - 1e-12
               and izz + ixx >= iyy - 1e-12)
        rep.check(tri, f"link '{name}': inertia triangle inequality holds")

    print("\n== 6. joints form a single-rooted tree ==")
    children = [jm.child for jm in chain]
    rep.check(len(children) == len(set(children)),
              "each link is the child of at most one joint",
              detail="dupes: " + str(sorted({c for c in children if children.count(c) > 1})))
    roots = [n for n in link_names if n not in set(children)]
    rep.check(len(roots) == 1, "exactly one root link", detail=f"roots={roots}")
    declared_base = spec["frame_conventions"]["base_frame"]
    rep.check(roots == [declared_base] if roots else False,
              f"root link matches frame_conventions.base_frame ('{declared_base}')")

    adj: dict[str, list[str]] = {n: [] for n in link_names}
    for jm in chain:
        if jm.parent in adj:
            adj[jm.parent].append(jm.child)
    reachable: set[str] = set()
    if roots:
        stack = [roots[0]]
        while stack:
            n = stack.pop()
            if n in reachable:
                continue
            reachable.add(n)
            stack.extend(adj.get(n, []))
    rep.check(reachable == set(link_names), "every link is reachable from the root",
              detail=f"unreachable={sorted(set(link_names) - reachable)}")
    rep.check(len(chain) == len(link_names) - 1,
              "joint count == link count - 1 (tree, no extra edges)",
              detail=f"{len(chain)} joints, {len(link_names)} links")

    print("\n== 7. joints listed parent-before-child ==")
    placed = {declared_base}
    order_ok = True
    for jm in chain:
        if jm.parent not in placed:
            order_ok = False
            rep.check(False, f"joint '{jm.name}': parent placed before child",
                      detail=f"parent '{jm.parent}' appears later")
        placed.add(jm.child)
    if order_ok:
        rep.check(True, "all joints listed after their parent joint")

    print("\n== 11. analysis block ==")
    rep.check(_num(ana.get("gravity")) and ana["gravity"] > 0,
              "analysis.gravity > 0", detail=f"{ana.get('gravity')} m/s^2")
    jset = {jm.name for jm in chain}
    raw_poses = (ana.get("reference_poses", {}) or {})
    exp_poses = lm.reference_poses(spec)   # wildcards ('*_hip_pitch') expanded
    for pose_name, raw_cfg in raw_poses.items():
        bad_wild = [k for k in (raw_cfg or {})
                    if k.startswith("*_") and not any(j.endswith("_" + k[2:]) or j == k[2:] for j in jset)]
        rep.check(not bad_wild, f"reference_pose '{pose_name}': wildcards match a joint",
                  detail=f"no match: {bad_wild}")
        cfg = exp_poses.get(pose_name, {})
        bad = [k for k in cfg if k not in jset]
        rep.check(not bad, f"reference_pose '{pose_name}': all joints exist",
                  detail=f"unknown: {bad}")
        in_range = all(jm.lower - 1e-9 <= float(cfg.get(jm.name, 0.0)) <= jm.upper + 1e-9
                       for jm in chain)
        rep.check(in_range, f"reference_pose '{pose_name}': all angles within joint limits")
    grd = ana.get("ground", {}) or {}
    if grd:
        fr = grd.get("friction")
        rep.check(isinstance(fr, list) and len(fr) == 3 and all(_num(x) and x >= 0 for x in fr),
                  "analysis.ground.friction is 3 non-negative numbers", detail=str(fr))
        rep.check(_num(grd.get("z_offset", 0.0)), "analysis.ground.z_offset is numeric")

    ws = ana.get("weight_shift", {}) or {}
    if ws:
        rep.check(ws.get("base_pose") in exp_poses,
                  "weight_shift.base_pose names a reference pose", detail=str(ws.get("base_pose")))
        rep.check(_num(ws.get("amplitude")) and ws["amplitude"] > 0,
                  "weight_shift.amplitude > 0", detail=f"{ws.get('amplitude')} m")
        sw = ws.get("sweep")
        rep.check(isinstance(sw, list) and sw and all(_num(x) and x > 0 for x in sw)
                  and sw == sorted(sw),
                  "weight_shift.sweep is a sorted list of positive magnitudes", detail=str(sw))
        for k in ("ramp_seconds", "hold_seconds", "settle_seconds"):
            rep.check(_num(ws.get(k)) and ws[k] > 0, f"weight_shift.{k} > 0")
        for k, v in (ws.get("accept", {}) or {}).items():
            rep.check(_num(v) and v > 0, f"weight_shift.accept.{k} > 0", detail=f"{v}")

    print("\n== 12. dynamics.actuators.control (PD gains) ==")
    ctrl = act.get("control", {}) or {}
    if not ctrl:
        rep.check(True, "no control block (dynamic MJCF will have zero-gain servos)")
    else:
        rep.check(_num(ctrl.get("kp")) and ctrl["kp"] > 0,
                  "control.kp > 0", detail=f"{ctrl.get('kp')} N*m/rad")
        rep.check(_num(ctrl.get("dampratio", 1.0)) and ctrl.get("dampratio", 1.0) >= 0,
                  "control.dampratio >= 0", detail=f"{ctrl.get('dampratio', 1.0)}")
        for jname, jov in (ctrl.get("overrides", {}) or {}).items():
            rep.check(jname in jset, f"control override '{jname}' is a real joint")
            if "kp" in (jov or {}):
                rep.check(_num(jov["kp"]) and jov["kp"] > 0,
                          f"control override '{jname}': kp > 0", detail=f"{jov['kp']}")

    print("\n== 13. base / mirror ==")
    b = spec.get("base", {}) or {}
    rep.check(b.get("type", "fixed") in ("fixed", "floating"),
              "base.type is fixed or floating", detail=str(b.get("type")))
    if b.get("type") == "floating":
        rp = b.get("rest_pose")
        rep.check(rp is None or rp in raw_poses,
                  "base.rest_pose names a reference pose", detail=str(rp))
    if spec.get("_mirrored"):
        rep.check(any(l["name"].startswith("l_") for l in links)
                  and any(l["name"].startswith("r_") for l in links),
                  "mirror expanded both l_ and r_ links")
        soles = [f["name"] for f in spec.get("frames_of_interest", []) or []]
        rep.check("l_foot_sole_center" in soles and "r_foot_sole_center" in soles,
                  "both foot sole frames present", detail=str(soles))

        # every physical l_* link has an r_* twin with equal mass + y-mirrored COM
        # (catches a desymmetrised sweep of dynamics.links.l_<part>.mass -- see U4)
        try:
            inr = lm.link_inertials(spec)
        except Exception:  # noqa: BLE001
            inr = {}
        for name, li in inr.items():
            if not name.startswith("l_"):
                continue
            twin = inr.get("r_" + name[2:])
            rep.check(twin is not None, f"physical link {name!r} has an r_ twin")
            if twin is None:
                continue
            rep.check(abs(li.mass - twin.mass) < 1e-9,
                      f"{name} / r_{name[2:]} masses match",
                      detail=f"{li.mass} vs {twin.mass} kg")
            rep.check(abs(li.com[0] - twin.com[0]) < 1e-9
                      and abs(li.com[1] + twin.com[1]) < 1e-9
                      and abs(li.com[2] - twin.com[2]) < 1e-9,
                      f"{name} / r_{name[2:]} COMs are sagittally mirrored",
                      detail=f"{li.com} vs {twin.com}")

    ub = spec.get("upper_body", {}) or {}
    if ub:
        print("\n== 13b. upper_body block ==")
        flat = {}
        _flat_num(ub, "", flat)
        rep.check(len(flat) > 0, "upper_body has numeric leaves", detail=f"{sorted(flat)}")
        for k, v in flat.items():
            rep.check(math.isfinite(v), f"upper_body.{k} is a finite number", detail=f"{v}")
        # every symbol referenced by a joint origin / link com / inertia box resolves
        try:
            lm.build_chain(spec)
            lm.link_inertials(spec)
            rep.check(True, "upper_body symbols all resolve in joints + dynamics")
        except Exception as exc:  # noqa: BLE001
            rep.check(False, "upper_body symbols resolve", detail=repr(exc))

    el = spec.get("electronics", {}) or {}
    if el:
        print("\n== 13c. electronics block (U3) ==")
        mounts = el.get("mounts", {}) or {}
        rep.check(bool(mounts), "electronics.mounts is non-empty")
        for mn, mv in mounts.items():
            rep.check((mv or {}).get("link") in {l["name"] for l in links},
                      f"mount preset '{mn}' -> a real link", detail=str((mv or {}).get("link")))
            for ax in "xyz":
                if ax in (mv or {}):
                    rep.check(_num(mv[ax]), f"mount '{mn}': {ax} offset numeric")
        for item in ("jetson", "battery"):
            it = el.get(item, {}) or {}
            rep.check(_num(it.get("mass")) and it["mass"] > 0,
                      f"electronics.{item}.mass > 0", detail=f"{it.get('mass')} kg")
            rep.check(it.get("mount") in mounts,
                      f"electronics.{item}.mount -> a mount preset", detail=str(it.get("mount")))
        for ln, lv in (el.get("layouts", {}) or {}).items():
            for item, preset in (lv or {}).items():
                rep.check(preset in mounts,
                          f"layout '{ln}': {item} -> a mount preset", detail=str(preset))
        # the mount_from joints resolved to real links
        for j in spec.get("joints", []) or []:
            if j.get("mount_from"):
                rep.check(j.get("parent") in {l["name"] for l in links},
                          f"joint '{j['name']}': mount resolved to a real parent link",
                          detail=str(j.get("parent")))

    print("\n== 9. forward kinematics at zero pose is finite ==")
    try:
        tf = lm.forward_kinematics(spec, {})
        sole = lm.frame_world_position(spec, tf, spec["frames_of_interest"][0]["name"])
        rep.check(all(math.isfinite(c) for c in sole),
                  "FK(zero) foot sole position is finite",
                  detail=f"sole={tuple(round(c, 6) for c in sole)} m")
        m_tot, com, _ = lm.center_of_mass(spec, {})
        rep.check(all(math.isfinite(c) for c in com) and m_tot > 0,
                  "COM(zero) is finite",
                  detail=f"m={m_tot:.4f} kg  com={tuple(round(c, 5) for c in com)} m")
    except Exception as exc:  # noqa: BLE001
        rep.check(False, "FK / COM at zero pose runs", detail=repr(exc))

    return _summary(rep)


def _summary(rep: Report) -> int:
    print("\n" + "=" * 60)
    if rep.failures:
        print(f"RESULT: FAIL  ({len(rep.failures)}/{rep.checks} checks failed)")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print(f"RESULT: PASS  ({rep.checks}/{rep.checks} checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(validate(sys.argv[1] if len(sys.argv) > 1 else None))
