"""Shared loader + pure-Python kinematics & dynamics for Cara's left leg.

Only third-party dependency: PyYAML. All linear algebra is plain Python
(3x3 matrices, 3-vectors) so every script runs on a stock interpreter.

SI units throughout: metres, kilograms, radians, newton-metres, seconds.

Frame convention (see config/left_leg.yaml -> frame_conventions):
    +X forward, +Y left, +Z up, right-handed.
    At the zero pose every link frame is axis-aligned with the base frame.
    Gravity acceleration vector is (0, 0, -g).

Layers exposed here, matching the YAML:
    * kinematics -- forward_kinematics(), frame_world_position()
    * dynamics   -- link_inertials(), center_of_mass(), potential_energy()
    * analysis   -- foot_jacobian_geometric()/_numeric(),
                    gravity_joint_torques(), joint_torques_from_foot_force()
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import yaml

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]
Transform = Tuple[Mat3, Vec3]  # (rotation, translation) in the base frame

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.normpath(os.path.join(_HERE, os.pardir, "config", "left_leg.yaml"))

IDENTITY3: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
IDENTITY: Transform = (IDENTITY3, (0.0, 0.0, 0.0))

REQUIRED_SYMBOLS = (
    "L_thigh", "L_shin", "h_ankle",
    "w_hip_half", "z_hip", "x_hip",
    "foot_len", "foot_width", "foot_x_off",
)

REQUIRED_JOINT_KEYS = (
    "name", "type", "parent", "child",
    "origin_expr", "axis", "limits", "purpose", "positive_rotation",
)

DEFAULT_GRAVITY = 9.81


# --------------------------------------------------------------------------- #
# Loading  (extends + mirror expansion)
# --------------------------------------------------------------------------- #
# A spec may:
#   * `extends: other.yaml`  -- deep-merge on top of a parent spec.
#   * `mirror: {source: "l_", target: "r_"}`  -- synthesise the opposite leg by
#     reflecting the source-prefixed links/joints/dynamics/frames through the
#     sagittal (x-z) plane:  position (x, y, z) -> (x, -y, z);  rotation axis
#     (ax, ay, az) -> (-ax, ay, -az)  [axial-vector reflection].  Joint LIMITS
#     are copied unchanged: the axis flip is chosen so the joint coordinate
#     keeps the same physical meaning (e.g. +angle = abduction) on both sides.
# All other code operates on the fully expanded (flat) spec.

_REPLACE_KEYS = {"reference_poses"}  # child value replaces, not merges

_MIRROR_SUBS = [("+Y", "\x00"), ("-Y", "+Y"), ("\x00", "-Y"),
                ("+y", "\x01"), ("-y", "+y"), ("\x01", "-y"),
                ("left", "\x02"), ("right", "left"), ("\x02", "right"),
                ("Left", "\x03"), ("Right", "Left"), ("\x03", "Right")]


def _mirror_text(s):
    if not isinstance(s, str):
        return s
    for a, b in _MIRROR_SUBS:
        s = s.replace(a, b)
    return s


def _negate_component(entry):
    """Negate one origin/axis component (number or expression string)."""
    if isinstance(entry, bool):
        return entry
    if isinstance(entry, (int, float)):
        return 0.0 if entry == 0 else -float(entry)
    if isinstance(entry, str):
        e = entry.strip()
        try:
            f = float(e)
            return 0.0 if f == 0 else -f
        except ValueError:
            return f"-({e})"
    return entry


def _mirror_position(v):          # (x, y, z) -> (x, -y, z)
    return [v[0], _negate_component(v[1]), v[2]]


def _mirror_axis(v):              # (ax, ay, az) -> (-ax, ay, -az)
    return [_negate_component(v[0]), v[1], _negate_component(v[2])]


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k in _REPLACE_KEYS or not (isinstance(v, dict) and isinstance(out.get(k), dict)):
            out[k] = copy.deepcopy(v)
        else:
            out[k] = _deep_merge(out[k], v)
    return out


def _apply_mirror(spec: dict) -> dict:
    m = spec.get("mirror")
    if not m:
        return spec
    src, tgt = m["source"], m["target"]
    spec = copy.deepcopy(spec)

    def rn(name):
        return tgt + name[len(src):] if isinstance(name, str) and name.startswith(src) else name

    # Emit ALL source-side entries first, then ALL mirrored entries, so each
    # leg's joint chain stays contiguous (generate_mjcf groups by chain order).
    links = spec.get("links", []) or []
    spec["links"] = list(links) + [
        {**lk, "name": rn(lk["name"])} for lk in links if lk["name"].startswith(src)
    ]

    joints = spec.get("joints", []) or []
    mirrored = []
    for j in joints:
        if j["name"].startswith(src) or str(j.get("child", "")).startswith(src):
            mj = copy.deepcopy(j)
            mj["name"] = rn(j["name"])
            mj["parent"] = rn(j["parent"])
            mj["child"] = rn(j["child"])
            mj["origin_expr"] = _mirror_position(j["origin_expr"])
            mj["axis"] = _mirror_axis(j["axis"])
            mj["positive_rotation"] = _mirror_text(j.get("positive_rotation", ""))
            mj["purpose"] = _mirror_text(j.get("purpose", ""))
            mirrored.append(mj)
    spec["joints"] = list(joints) + mirrored

    dl = ((spec.get("dynamics") or {}).get("links") or {})
    for name in list(dl):
        if name.startswith(src):
            d = copy.deepcopy(dl[name])
            if isinstance(d.get("com"), list):
                d["com"] = _mirror_position(d["com"])
            dl[rn(name)] = d

    fois = spec.get("frames_of_interest", []) or []
    spec["frames_of_interest"] = []
    for foi in fois:
        spec["frames_of_interest"].append(foi)
        if str(foi.get("link", "")).startswith(src):
            mf = copy.deepcopy(foi)
            mf["name"] = rn(foi["name"])
            mf["link"] = rn(foi["link"])
            mf["xyz_expr"] = _mirror_position(foi["xyz_expr"])
            if "description" in mf:
                mf["description"] = _mirror_text(mf["description"])
            spec["frames_of_interest"].append(mf)

    act = (spec.get("dynamics") or {}).get("actuators") or {}
    for blk in (act.get("overrides") or {}, (act.get("control") or {}).get("overrides") or {}):
        for name in list(blk):
            if name.startswith(src):
                blk[rn(name)] = copy.deepcopy(blk[name])

    return spec


def _load_raw(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_spec(path: str | None = None) -> dict:
    """Load a YAML spec, applying `extends` (deep merge) then `mirror` expansion.

    The returned spec is fully flat: `links`, `joints`, `dynamics.links` and
    `frames_of_interest` list every side explicitly.
    """
    path = os.path.abspath(path or DEFAULT_CONFIG)
    src = os.path.basename(path)
    spec = _load_raw(path)
    here = os.path.dirname(path)
    while "extends" in spec:
        parent_path = os.path.join(here, spec.pop("extends"))
        parent = _load_raw(parent_path)
        spec = _deep_merge(parent, spec)
        here = os.path.dirname(os.path.abspath(parent_path))
    spec = _apply_mirror(spec)
    spec["_source"] = src
    return spec


def base_spec(spec: dict) -> dict:
    """Floating vs fixed base config (MJCF).  Defaults to a fixed (welded) base."""
    b = spec.get("base", {}) or {}
    return {
        "type": b.get("type", "fixed"),
        "rest_pose": b.get("rest_pose"),
        "rest_height": (float(b["rest_height"]) if b.get("rest_height") is not None else None),
    }


def resolve_symbols(spec: dict) -> Dict[str, float]:
    """Flatten the scalar entries of provisional_geometry into a name->float map."""
    pg = spec.get("provisional_geometry", {}) or {}
    return {k: float(v) for k, v in pg.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def eval_expr(entry, syms: Dict[str, float]) -> float:
    """Evaluate one component: a number, or short arithmetic over the
    provisional_geometry symbols (e.g. 0, "-L_thigh", "-L_thigh/2", "foot_x_off").
    Builtins are stripped; only the provided symbols are in scope.
    """
    if isinstance(entry, bool):
        raise TypeError(f"boolean is not a valid component: {entry!r}")
    if isinstance(entry, (int, float)):
        return float(entry)
    if not isinstance(entry, str):
        raise TypeError(f"component must be a number or expression string, got {entry!r}")
    try:
        return float(eval(entry, {"__builtins__": {}}, dict(syms)))  # noqa: S307 - trusted local config
    except NameError as exc:
        raise KeyError(f"unknown symbol in expression {entry!r}: {exc}") from exc


def resolve_vec3(expr_list: Sequence, syms: Dict[str, float]) -> Vec3:
    if len(expr_list) != 3:
        raise ValueError(f"expected 3 components, got {expr_list!r}")
    return tuple(eval_expr(e, syms) for e in expr_list)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Minimal linear algebra
# --------------------------------------------------------------------------- #
def vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(c * c for c in a))


def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def normalize(a: Vec3) -> Vec3:
    n = vec_norm(a)
    if n == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def mat_mul(A: Mat3, B: Mat3) -> Mat3:
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def mat_vec(A: Mat3, v: Vec3) -> Vec3:
    return tuple(sum(A[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore[return-value]


def rot_axis_angle(axis: Vec3, theta: float) -> Mat3:
    """Rodrigues rotation matrix for a rotation of `theta` rad about `axis`."""
    x, y, z = normalize(axis)
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return (
        (c + x * x * C,     x * y * C - z * s, x * z * C + y * s),
        (y * x * C + z * s, c + y * y * C,     y * z * C - x * s),
        (z * x * C - y * s, z * y * C + x * s, c + z * z * C),
    )


def tf_compose(t1: Transform, t2: Transform) -> Transform:
    r1, p1 = t1
    r2, p2 = t2
    return (mat_mul(r1, r2), vec_add(mat_vec(r1, p2), p1))


# --------------------------------------------------------------------------- #
# Joint chain model
# --------------------------------------------------------------------------- #
@dataclass
class JointModel:
    name: str
    jtype: str
    parent: str
    child: str
    origin: Vec3
    axis: Vec3
    lower: float
    upper: float
    effort: float
    velocity: float
    purpose: str
    positive_rotation: str

    @property
    def axis_norm(self) -> float:
        return vec_norm(self.axis)

    def clamp(self, angle: float) -> float:
        return max(self.lower, min(self.upper, angle))


def _actuator_tables(spec: dict) -> tuple[dict, dict]:
    act = (spec.get("dynamics", {}) or {}).get("actuators", {}) or {}
    return act.get("defaults", {}) or {}, act.get("overrides", {}) or {}


def build_chain(spec: dict) -> List[JointModel]:
    """Resolve every joint into a JointModel with numeric origin + limits."""
    syms = resolve_symbols(spec)
    defaults, overrides = _actuator_tables(spec)

    chain: List[JointModel] = []
    for j in spec.get("joints", []):
        missing = [k for k in REQUIRED_JOINT_KEYS if k not in j]
        if missing:
            raise KeyError(f"joint {j.get('name', '<unnamed>')} missing keys: {missing}")
        lim = j["limits"]
        ov = overrides.get(j["name"], {}) or {}
        chain.append(JointModel(
            name=j["name"],
            jtype=j["type"],
            parent=j["parent"],
            child=j["child"],
            origin=resolve_vec3(j["origin_expr"], syms),
            axis=tuple(float(a) for a in j["axis"]),  # type: ignore[arg-type]
            lower=float(lim["lower"]),
            upper=float(lim["upper"]),
            effort=float(ov.get("effort", defaults.get("effort", 0.0))),
            velocity=float(ov.get("velocity", defaults.get("velocity", 0.0))),
            purpose=" ".join(str(j["purpose"]).split()),
            positive_rotation=" ".join(str(j["positive_rotation"]).split()),
        ))
    return chain


def joint_names(spec: dict) -> List[str]:
    return [jm.name for jm in build_chain(spec)]


def joint_limits(spec: dict) -> Dict[str, Tuple[float, float]]:
    return {jm.name: (jm.lower, jm.upper) for jm in build_chain(spec)}


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
def forward_kinematics(spec: dict, q: Dict[str, float] | None = None) -> Dict[str, Transform]:
    """Base-frame transform for every link, given a joint-angle map.
    Missing joints default to 0. Joints must be listed parent-before-child.
    """
    q = dict(q or {})
    base = spec["frame_conventions"]["base_frame"]
    transforms: Dict[str, Transform] = {base: IDENTITY}

    for jm in build_chain(spec):
        if jm.parent not in transforms:
            raise ValueError(
                f"joint {jm.name!r}: parent {jm.parent!r} not placed yet "
                f"(joints must be listed parent-before-child)"
            )
        t_origin: Transform = (IDENTITY3, jm.origin)
        t_joint: Transform = (rot_axis_angle(jm.axis, float(q.get(jm.name, 0.0))), (0.0, 0.0, 0.0))
        transforms[jm.child] = tf_compose(tf_compose(transforms[jm.parent], t_origin), t_joint)

    return transforms


def _find_foi(spec: dict, name: str) -> dict:
    for foi in spec.get("frames_of_interest", []) or []:
        if foi["name"] == name:
            return foi
    raise KeyError(f"no frame_of_interest named {name!r}")


def frame_world_position(spec: dict, transforms: Dict[str, Transform], name: str) -> Vec3:
    """World position of a named entry in frames_of_interest."""
    foi = _find_foi(spec, name)
    local = resolve_vec3(foi["xyz_expr"], resolve_symbols(spec))
    r, p = transforms[foi["link"]]
    return vec_add(mat_vec(r, local), p)


def link_origin(transforms: Dict[str, Transform], link: str) -> Vec3:
    return transforms[link][1]


def foot_position(spec: dict, q: Dict[str, float] | None = None,
                  point: str = "l_foot_sole_center") -> Vec3:
    return frame_world_position(spec, forward_kinematics(spec, q), point)


# --------------------------------------------------------------------------- #
# Dynamics: provisional mass / COM / inertia
# --------------------------------------------------------------------------- #
@dataclass
class LinkInertial:
    name: str
    mass: float                 # kg
    com: Vec3                   # in the link frame, m
    inertia_diag: Vec3         # (Ixx, Iyy, Izz) about the COM, link-axis-aligned
    method: str                # how the tensor was approximated
    shape: Tuple[str, tuple]   # ("box", (dx,dy,dz)) or ("cylinder", (r, L)) for viz


def _inertia_solid_box(m: float, extents: Vec3) -> Vec3:
    dx, dy, dz = extents
    return (m / 12.0 * (dy * dy + dz * dz),
            m / 12.0 * (dx * dx + dz * dz),
            m / 12.0 * (dx * dx + dy * dy))


def _inertia_uniform_rod_z(m: float, length: float, radius: float) -> Vec3:
    transverse = m * length * length / 12.0
    axial = 0.5 * m * radius * radius
    return (transverse, transverse, axial)


def link_inertials(spec: dict) -> Dict[str, LinkInertial]:
    """Provisional inertial properties for the PHYSICAL links only.

    Virtual coupling links (is_physical: false) are skipped entirely -- they
    are massless mathematical abstractions and must never carry inertia.
    """
    syms = resolve_symbols(spec)
    out: Dict[str, LinkInertial] = {}
    dyn = (spec.get("dynamics", {}) or {}).get("links", {}) or {}
    for name, d in dyn.items():
        if not (d or {}).get("is_physical", False):
            continue
        m = float(d["mass"])
        com = resolve_vec3(d["com"], syms)
        inr = d["inertia"]
        method = inr["method"]
        if method == "solid_box":
            extents = resolve_vec3(inr["box"], syms)
            diag = _inertia_solid_box(m, extents)
            shape = ("box", tuple(extents))
        elif method == "uniform_rod_z":
            length = eval_expr(inr["length"], syms)
            radius = eval_expr(inr.get("radius", 0.0), syms)
            diag = _inertia_uniform_rod_z(m, length, radius)
            shape = ("cylinder", (radius, length))
        else:
            raise ValueError(f"link {name!r}: unknown inertia method {method!r}")
        out[name] = LinkInertial(name, m, com, diag, method, shape)
    return out


def physical_link_names(spec: dict) -> List[str]:
    return list(link_inertials(spec).keys())


def total_mass(spec: dict) -> float:
    return sum(li.mass for li in link_inertials(spec).values())


def analysis_gravity(spec: dict) -> float:
    return float((spec.get("analysis", {}) or {}).get("gravity", DEFAULT_GRAVITY))


def reference_poses(spec: dict) -> Dict[str, Dict[str, float]]:
    """Named joint configs.  A key like `*_hip_pitch` expands to every joint
    whose name ends with `hip_pitch` (both legs)."""
    poses = (spec.get("analysis", {}) or {}).get("reference_poses", {}) or {}
    jnames = joint_names(spec)
    out: Dict[str, Dict[str, float]] = {}
    for name, cfg in poses.items():
        expanded: Dict[str, float] = {}
        for k, v in (cfg or {}).items():
            if isinstance(k, str) and k.startswith("*_"):
                suffix = k[2:]
                for jn in jnames:
                    if jn == suffix or jn.endswith("_" + suffix):
                        expanded[jn] = float(v)
            else:
                expanded[k] = float(v)
        out[name] = expanded
    return out


def pose_qpos(spec: dict, cfg: Dict[str, float]) -> list:
    """A reference pose expanded to the full ordered joint vector (missing -> 0)."""
    return [float(cfg.get(name, 0.0)) for name in joint_names(spec)]


def actuator_control(spec: dict) -> Dict[str, Dict[str, float]]:
    """Per-joint PD position-servo gains: {joint: {'kp': .., 'dampratio': ..}}.

    Reads dynamics.actuators.control (kp / dampratio defaults + per-joint
    overrides). Returns an entry for every joint even if the block is absent
    (falling back to kp=0, which a caller can treat as 'no control').
    """
    ctrl = ((spec.get("dynamics", {}) or {}).get("actuators", {}) or {}).get("control", {}) or {}
    kp0 = float(ctrl.get("kp", 0.0))
    dr0 = float(ctrl.get("dampratio", 1.0))
    ov = ctrl.get("overrides", {}) or {}
    out: Dict[str, Dict[str, float]] = {}
    for name in joint_names(spec):
        jov = ov.get(name, {}) or {}
        out[name] = {"kp": float(jov.get("kp", kp0)),
                     "dampratio": float(jov.get("dampratio", dr0))}
    return out


def ground_params(spec: dict) -> dict:
    """analysis.ground settings for the dynamic MJCF."""
    g = (spec.get("analysis", {}) or {}).get("ground", {}) or {}
    return {
        "friction": [float(x) for x in g.get("friction", [1.0, 0.005, 0.0001])],
        "z_offset": float(g.get("z_offset", 0.0)),
    }


ExtraMass = Tuple[float, str]  # (mass_kg, frame_of_interest_name)


def link_world_coms(spec: dict, q: Dict[str, float] | None = None
                    ) -> List[Tuple[str, float, Vec3]]:
    """(name, mass, world COM position) for every physical link."""
    tf = forward_kinematics(spec, q)
    rows: List[Tuple[str, float, Vec3]] = []
    for name, li in link_inertials(spec).items():
        r, p = tf[name]
        rows.append((name, li.mass, vec_add(mat_vec(r, li.com), p)))
    return rows


def center_of_mass(spec: dict, q: Dict[str, float] | None = None,
                   extra_masses: Sequence[ExtraMass] | None = None
                   ) -> Tuple[float, Vec3, List[Tuple[str, float, Vec3]]]:
    """Return (total_mass, world COM, per-contributor rows).

    Rows are (name, mass, world position); extra point masses are included
    both in the totals and as rows named 'extra:<frame>'.
    """
    rows = link_world_coms(spec, q)
    if extra_masses:
        tf = forward_kinematics(spec, q)
        for m, frame in extra_masses:
            rows.append((f"extra:{frame}", float(m),
                         frame_world_position(spec, tf, frame)))
    m_tot = sum(m for _, m, _ in rows)
    if m_tot <= 0.0:
        raise ValueError("total mass is zero -- cannot compute a centre of mass")
    acc = (0.0, 0.0, 0.0)
    for _, m, pos in rows:
        acc = vec_add(acc, vec_scale(pos, m))
    return m_tot, vec_scale(acc, 1.0 / m_tot), rows


def potential_energy(spec: dict, q: Dict[str, float] | None = None,
                     g: float | None = None,
                     extra_masses: Sequence[ExtraMass] | None = None) -> float:
    """Gravitational PE, U = sum_i m_i g z_i  (z up)."""
    g = analysis_gravity(spec) if g is None else g
    _, com, rows = center_of_mass(spec, q, extra_masses)
    m_tot = sum(m for _, m, _ in rows)
    return m_tot * g * com[2]


# --------------------------------------------------------------------------- #
# Analysis: foot Jacobian
# --------------------------------------------------------------------------- #
def _world_joint_axes_origins(spec: dict, q: Dict[str, float] | None = None
                              ) -> List[Tuple[str, Vec3, Vec3]]:
    """(joint name, world unit axis, world joint-origin position) for each joint,
    in chain order (root -> foot)."""
    tf = forward_kinematics(spec, q)
    rows = []
    for jm in build_chain(spec):
        r_child, p_child = tf[jm.child]
        axis_world = normalize(mat_vec(r_child, jm.axis))
        rows.append((jm.name, axis_world, p_child))
    return rows


def foot_jacobian_geometric(spec: dict, q: Dict[str, float] | None = None,
                            point: str = "l_foot_sole_center"
                            ) -> Tuple[List[str], List[Vec3]]:
    """Analytic 3xN position Jacobian of `point`: column j = a_j x (p - o_j).

    Returns (joint_names, columns). All 6 joints are ancestors of the foot, so
    every column is non-zero.
    """
    tf = forward_kinematics(spec, q)
    p_foot = frame_world_position(spec, tf, point)
    names, cols = [], []
    for name, axis_world, o_j in _world_joint_axes_origins(spec, q):
        names.append(name)
        cols.append(cross(axis_world, vec_sub(p_foot, o_j)))
    return names, cols


def foot_jacobian_numeric(spec: dict, q: Dict[str, float] | None = None,
                          h: float = 1e-6, point: str = "l_foot_sole_center"
                          ) -> Tuple[List[str], List[Vec3]]:
    """Central-difference 3xN position Jacobian of `point`."""
    q = dict(q or {})
    names = joint_names(spec)
    cols = []
    for n in names:
        qp = dict(q); qp[n] = qp.get(n, 0.0) + h
        qm = dict(q); qm[n] = qm.get(n, 0.0) - h
        pp = frame_world_position(spec, forward_kinematics(spec, qp), point)
        pm = frame_world_position(spec, forward_kinematics(spec, qm), point)
        cols.append(tuple((a - b) / (2.0 * h) for a, b in zip(pp, pm)))  # type: ignore[arg-type]
    return names, cols


def jacobian_times(cols: Sequence[Vec3], qdot: Sequence[float]) -> Vec3:
    """J * qdot  ->  the 3-vector foot velocity."""
    out = (0.0, 0.0, 0.0)
    for col, qd in zip(cols, qdot):
        out = vec_add(out, vec_scale(col, qd))
    return out


def joint_torques_from_foot_force(spec: dict, force_world: Vec3,
                                  q: Dict[str, float] | None = None,
                                  point: str = "l_foot_sole_center"
                                  ) -> Dict[str, float]:
    """tau = J^T F : static joint torques in equilibrium with an external force
    `force_world` applied to the environment at the foot point."""
    names, cols = foot_jacobian_geometric(spec, q, point)
    return {n: dot(col, force_world) for n, col in zip(names, cols)}


# --------------------------------------------------------------------------- #
# Analysis: gravitational joint torques
# --------------------------------------------------------------------------- #
def gravity_joint_torques(spec: dict, q: Dict[str, float] | None = None,
                          g: float | None = None,
                          extra_masses: Sequence[ExtraMass] | None = None
                          ) -> Dict[str, float]:
    """Holding torque each joint must supply to balance gravity, tau_j = dU/dq_j.

    Base-fixed model: the pelvis is ground. This is the torque to hold the
    DISTAL leg segments (plus any `extra_masses`) in pose `q`. It does NOT
    include body-weight support unless you pass that as an extra point mass at
    the foot (see gravity_torques.py --carry-fraction).

    tau_j = sum_i m_i * g * [ a_j x (r_i - o_j) ]_z   over links i distal to j.
    """
    g = analysis_gravity(spec) if g is None else g
    tf = forward_kinematics(spec, q)
    chain = build_chain(spec)
    child_index = {jm.child: i for i, jm in enumerate(chain)}

    # (world position, mass, index of the deepest joint that moves this mass)
    masses: List[Tuple[Vec3, float, int]] = []
    for name, li in link_inertials(spec).items():
        r, p = tf[name]
        world_com = vec_add(mat_vec(r, li.com), p)
        last = child_index.get(name, -1)   # -1 => root link, no joint moves it
        masses.append((world_com, li.mass, last))
    for m, frame in (extra_masses or []):
        foi = _find_foi(spec, frame)
        last = child_index[foi["link"]]
        masses.append((frame_world_position(spec, tf, frame), float(m), last))

    axes_origins = _world_joint_axes_origins(spec, q)
    tau: Dict[str, float] = {}
    for k, (name, axis_world, o_j) in enumerate(axes_origins):
        t = 0.0
        for world_pos, m, last in masses:
            if last >= k:  # this mass is distal to joint k
                dz_dqk = cross(axis_world, vec_sub(world_pos, o_j))[2]
                t += m * g * dz_dqk
        tau[name] = t
    return tau


def gravity_joint_torques_fd(spec: dict, q: Dict[str, float] | None = None,
                             g: float | None = None,
                             extra_masses: Sequence[ExtraMass] | None = None,
                             eps: float = 1e-6) -> Dict[str, float]:
    """Same quantity by central-difference of the potential energy -- a check."""
    q = dict(q or {})
    out: Dict[str, float] = {}
    for n in joint_names(spec):
        qp = dict(q); qp[n] = qp.get(n, 0.0) + eps
        qm = dict(q); qm[n] = qm.get(n, 0.0) - eps
        up = potential_energy(spec, qp, g, extra_masses)
        um = potential_energy(spec, qm, g, extra_masses)
        out[n] = (up - um) / (2.0 * eps)
    return out
