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
            if "axis" in j:                       # fixed / locked joints may have none
                mj["axis"] = _mirror_axis(j["axis"])
            if "positive_rotation" in j:
                mj["positive_rotation"] = _mirror_text(j["positive_rotation"])
            if "purpose" in j:
                mj["purpose"] = _mirror_text(j["purpose"])
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


_APPEND_KEYS = ("links", "joints", "frames_of_interest")


def _merge_include(spec: dict, frag: dict, frag_name: str) -> dict:
    """Compose an `include:` fragment into `spec` -- ADDITIVE, not a deep merge.

    Lists (links / joints / frames_of_interest) are appended (name collisions
    are an error); `dynamics.links` and the free-form `upper_body` / `electronics`
    parameter blocks are dict-merged; `analysis` sub-dicts are deep-merged.
    The fragment's `meta`, `base`, `mirror`, `extends` are ignored.
    """
    out = copy.deepcopy(spec)
    for key in _APPEND_KEYS:
        have = {e["name"] for e in out.get(key, []) or []}
        for e in frag.get(key, []) or []:
            if e.get("name") in have:
                raise ValueError(f"include {frag_name!r}: {key} entry {e['name']!r} "
                                 f"already exists in the base spec")
            out.setdefault(key, []).append(copy.deepcopy(e))
    fd = (frag.get("dynamics", {}) or {}).get("links", {}) or {}
    for name, d in fd.items():
        out.setdefault("dynamics", {}).setdefault("links", {})
        if name in out["dynamics"]["links"]:
            raise ValueError(f"include {frag_name!r}: dynamics.links[{name!r}] already exists")
        out["dynamics"]["links"][name] = copy.deepcopy(d)
    for blk in ("upper_body", "electronics", "analysis"):
        if blk in frag:
            out[blk] = _deep_merge(out.get(blk, {}) or {}, frag[blk])
    return out


def load_spec(path: str | None = None) -> dict:
    """Load a YAML spec: apply `extends` (deep merge) then `include` (additive
    compose) then `mirror` expansion.

    The returned spec is fully flat: `links`, `joints`, `dynamics.links` and
    `frames_of_interest` list every side / subsystem explicitly.
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
    for inc in spec.pop("include", []) or []:
        frag = _load_raw(os.path.join(here, inc))
        spec = _merge_include(spec, frag, inc)
    _resolve_mounts(spec)
    mirrored = bool(spec.get("mirror"))
    spec = _apply_mirror(spec)
    spec["_source"] = src
    spec["_mirrored"] = mirrored
    return spec


def _resolve_mounts(spec: dict) -> None:
    """Wire up `mount_from` joints (electronics).  A joint with
        mount_from: electronics.jetson.mount
    takes its `parent` and `origin_expr` from the named mount preset in
    `electronics.mounts[<that value>]` = {link, x, y, z}.  `mount_from` is kept
    on the joint so the wiring can be re-resolved after a placement override.
    """
    mounts = (spec.get("electronics", {}) or {}).get("mounts", {}) or {}
    for j in spec.get("joints", []) or []:
        ref = j.get("mount_from")
        if not ref:
            continue
        node = spec
        for k in ref.split("."):
            node = node[k]
        preset = mounts[node]
        j["parent"] = preset["link"]
        j["origin_expr"] = [float(preset.get("x", 0.0)),
                            float(preset.get("y", 0.0)),
                            float(preset.get("z", 0.0))]


def apply_electronics_layout(spec: dict, layout: Dict[str, str]) -> dict:
    """Set `electronics.<item>.mount = <preset>` for each item in `layout`
    ({'jetson': 'torso_mid', 'battery': 'pelvis_low'}) and re-resolve the mount
    joints, in place.  Returns the spec."""
    for item, preset in layout.items():
        spec.setdefault("electronics", {}).setdefault(item, {})["mount"] = preset
    _resolve_mounts(spec)
    return spec


def electronics_layouts(spec: dict) -> Dict[str, Dict[str, str]]:
    return dict((spec.get("electronics", {}) or {}).get("layouts", {}) or {})


def base_spec(spec: dict) -> dict:
    """Floating vs fixed base config (MJCF).  Defaults to a fixed (welded) base."""
    b = spec.get("base", {}) or {}
    return {
        "type": b.get("type", "fixed"),
        "rest_pose": b.get("rest_pose"),
        "rest_height": (float(b["rest_height"]) if b.get("rest_height") is not None else None),
    }


def _flatten_scalars(node, prefix, out):
    if isinstance(node, dict):
        for k, v in node.items():
            _flatten_scalars(v, f"{prefix}{k}_" if prefix else f"{k}_", out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out[prefix[:-1]] = float(node)


def resolve_symbols(spec: dict) -> Dict[str, float]:
    """name -> float map that origin_expr / com / box expressions may reference.

    Sources: the flat `provisional_geometry` symbols (kinematic geometry), plus
    every numeric leaf of the `upper_body` and `electronics` blocks flattened
    with underscores (e.g. upper_body.torso.com_z -> `torso_com_z`).
    """
    syms: Dict[str, float] = {}
    pg = spec.get("provisional_geometry", {}) or {}
    for k, v in pg.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            syms[k] = float(v)
    for blk in ("upper_body", "electronics"):
        _flatten_scalars(spec.get(blk, {}) or {}, "", syms)
    return syms


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


def mat_transpose(A: Mat3) -> Mat3:
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def so3_log(R: Mat3) -> Vec3:
    """Rotation matrix -> rotation vector (axis * angle).  Valid for the small
    orientation errors this codebase produces; not hardened for angle ~ pi."""
    tr = R[0][0] + R[1][1] + R[2][2]
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    angle = math.acos(c)
    if angle < 1e-12:
        return (0.0, 0.0, 0.0)
    s = 2.0 * math.sin(angle)
    return ((R[2][1] - R[1][2]) / s * angle,
            (R[0][2] - R[2][0]) / s * angle,
            (R[1][0] - R[0][1]) / s * angle)


def solve_linear(A: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """Solve A x = b for a square matrix via Gaussian elimination + partial pivot."""
    n = len(b)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col] or 1e-15
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / pv
            for k in range(col, n + 1):
                M[r][k] -= f * M[col][k]
    return [M[i][n] / (M[i][i] or 1e-15) for i in range(n)]


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
    fixed: bool = False          # weld (type: fixed) OR locked -- 0 DOF, no actuator

    @property
    def axis_norm(self) -> float:
        return vec_norm(self.axis)

    @property
    def actuated(self) -> bool:
        return not self.fixed

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
        fixed = j["type"] == "fixed" or bool(j.get("locked", False))
        required = ("name", "type", "parent", "child", "origin_expr") if fixed else REQUIRED_JOINT_KEYS
        missing = [k for k in required if k not in j]
        if missing:
            raise KeyError(f"joint {j.get('name', '<unnamed>')} missing keys: {missing}")
        lim = j.get("limits", {"lower": 0.0, "upper": 0.0})
        ov = overrides.get(j["name"], {}) or {}
        chain.append(JointModel(
            name=j["name"],
            jtype=j["type"],
            parent=j["parent"],
            child=j["child"],
            origin=resolve_vec3(j["origin_expr"], syms),
            axis=tuple(float(a) for a in j.get("axis", (0.0, 0.0, 1.0))),  # type: ignore[arg-type]
            lower=float(lim["lower"]),
            upper=float(lim["upper"]),
            effort=float(ov.get("effort", defaults.get("effort", 0.0))),
            velocity=float(ov.get("velocity", defaults.get("velocity", 0.0))),
            purpose=" ".join(str(j.get("purpose", "")).split()),
            positive_rotation=" ".join(str(j.get("positive_rotation", "")).split()),
            fixed=fixed,
        ))
    return chain


def joint_names(spec: dict) -> List[str]:
    """Every joint in the tree, actuated or not."""
    return [jm.name for jm in build_chain(spec)]


def actuated_joint_names(spec: dict) -> List[str]:
    """Joints with a DOF -- the ones that get a PD servo and a qpos slot."""
    return [jm.name for jm in build_chain(spec) if jm.actuated]


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
        angle = 0.0 if jm.fixed else float(q.get(jm.name, 0.0))
        t_origin: Transform = (IDENTITY3, jm.origin)
        t_joint: Transform = (rot_axis_angle(jm.axis, angle), (0.0, 0.0, 0.0))
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


def _inertia_solid_sphere(m: float, radius: float) -> Vec3:
    i = 0.4 * m * radius * radius
    return (i, i, i)


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
        m = eval_expr(d["mass"], syms)   # number or expression over the symbols
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
        elif method == "solid_sphere":
            radius = eval_expr(inr["radius"], syms)
            diag = _inertia_solid_sphere(m, radius)
            shape = ("sphere", (radius,))
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
    """A reference pose as the ordered vector of ACTUATED joint angles (missing -> 0).
    This is the MJCF hinge-qpos / ctrl order."""
    return [float(cfg.get(name, 0.0)) for name in actuated_joint_names(spec)]


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
    for name in actuated_joint_names(spec):
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


def whole_body_inertia(spec: dict, q: Dict[str, float] | None = None,
                       about: str = "com") -> Mat3:
    """3x3 inertia tensor of all physical links, in the WORLD frame, about
    either the whole-body COM (`about="com"`) or the base-frame origin
    (`about="base"`).  Parallel-axis:  I = sum_i [ R_i I_i R_iᵀ
                                              + m_i (|d|² E3 - d dᵀ) ]  with
    d = com_i - reference.  `I_i` is the per-link diagonal about its own COM.
    """
    tf = forward_kinematics(spec, q)
    inertials = link_inertials(spec)
    if about == "com":
        _, ref, _ = center_of_mass(spec, q)
    else:
        ref = tf[spec["frame_conventions"]["base_frame"]][1]

    total = [[0.0] * 3 for _ in range(3)]
    for name, li in inertials.items():
        r, p = tf[name]
        com_w = vec_add(mat_vec(r, li.com), p)
        d = vec_sub(com_w, ref)
        d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        # R I_diag Rᵀ
        idiag = li.inertia_diag
        rot_i = [[sum(r[i][k] * idiag[k] * r[j][k] for k in range(3)) for j in range(3)]
                 for i in range(3)]
        for i in range(3):
            for j in range(3):
                pa = li.mass * ((d2 if i == j else 0.0) - d[i] * d[j])
                total[i][j] += rot_i[i][j] + pa
    return tuple(tuple(row) for row in total)  # type: ignore[return-value]


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
        if jm.fixed:
            continue
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

    # actuated-joint ancestors of each link (tree, not a single chain)
    parent_link = {jm.child: jm.parent for jm in chain}
    joint_to_child = {jm.child: jm for jm in chain}
    base = spec["frame_conventions"]["base_frame"]

    def ancestors(link: str) -> set:
        anc, cur = set(), link
        while cur in parent_link and cur != base:
            jm = joint_to_child[cur]
            if jm.actuated:
                anc.add(jm.name)
            cur = jm.parent
        return anc

    link_anc = {name: ancestors(name) for name in link_inertials(spec)}

    masses: List[Tuple[Vec3, float, set]] = []
    for name, li in link_inertials(spec).items():
        r, p = tf[name]
        world_com = vec_add(mat_vec(r, li.com), p)
        masses.append((world_com, li.mass, link_anc[name]))
    for m, frame in (extra_masses or []):
        foi = _find_foi(spec, frame)
        masses.append((frame_world_position(spec, tf, frame), float(m), ancestors(foi["link"])))

    tau: Dict[str, float] = {}
    for name, axis_world, o_j in _world_joint_axes_origins(spec, q):
        t = 0.0
        for world_pos, m, anc in masses:
            if name in anc:   # this mass hangs off joint `name`
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


# --------------------------------------------------------------------------- #
# Task-space / inverse kinematics for one leg
# --------------------------------------------------------------------------- #
# Used by scripts/weight_shift.py to turn a desired pelvis/COM displacement into
# joint targets, instead of hard-coding a hip-roll trajectory.  It is a plain
# damped-least-squares Newton loop over the already-validated forward_kinematics
# and a spatial Jacobian -- no learned model, no gain tuning.

def leg_joint_names(spec: dict, prefix: str) -> List[str]:
    return [jm.name for jm in build_chain(spec) if jm.name.startswith(prefix)]


def foot_spatial_jacobian(spec: dict, q: Dict[str, float], prefix: str, point: str
                          ) -> Tuple[List[str], List[Tuple[float, ...]]]:
    """6xN spatial Jacobian of `point` w.r.t. one leg's joints.

    Returns (joint_names, columns) where each column is the 6-vector
    [linear-velocity-of-point (3); angular-velocity-of-foot (3)].
    Column j (revolute):  [ a_j x (p - o_j) ; a_j ]  with a_j, o_j in world.
    """
    tf = forward_kinematics(spec, q)
    p = frame_world_position(spec, tf, point)
    names, cols = [], []
    for jm in build_chain(spec):
        if not jm.name.startswith(prefix):
            continue
        r_child, p_child = tf[jm.child]
        a = normalize(mat_vec(r_child, jm.axis))
        lin = cross(a, vec_sub(p, p_child))
        names.append(jm.name)
        cols.append((lin[0], lin[1], lin[2], a[0], a[1], a[2]))
    return names, cols


def leg_ik(spec: dict, prefix: str, point: str,
           target_pos: Vec3, target_rot: Mat3,
           q_seed: Dict[str, float],
           free_joints: Sequence[str] | None = None,
           task_rows: Sequence[int] | None = None,
           iters: int = 120, tol: float = 1e-8,
           damping: float = 1e-6, max_step: float = 0.25
           ) -> Tuple[Dict[str, float], float]:
    """Damped-least-squares IK for one leg.

    Drives the selected components of the foot error to zero by moving the
    selected joints.  The 6-D foot error is
        [ target_pos - point ; so3_log(target_rot * R_footᵀ) ]   (pelvis frame)
    `task_rows` picks which of those 6 to constrain (default all 6);
    `free_joints` picks which joints move (default all 6 of this leg).
    For a pure lateral shift: free = [<prefix>hip_roll, <prefix>ankle_roll],
    task_rows = [1, 3]  (foot y-position, foot roll).

    Joint limits are respected. Returns ({joint: angle}, max |constrained error|).
    """
    q = {n: float(q_seed.get(n, 0.0)) for n in joint_names(spec)}
    all_leg = leg_joint_names(spec, prefix)
    free = list(free_joints) if free_joints is not None else all_leg
    rows = list(task_rows) if task_rows is not None else [0, 1, 2, 3, 4, 5]
    lims = joint_limits(spec)
    foot_link = prefix + "foot"
    residual = 1e9
    for _ in range(iters):
        tf = forward_kinematics(spec, q)
        p = frame_world_position(spec, tf, point)
        r = tf[foot_link][0]
        full_err = list(vec_sub(target_pos, p)) + list(so3_log(mat_mul(target_rot, mat_transpose(r))))
        err = [full_err[i] for i in rows]
        residual = max(abs(x) for x in err)
        if residual < tol:
            break
        names, cols = foot_spatial_jacobian(spec, q, prefix, point)
        idx = [names.index(n) for n in free]
        m, k = len(rows), len(free)
        # reduced Jacobian  Jr[m][k];  dq = Jrᵀ (Jr Jrᵀ + λI)⁻¹ err
        jr = [[cols[idx[j]][rows[i]] for j in range(k)] for i in range(m)]
        jjt = [[sum(jr[i][c] * jr[l][c] for c in range(k)) + (damping if i == l else 0.0)
                for l in range(m)] for i in range(m)]
        y = solve_linear(jjt, err)
        dq = [sum(jr[i][c] * y[i] for i in range(m)) for c in range(k)]
        for n, d in zip(free, dq):
            d = max(-max_step, min(max_step, d))
            q[n] = min(lims[n][1], max(lims[n][0], q[n] + d))
    return ({n: q[n] for n in all_leg}, residual)


def convex_hull_2d(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """CCW convex hull (Andrew's monotone chain) of 2-D points."""
    pts = sorted(set((round(p[0], 9), round(p[1], 9)) for p in points))
    if len(pts) <= 2:
        return pts

    def crs(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List = []
    for p in pts:
        while len(lower) >= 2 and crs(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List = []
    for p in reversed(pts):
        while len(upper) >= 2 and crs(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_signed_margin(poly: Sequence[Sequence[float]], q: Sequence[float]) -> float:
    """Signed distance from q to the polygon boundary: + inside, - outside."""
    if len(poly) < 3:
        return -1e9
    inside, mind, n = True, 1e18, len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        nx, ny = ey, -ex                       # outward normal for a CCW polygon
        L = math.hypot(nx, ny) or 1.0
        d = ((q[0] - a[0]) * nx + (q[1] - a[1]) * ny) / L
        if d > 1e-12:
            inside = False
        mind = min(mind, abs(d))
    return mind if inside else -mind


def nominal_foot_poses(spec: dict, cfg: Dict[str, float]
                       ) -> Dict[str, Tuple[Vec3, Mat3, Vec3]]:
    """For a reference pose, the pelvis-frame (sole position, foot rotation,
    ankle-centre position) of every foot -- the IK targets for zero shift."""
    tf = forward_kinematics(spec, cfg)
    out = {}
    for foi in spec.get("frames_of_interest", []) or []:
        prefix = foi["link"][:foi["link"].index("foot")]
        sole = frame_world_position(spec, tf, foi["name"])
        out[prefix] = (sole, tf[foi["link"]][0], tf[foi["link"]][1])
    return out
