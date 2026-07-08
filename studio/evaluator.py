"""Evaluate a studio project into concrete geometry.

Expressions are restricted Python arithmetic, parsed with ast and evaluated
against the measurement/derived scalars and the defined points.  All
construction math is delegated to geometry.py — the same functions that
studio-generated pattern code calls — so the live preview, the parity tests,
and exported patterns share one implementation.
"""

import ast
import math

import numpy as np

import geometry
from .model import POINT_KINDS

N_SAMPLES = 33   # per curve segment, for the canvas


class EvalError(ValueError):
    pass


# ── Expressions ───────────────────────────────────────────────────────────────

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_SCALAR_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "min": min, "max": max,
}
_POINT_FUNCS = {"dist", "liney", "linex"}


def _resolve_point_node(node, points, pieces):
    """Resolve an AST node that must denote a point (Name or piece.Name)."""
    if isinstance(node, ast.Name):
        if node.id in points:
            return points[node.id]
        raise EvalError(f"unknown point {node.id!r}")
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        piece_id, name = node.value.id, node.attr
        if piece_id in pieces and name in pieces[piece_id]:
            return pieces[piece_id][name]
        raise EvalError(f"unknown point {piece_id}.{name}")
    raise EvalError("expected a point reference")


def eval_expr(expr, scalars, points=None, pieces=None):
    """Evaluate an expression to a float."""
    points = points or {}
    pieces = pieces or {}
    if isinstance(expr, (int, float)):
        return float(expr)
    if not isinstance(expr, str) or not expr.strip():
        raise EvalError("empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise EvalError(f"syntax error in {expr!r}: {e.msg}")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise EvalError(f"literal {node.value!r} is not a number")
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):  return a + b
            if isinstance(node.op, ast.Sub):  return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div):  return a / b
            if isinstance(node.op, ast.Pow):  return a ** b
            return a % b
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Name):
            if node.id in scalars:
                v = scalars[node.id]
                if v is None:
                    raise EvalError(f"{node.id!r} has no value (its own definition failed)")
                return float(v)
            raise EvalError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Attribute):
            # A.x / A.y / piece.A.x — coordinate access on a point
            if node.attr in ("x", "y"):
                pt = _resolve_point_node(node.value, points, pieces)
                return float(pt[0] if node.attr == "x" else pt[1])
            raise EvalError(f"unknown attribute .{node.attr} (only .x and .y)")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in _SCALAR_FUNCS:
                args = [ev(a) for a in node.args]
                return float(_SCALAR_FUNCS[fname](*args))
            if fname == "dist":
                p, q = (_resolve_point_node(a, points, pieces) for a in node.args)
                return float(np.linalg.norm(p - q))
            if fname in ("liney", "linex"):
                if len(node.args) != 3:
                    raise EvalError(f"{fname}(P1, P2, value) takes 3 arguments")
                p1 = _resolve_point_node(node.args[0], points, pieces)
                p2 = _resolve_point_node(node.args[1], points, pieces)
                v = ev(node.args[2])
                return float(getattr(geometry, fname)(p1, p2, v))
            raise EvalError(f"unknown function {fname!r}")
        raise EvalError(f"unsupported expression element: {ast.dump(node)[:40]}")

    try:
        return float(ev(tree))
    except EvalError:
        raise
    except ZeroDivisionError:
        raise EvalError(f"division by zero in {expr!r}")
    except Exception as e:
        raise EvalError(f"error in {expr!r}: {e}")


def _resolve_point_ref(ref, points, pieces):
    """Resolve a "A" or "piece.A" reference string to a point array."""
    if "." in ref:
        piece_id, _, name = ref.partition(".")
        if piece_id in pieces and name in pieces[piece_id]:
            return pieces[piece_id][name]
        raise EvalError(f"unknown point {ref!r}")
    if ref in points:
        return points[ref]
    raise EvalError(f"unknown point {ref!r}")


# ── Points ────────────────────────────────────────────────────────────────────

def eval_point(pt, scalars, points, pieces):
    kind = pt.get("kind", "formula")
    if kind not in POINT_KINDS:
        raise EvalError(f"unknown point kind {kind!r}")

    def E(param, default=None):
        expr = pt.get(param, default)
        if expr is None or expr == "":
            if default is not None:
                expr = default
            else:
                raise EvalError(f"missing {param!r}")
        return eval_expr(expr, scalars, points, pieces)

    def P(param):
        ref = pt.get(param, "")
        if not ref:
            raise EvalError(f"missing point reference {param!r}")
        return _resolve_point_ref(ref, points, pieces)

    if kind == "formula":
        return np.array([E("x"), E("y")])
    if kind == "offset":
        return P("from") + np.array([E("dx", "0"), E("dy", "0")])
    if kind == "midpoint":
        p1, p2 = P("p1"), P("p2")
        t = E("t", "0.5")
        return p1 + t * (p2 - p1)
    if kind == "along":
        return geometry.along(P("from"), P("toward"), E("dist"))
    if kind == "circle_h":
        return geometry.circle_h(P("center"), E("radius"), E("y"),
                                 pt.get("branch", "left"))
    if kind == "circle_v":
        return geometry.circle_v(P("center"), E("radius"), E("x"),
                                 pt.get("branch", "down"))
    if kind == "on_line":
        if pt.get("x"):
            return geometry.on_line(P("p1"), P("p2"), x=E("x"))
        return geometry.on_line(P("p1"), P("p2"), y=E("y"))
    if kind == "intersect_lines":
        return geometry.intersect_lines(P("a1"), P("a2"), P("b1"), P("b2"))
    raise EvalError(f"unhandled point kind {kind!r}")   # unreachable


# ── Segments ──────────────────────────────────────────────────────────────────

def _tangent_vector(spec, p_from, p_to, endpoint, scalars, points, pieces):
    """Direction-of-travel vector for one end of a cubic segment."""
    spec = spec or {}
    d = spec.get("dir", "chord")
    if d == "chord":       return p_to - p_from
    if d == "horizontal+": return np.array([1.0, 0.0])
    if d == "horizontal-": return np.array([-1.0, 0.0])
    if d == "vertical+":   return np.array([0.0, 1.0])
    if d == "vertical-":   return np.array([0.0, -1.0])
    if d.startswith("toward:"):
        target = _resolve_point_ref(d[len("toward:"):], points, pieces)
        base = p_from if endpoint == "from" else p_to
        return target - base
    if d == "angle":
        deg = eval_expr(spec.get("angle", "0"), scalars, points, pieces)
        rad = math.radians(deg)
        return np.array([math.cos(rad), math.sin(rad)])
    raise EvalError(f"unknown tangent dir {d!r}")


def _cubic_params(seg, scalars, points, pieces):
    """Resolve a cubic segment to (p0, p1, dir0, len0, dir1, len1)."""
    p0 = _resolve_point_ref(seg["from"], points, pieces)
    p1 = _resolve_point_ref(seg["to"], points, pieces)
    ctx = dict(scalars)
    ctx["chord"]  = float(np.linalg.norm(p1 - p0))
    ctx["width"]  = float(abs(p1[0] - p0[0]))
    ctx["height"] = float(abs(p1[1] - p0[1]))
    tf = seg.get("tangent_from") or {}
    tt = seg.get("tangent_to") or {}
    dir0 = _tangent_vector(tf, p0, p1, "from", scalars, points, pieces)
    dir1 = _tangent_vector(tt, p0, p1, "to", scalars, points, pieces)
    len0 = eval_expr(tf.get("len") or "chord/3", ctx, points, pieces)
    len1 = eval_expr(tt.get("len") or "chord/3", ctx, points, pieces)
    return p0, p1, dir0, len0, dir1, len1


def _quad_samples(p0, cp, p1, n=N_SAMPLES):
    ts = np.linspace(0, 1, n)[:, None]
    return (1 - ts)**2 * p0 + 2 * (1 - ts) * ts * cp + ts**2 * p1


def eval_segment(seg, scalars, points, pieces):
    """Return (render_entries, canvas_samples) for one segment definition.

    render_entries: list of outline tuples for render._write_svg
    canvas_samples: Nx2 np.array of sampled points along the segment
    """
    stype = seg["type"]
    if stype in ("line", "dart"):
        p0 = _resolve_point_ref(seg["from"], points, pieces)
        p1 = _resolve_point_ref(seg["to"], points, pieces)
        return [(stype, p0, p1)], np.array([p0, p1])
    if stype == "quadratic":
        p0 = _resolve_point_ref(seg["from"], points, pieces)
        p1 = _resolve_point_ref(seg["to"], points, pieces)
        cp = _resolve_point_ref(seg["cp"], points, pieces)
        return [("quadratic", p0, cp, p1)], _quad_samples(p0, cp, p1)
    if stype == "cubic":
        p0, p1, dir0, len0, dir1, len1 = _cubic_params(seg, scalars, points, pieces)
        def func(t, _p0=p0, _p1=p1, _d0=dir0, _l0=len0, _d1=dir1, _l1=len1):
            return geometry.cubic_from_tangents(_p0, _p1, _d0, _l0, _d1, _l1, t)
        samples = func(np.linspace(0, 1, N_SAMPLES))
        return [("cubic_curve", func, p0, p1)], samples
    if stype == "catmull_chain":
        pts = [_resolve_point_ref(r, points, pieces) for r in seg["through"]]
        entries = geometry.catmull_rom_chain(pts)
        samples = []
        for i, (_, func, _p0, _p1) in enumerate(entries):
            smp = func(np.linspace(0, 1, N_SAMPLES))
            samples.extend(smp if i == 0 else smp[1:])
        return entries, np.array(samples)
    raise EvalError(f"unknown segment type {stype!r}")


# ── Whole-project evaluation ──────────────────────────────────────────────────

def _group_fields(project, group_id):
    for grp in project.get("measurementGroups", []):
        if grp.get("id") == group_id:
            return [f["key"] for f in grp.get("fields", [])]
    return []


def evaluate(project, values):
    """Evaluate every piece of the project at the given measurement values.

    values: {measurement key: float}; optional-group measurements may be
    absent, in which case pieces gated on that group are marked skipped.

    Returns:
      {"scalars": {...}, "errors": [...],           # derived-value errors
       "pieces": {piece_id: {
           "skipped": bool,
           "points": {name: np.array},
           "point_order": [names],
           "errors": [{"where", "name", "message"}],
           "outline": [render entries],             # for _write_svg
           "canvas_segments": [{"index", "type", "samples": Nx2 array}],
           "curve_seam_segments": [render entries],
           "construction_lines": [(p0, p1)],
           "unclipped_construction_lines": [(p0, p1)],
           "dart_lines": [(p0, p1)],
           "annotations": [(text, np.array)],
       }}}
    """
    scalars = {k: float(v) for k, v in values.items() if v is not None}
    top_errors = []
    for d in project.get("derived", []):
        try:
            scalars[d["name"]] = eval_expr(d.get("expr", ""), scalars)
        except EvalError as e:
            scalars[d["name"]] = None
            top_errors.append({"where": "derived", "name": d.get("name", "?"),
                               "message": str(e)})

    pieces_ns = {}
    out_pieces = {}
    for piece in project.get("pieces", []):
        piece_id = piece["id"]
        result = {
            "skipped": False, "points": {}, "point_order": [], "errors": [],
            "outline": [], "canvas_segments": [], "curve_seam_segments": [],
            "construction_lines": [], "unclipped_construction_lines": [],
            "dart_lines": [], "annotations": [],
        }
        out_pieces[piece_id] = result

        group = piece.get("group")
        if group and any(k not in scalars for k in _group_fields(project, group)):
            result["skipped"] = True
            continue

        points = {}
        for pt in piece.get("points", []):
            name = pt.get("name", "?")
            result["point_order"].append(name)
            try:
                points[name] = eval_point(pt, scalars, points, pieces_ns)
            except (EvalError, ValueError) as e:
                result["errors"].append({"where": "point", "name": name,
                                         "message": str(e)})
        result["points"] = points

        for si, seg in enumerate(piece.get("segments", [])):
            try:
                entries, samples = eval_segment(seg, scalars, points, pieces_ns)
                result["outline"].extend(entries)
                result["canvas_segments"].append(
                    {"index": si, "type": seg["type"], "samples": samples})
                if seg.get("curve_seam"):
                    result["curve_seam_segments"].extend(
                        e for e in entries if e[0] == "cubic_curve")
            except (EvalError, ValueError, KeyError) as e:
                result["errors"].append({"where": "segment", "name": f"#{si + 1}",
                                         "message": str(e)})

        for key in ("construction_lines", "unclipped_construction_lines",
                    "dart_lines"):
            for pair in piece.get(key, []):
                try:
                    result[key].append(tuple(
                        _resolve_point_ref(r, points, pieces_ns) for r in pair))
                except EvalError as e:
                    result["errors"].append({"where": key, "name": "–".join(pair),
                                             "message": str(e)})

        for ai, ann in enumerate(piece.get("text_annotations", [])):
            try:
                pos = np.array([
                    eval_expr(ann.get("x", "0"), scalars, points, pieces_ns),
                    eval_expr(ann.get("y", "0"), scalars, points, pieces_ns)])
                result["annotations"].append((ann.get("text", ""), pos))
            except EvalError as e:
                result["errors"].append({"where": "annotation", "name": f"#{ai + 1}",
                                         "message": str(e)})

        pieces_ns[piece_id] = points

    return {"scalars": scalars, "errors": top_errors, "pieces": out_pieces}


# ── Bridging to the master renderer ───────────────────────────────────────────

def render_args(piece_def, piece_result, seam_allowance=0.75, white_fill=False):
    """Build the render._write_svg keyword arguments for one evaluated piece."""
    from render import FONT_SIZE

    points = piece_result["points"]
    labels = piece_def.get("labels", {})
    style = piece_def.get("style") or {}
    annotations = [
        (text, pos, "#aaa", FONT_SIZE * 2)
        for text, pos in piece_result["annotations"]
    ]
    return dict(
        outline=piece_result["outline"],
        construction_lines=piece_result["construction_lines"],
        dart_lines=piece_result["dart_lines"],
        fill="white" if white_fill else style.get("fill", "#dce8f5"),
        stroke=style.get("stroke", "#2255aa"),
        outline_labels={n: points[n] for n in labels.get("outline", []) if n in points},
        interior_labels={n: points[n] for n in labels.get("interior", []) if n in points},
        seam_allowance=seam_allowance,
        label_offsets=piece_def.get("label_offsets") or None,
        curve_seam_segments=piece_result["curve_seam_segments"] or None,
        curve_seam_allowance=seam_allowance if piece_result["curve_seam_segments"] else None,
        unclipped_construction_lines=piece_result["unclipped_construction_lines"] or None,
        text_annotations=annotations or None,
    )
