"""Project-JSON schema helpers and structural validation.

A project file (<id>.studio.json) is the single source the studio edits.
Expression *syntax/value* errors are reported by evaluator.py at eval time;
this module checks structure: identifiers, uniqueness, and references.
"""

import re

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Names with meaning inside expressions — measurements, derived values and
# points must not shadow them.
RESERVED = {"sqrt", "abs", "min", "max", "dist", "liney", "linex",
            "chord", "width", "height",
            "seam_allowance", "white_fill"}

POINT_KINDS = {
    "formula":         ["x", "y"],
    "offset":          ["from", "dx", "dy"],
    "midpoint":        ["p1", "p2"],            # optional "t" (default 0.5)
    "along":           ["from", "toward", "dist"],
    "circle_h":        ["center", "radius", "y", "branch"],   # branch: left|right
    "circle_v":        ["center", "radius", "x", "branch"],   # branch: down|up
    "on_line":         ["p1", "p2"],            # plus "x" or "y"
    "intersect_lines": ["a1", "a2", "b1", "b2"],
}

# Which params of each kind are point references (the rest are expressions)
POINT_REF_PARAMS = {
    "formula":         [],
    "offset":          ["from"],
    "midpoint":        ["p1", "p2"],
    "along":           ["from", "toward"],
    "circle_h":        ["center"],
    "circle_v":        ["center"],
    "on_line":         ["p1", "p2"],
    "intersect_lines": ["a1", "a2", "b1", "b2"],
}

SEGMENT_TYPES = ("line", "dart", "quadratic", "cubic", "catmull_chain")
TANGENT_DIRS = ("chord", "horizontal+", "horizontal-", "vertical+", "vertical-")
DEFAULT_STYLE = {"fill": "#dce8f5", "stroke": "#2255aa"}


def new_project(pattern_id="new_pattern"):
    """A minimal starter project."""
    return {
        "id": pattern_id,
        "name": pattern_id.replace("_", " "),
        "source": "",
        "measurementGroups": [
            {"id": "main", "label": "Main", "required": True, "fields": []},
        ],
        "derived": [],
        "options": [],
        "testSizes": [],
        "pieces": [
            {
                "id": "piece1", "label": "Piece 1",
                "group": None, "fold": None,
                "points": [], "segments": [],
                "construction_lines": [], "unclipped_construction_lines": [],
                "dart_lines": [],
                "labels": {"outline": [], "interior": []},
                "label_offsets": {},
                "style": dict(DEFAULT_STYLE),
                "text_annotations": [],
            },
        ],
    }


def _point_ref_ok(ref, own_points, earlier_pieces):
    """A point reference is "A" (own piece, already defined) or "piece.A"."""
    if "." in ref:
        piece_id, _, name = ref.partition(".")
        return piece_id in earlier_pieces and name in earlier_pieces[piece_id]
    return ref in own_points


def validate(project):
    """Return a list of {path, message} structural issues."""
    issues = []

    def err(path, message):
        issues.append({"path": path, "message": message})

    pid = project.get("id", "")
    if not IDENT_RE.match(pid):
        err("id", f"pattern id {pid!r} must be a valid identifier")
    if not project.get("name"):
        err("name", "pattern name is empty")

    # measurements
    scalar_names = set()
    group_ids = set()
    optional_groups = set()
    for gi, grp in enumerate(project.get("measurementGroups", [])):
        gpath = f"measurementGroups[{gi}]"
        gid = grp.get("id", "")
        if not IDENT_RE.match(gid):
            err(gpath, f"group id {gid!r} must be a valid identifier")
        if gid in group_ids:
            err(gpath, f"duplicate group id {gid!r}")
        group_ids.add(gid)
        if not grp.get("required", True):
            optional_groups.add(gid)
        for fi, f in enumerate(grp.get("fields", [])):
            key = f.get("key", "")
            fpath = f"{gpath}.fields[{fi}]"
            if not IDENT_RE.match(key):
                err(fpath, f"measurement key {key!r} must be a valid identifier")
            elif key in RESERVED:
                err(fpath, f"measurement key {key!r} is reserved")
            elif key in scalar_names:
                err(fpath, f"duplicate measurement key {key!r}")
            scalar_names.add(key)

    # derived values
    for di, d in enumerate(project.get("derived", [])):
        name = d.get("name", "")
        dpath = f"derived[{di}]"
        if not IDENT_RE.match(name):
            err(dpath, f"derived name {name!r} must be a valid identifier")
        elif name in RESERVED:
            err(dpath, f"derived name {name!r} is reserved")
        elif name in scalar_names:
            err(dpath, f"duplicate name {name!r} (already a measurement or derived value)")
        scalar_names.add(name)

    # options
    option_keys = set()
    for oi, o in enumerate(project.get("options", [])):
        key = o.get("key", "")
        opath = f"options[{oi}]"
        if not IDENT_RE.match(key):
            err(opath, f"option key {key!r} must be a valid identifier")
        elif key in RESERVED or key in scalar_names or key in option_keys:
            err(opath, f"option key {key!r} collides with another name")
        option_keys.add(key)

    # pieces
    earlier_pieces = {}   # piece id → set of point names
    piece_ids = set()
    for pi, piece in enumerate(project.get("pieces", [])):
        ppath = f"pieces[{pi}]"
        piece_id = piece.get("id", "")
        if not IDENT_RE.match(piece_id):
            err(ppath, f"piece id {piece_id!r} must be a valid identifier")
        if piece_id in ("settings", "manifest", "__init__", "__main__"):
            err(ppath, f"piece id {piece_id!r} collides with a generated filename")
        if piece_id in piece_ids:
            err(ppath, f"duplicate piece id {piece_id!r}")
        piece_ids.add(piece_id)
        if piece.get("group") and piece["group"] not in optional_groups:
            err(ppath, f"piece group {piece['group']!r} is not an optional measurement group")

        own_points = set()
        for qi, pt in enumerate(piece.get("points", [])):
            qpath = f"{ppath}.points[{qi}]"
            name = pt.get("name", "")
            kind = pt.get("kind", "formula")
            if not IDENT_RE.match(name):
                err(qpath, f"point name {name!r} must be a valid identifier")
            elif name in RESERVED or name in scalar_names:
                err(qpath, f"point name {name!r} collides with a measurement/derived name")
            elif name in own_points:
                err(qpath, f"duplicate point name {name!r}")
            if kind not in POINT_KINDS:
                err(qpath, f"unknown point kind {kind!r}")
            else:
                for ref_param in POINT_REF_PARAMS[kind]:
                    ref = pt.get(ref_param, "")
                    if not _point_ref_ok(ref, own_points, earlier_pieces):
                        err(qpath, f"{ref_param}: unknown point {ref!r} "
                                   f"(points may only reference earlier points)")
                if kind == "on_line" and not (pt.get("x") or pt.get("y")):
                    err(qpath, "on_line needs an x or y expression")
            own_points.add(name)

        def check_ref(ref, path, what="point"):
            if not _point_ref_ok(ref, own_points, earlier_pieces):
                err(path, f"unknown {what} {ref!r}")

        for si, seg in enumerate(piece.get("segments", [])):
            spath = f"{ppath}.segments[{si}]"
            stype = seg.get("type", "")
            if stype not in SEGMENT_TYPES:
                err(spath, f"unknown segment type {stype!r}")
                continue
            if stype == "catmull_chain":
                through = seg.get("through", [])
                if len(through) < 3:
                    err(spath, "catmull_chain needs at least 3 points")
                for ref in through:
                    check_ref(ref, spath)
            else:
                check_ref(seg.get("from", ""), spath)
                check_ref(seg.get("to", ""), spath)
                if stype == "quadratic":
                    check_ref(seg.get("cp", ""), spath, "control point")
                if stype == "cubic":
                    for end in ("tangent_from", "tangent_to"):
                        spec = seg.get(end) or {}
                        d = spec.get("dir", "chord")
                        if d.startswith("toward:"):
                            check_ref(d[len("toward:"):], spath)
                        elif d not in TANGENT_DIRS and d != "angle":
                            err(spath, f"{end}: unknown tangent dir {d!r}")

        for key in ("construction_lines", "unclipped_construction_lines", "dart_lines"):
            for li, pair in enumerate(piece.get(key, [])):
                for ref in pair:
                    check_ref(ref, f"{ppath}.{key}[{li}]")

        labels = piece.get("labels", {})
        for kind_key in ("outline", "interior"):
            for ref in labels.get(kind_key, []):
                check_ref(ref, f"{ppath}.labels.{kind_key}")
        if piece.get("fold") and piece["fold"].get("point") not in own_points:
            err(f"{ppath}.fold", f"fold point {piece['fold'].get('point')!r} unknown")

        earlier_pieces[piece_id] = own_points

    return issues
