"""Generate a pattern folder from a studio project.

Emits the same shapes as the hand-written patterns/bodice/ folder:
one .py file per piece (drafting math only), settings.py (render settings),
manifest.json, __init__.py (build/render entry points incl. render_web),
__main__.py (manifest-driven CLI), plus the .studio.json project itself so
the pattern stays editable in the studio.
"""

import ast
import json
import pprint
from pathlib import Path

from .model import POINT_REF_PARAMS

GEOMETRY_FUNCS = ("circle_h", "circle_v", "on_line", "along",
                  "intersect_lines", "liney", "linex",
                  "cubic_from_tangents", "catmull_rom_chain")


# ── Expression translation ────────────────────────────────────────────────────

class _Translate(ast.NodeTransformer):
    """studio expression AST → generated-python AST.

    * scalar names (measurements/derived) → m.<name>
    * A.x / A.y / front.A.x → A[0] / A[1] / front.A[0]
    * dist(A, B) → np.linalg.norm(A - B); sqrt(e) → np.sqrt(e)
    * substitutions (chord/width/height in tangent lengths) → given source
    """

    def __init__(self, scalar_names, substitutions=None):
        self.scalars = set(scalar_names)
        self.substitutions = substitutions or {}

    def visit_Name(self, node):
        if node.id in self.substitutions:
            return ast.parse(self.substitutions[node.id], mode="eval").body
        if node.id in self.scalars:
            return ast.Attribute(value=ast.Name(id="m", ctx=ast.Load()),
                                 attr=node.id, ctx=ast.Load())
        return node

    def visit_Attribute(self, node):
        if node.attr in ("x", "y"):
            base = self.visit(node.value)
            return ast.Subscript(value=base,
                                 slice=ast.Constant(0 if node.attr == "x" else 1),
                                 ctx=ast.Load())
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id == "dist" and len(node.args) == 2:
                diff = ast.BinOp(left=node.args[0], op=ast.Sub(),
                                 right=node.args[1])
                return ast.Call(
                    func=ast.parse("np.linalg.norm", mode="eval").body,
                    args=[diff], keywords=[])
            if node.func.id == "sqrt":
                node.func = ast.parse("np.sqrt", mode="eval").body
        return node


def pyexpr(expr, scalar_names, substitutions=None):
    """Translate a studio expression string to generated-python source."""
    tree = ast.parse(str(expr), mode="eval")
    new = _Translate(scalar_names, substitutions).visit(tree)
    ast.fix_missing_locations(new)
    return ast.unparse(new)


def _ref(ref):
    """Point reference "A" or "front.A" → python source (same spelling)."""
    return ref


# ── Piece file ────────────────────────────────────────────────────────────────

def _point_line(pt, scal):
    name = pt["name"]
    kind = pt.get("kind", "formula")
    E = lambda param, default=None: pyexpr(pt.get(param) or default, scal)
    if kind == "formula":
        return f"{name} = np.array([{E('x')}, {E('y')}])"
    if kind == "offset":
        return f"{name} = {_ref(pt['from'])} + np.array([{E('dx', '0')}, {E('dy', '0')}])"
    if kind == "midpoint":
        t = pt.get("t") or "0.5"
        if str(t).strip() == "0.5":
            return f"{name} = ({_ref(pt['p1'])} + {_ref(pt['p2'])}) / 2"
        return (f"{name} = {_ref(pt['p1'])} + ({E('t')})"
                f" * ({_ref(pt['p2'])} - {_ref(pt['p1'])})")
    if kind == "along":
        return (f"{name} = along({_ref(pt['from'])}, {_ref(pt['toward'])}, "
                f"{E('dist')})")
    if kind == "circle_h":
        return (f"{name} = circle_h({_ref(pt['center'])}, {E('radius')}, "
                f"{E('y')}, {pt.get('branch', 'left')!r})")
    if kind == "circle_v":
        return (f"{name} = circle_v({_ref(pt['center'])}, {E('radius')}, "
                f"{E('x')}, {pt.get('branch', 'down')!r})")
    if kind == "on_line":
        axis = "x" if pt.get("x") else "y"
        return (f"{name} = on_line({_ref(pt['p1'])}, {_ref(pt['p2'])}, "
                f"{axis}={E(axis)})")
    if kind == "intersect_lines":
        return (f"{name} = intersect_lines({_ref(pt['a1'])}, {_ref(pt['a2'])}, "
                f"{_ref(pt['b1'])}, {_ref(pt['b2'])})")
    raise ValueError(f"unknown point kind {kind!r}")


def _tangent_src(spec, from_src, to_src, endpoint, scal):
    """Python source for one tangent direction vector."""
    spec = spec or {}
    d = spec.get("dir", "chord")
    if d == "chord":       return f"({to_src} - {from_src})"
    if d == "horizontal+": return "np.array([1.0, 0.0])"
    if d == "horizontal-": return "np.array([-1.0, 0.0])"
    if d == "vertical+":   return "np.array([0.0, 1.0])"
    if d == "vertical-":   return "np.array([0.0, -1.0])"
    if d.startswith("toward:"):
        target = _ref(d[len("toward:"):])
        base = from_src if endpoint == "from" else to_src
        return f"({target} - {base})"
    if d == "angle":
        a = pyexpr(spec.get("angle", "0"), scal)
        return (f"np.array([np.cos(np.radians({a})), "
                f"np.sin(np.radians({a}))])")
    raise ValueError(f"unknown tangent dir {d!r}")


def _cubic_entry(seg, scal):
    f, t = _ref(seg["from"]), _ref(seg["to"])
    subs = {
        "chord":  f"np.linalg.norm({t} - {f})",
        "width":  f"abs({t}[0] - {f}[0])",
        "height": f"abs({t}[1] - {f}[1])",
    }
    tf = seg.get("tangent_from") or {}
    tt = seg.get("tangent_to") or {}
    d0 = _tangent_src(tf, f, t, "from", scal)
    d1 = _tangent_src(tt, f, t, "to", scal)
    l0 = pyexpr(tf.get("len") or "chord/3", scal, subs)
    l1 = pyexpr(tt.get("len") or "chord/3", scal, subs)
    return (f'("cubic_curve", lambda t, _p0={f}, _p1={t}, _d0={d0}, _l0={l0}, '
            f"_d1={d1}, _l1={l1}: "
            f"cubic_from_tangents(_p0, _p1, _d0, _l0, _d1, _l1, t), {f}, {t})")


def _segments_src(piece, scal):
    """Return (setup_lines, outline_src, curve_seam_src)."""
    setup, parts, seam_vars = [], [], []
    literal = []   # consecutive single-entry segments

    def flush():
        if literal:
            joined = ",\n        ".join(literal)
            parts.append(f"[\n        {joined},\n    ]")
            literal.clear()

    for si, seg in enumerate(piece.get("segments", [])):
        stype = seg["type"]
        if stype in ("line", "dart"):
            entry = f'("{stype}", {_ref(seg["from"])}, {_ref(seg["to"])})'
        elif stype == "quadratic":
            entry = (f'("quadratic", {_ref(seg["from"])}, {_ref(seg["cp"])}, '
                     f'{_ref(seg["to"])})')
        elif stype == "cubic":
            entry = _cubic_entry(seg, scal)
        elif stype == "catmull_chain":
            pts = ", ".join(_ref(r) for r in seg["through"])
            chain = f"catmull_rom_chain([{pts}])"
            if seg.get("curve_seam"):
                var = f"_seam_chain_{si}"
                setup.append(f"{var} = {chain}")
                chain = var
                seam_vars.append(var)
            flush()
            parts.append(chain)
            continue
        else:
            raise ValueError(f"unknown segment type {stype!r}")

        if seg.get("curve_seam") and stype == "cubic":
            var = f"_seam_curve_{si}"
            setup.append(f"{var} = [{entry}]")
            flush()
            parts.append(var)
            seam_vars.append(var)
        else:
            literal.append(entry)
    flush()

    outline_src = "\n        + ".join(parts) if parts else "[]"
    if len(parts) > 1:
        outline_src = f"(\n        {outline_src}\n    )"
    seam_src = " + ".join(seam_vars) if seam_vars else "[]"
    return setup, outline_src, seam_src


def _pairs_src(piece, key, scal):
    pairs = piece.get(key, [])
    if not pairs:
        return "[]"
    rows = ", ".join(f"({_ref(a)}, {_ref(b)})" for a, b in pairs)
    return f"[{rows}]"


def _piece_deps(piece, earlier_ids):
    """Earlier piece ids referenced by this piece (via qualified refs)."""
    deps = []

    def note(ref):
        if isinstance(ref, str) and "." in ref:
            piece_id = ref.partition(".")[0]
            if piece_id in earlier_ids and piece_id not in deps:
                deps.append(piece_id)

    def note_expr(expr):
        if not isinstance(expr, str):
            return
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return
        for node in ast.walk(tree):
            # front.A.x — the innermost Name is the piece id
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)):
                note(f"{node.value.value.id}.{node.value.attr}")
            # front.A (bare point ref, e.g. inside dist())
            elif (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.attr not in ("x", "y")):
                note(f"{node.value.id}.{node.attr}")

    for pt in piece.get("points", []):
        for param in POINT_REF_PARAMS.get(pt.get("kind", "formula"), []):
            note(pt.get(param, ""))
        for param, value in pt.items():
            if param not in ("name", "kind", "branch") and isinstance(value, str):
                note_expr(value)
    for seg in piece.get("segments", []):
        for param in ("from", "to", "cp"):
            note(seg.get(param, ""))
        for r in seg.get("through", []):
            note(r)
        for end in ("tangent_from", "tangent_to"):
            spec = seg.get(end) or {}
            if spec.get("dir", "").startswith("toward:"):
                note(spec["dir"][len("toward:"):])
    for key in ("construction_lines", "unclipped_construction_lines", "dart_lines"):
        for pair in piece.get(key, []):
            for r in pair:
                note(r)
    return deps


def gen_piece(project, piece, scal):
    """Generate the source of one piece file."""
    earlier = [p["id"] for p in project["pieces"]]
    earlier = earlier[:earlier.index(piece["id"])]
    deps = _piece_deps(piece, set(earlier))
    args = ", ".join(["m"] + deps)
    point_lines = [_point_line(pt, scal) for pt in piece.get("points", [])]
    setup, outline_src, seam_src = _segments_src(piece, scal)

    ann_rows = []
    for ann in piece.get("text_annotations", []):
        x = pyexpr(ann.get("x", "0"), scal)
        y = pyexpr(ann.get("y", "0"), scal)
        ann_rows.append(f"({ann.get('text', '')!r}, np.array([{x}, {y}]))")
    ann_src = ("[" + ",\n        ".join(ann_rows) + "]") if ann_rows else "[]"

    used = [f for f in GEOMETRY_FUNCS
            if any(f + "(" in line for line in point_lines + setup
                   + [outline_src])]
    geo_import = (f"from geometry import {', '.join(used)}\n" if used else "")

    point_names = ", ".join(f"{p['name']}={p['name']}"
                            for p in piece.get("points", []))
    body_points = "\n    ".join(point_lines) if point_lines else "pass"
    body_setup = ("\n    " + "\n    ".join(setup)) if setup else ""

    return f'''"""{piece.get('label', piece['id'])} piece of the {project['name']} pattern.

Generated by Pattern Studio — regenerate via
`python -m studio patterns/{project['id']}/{project['id']}.studio.json`.
"""

import numpy as np
from types import SimpleNamespace

{geo_import}

def build({args}):
    """Compute all points and outlines for this piece.
    m: SimpleNamespace of measurements and derived values."""

    # ── Points ────────────────────────────────────────────────────────────
    {body_points}
{body_setup}

    # ── Outline ───────────────────────────────────────────────────────────
    outline = {outline_src}

    return SimpleNamespace(
        {point_names + ',' if point_names else ''}
        outline=outline,
        construction_lines={_pairs_src(piece, "construction_lines", scal)},
        unclipped_construction_lines={_pairs_src(piece, "unclipped_construction_lines", scal)},
        dart_lines={_pairs_src(piece, "dart_lines", scal)},
        curve_seam_segments={seam_src},
        text_annotations={ann_src},
    )
'''


# ── settings.py ───────────────────────────────────────────────────────────────

def gen_settings(project):
    pieces = {}
    for piece in project["pieces"]:
        labels = piece.get("labels", {})
        pieces[piece["id"]] = {
            "label": piece.get("label", piece["id"]),
            "group": piece.get("group"),
            "fold_point": (piece.get("fold") or {}).get("point"),
            "style": piece.get("style") or {"fill": "#dce8f5", "stroke": "#2255aa"},
            "outline_labels": labels.get("outline", []),
            "interior_labels": labels.get("interior", []),
            "label_offsets": piece.get("label_offsets") or {},
        }
    body = pprint.pformat(pieces, indent=4, width=78, sort_dicts=False)
    return (f'"""Render settings for the {project["name"]} pattern.\n\n'
            f'Generated by Pattern Studio.\n"""\n\n'
            f"PIECES = {body}\n")


# ── manifest.json ─────────────────────────────────────────────────────────────

def gen_manifest(project):
    files = ["__init__.py"] + [f"{p['id']}.py" for p in project["pieces"]] \
            + ["settings.py"]
    manifest = {
        "id": project["id"],
        "name": project["name"],
        "source": project.get("source", ""),
        "files": files,
        "measurementGroups": project.get("measurementGroups", []),
        "options": project.get("options", []),
        "pieces": [
            {"id": p["id"], "label": p.get("label", p["id"]),
             **({"group": p["group"]} if p.get("group") else {})}
            for p in project["pieces"]
        ],
        "testSizes": project.get("testSizes", []),
    }
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


# ── __init__.py ───────────────────────────────────────────────────────────────

_INIT_TEMPLATE = '''"""{name} pattern.

Generated by Pattern Studio — reopen with
`python -m studio patterns/{id}/{id}.studio.json`.

CLI: python -m patterns.{id} --help
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from render import _write_svg, FONT_SIZE, fold_outline, mirror_point
from . import settings
{piece_imports}

_MANIFEST = json.loads((Path(__file__).parent / "manifest.json").read_text())
MEASUREMENT_GROUPS = {{g["id"]: [f["key"] for f in g["fields"]]
                      for g in _MANIFEST["measurementGroups"]}}
OPTIONAL_GROUPS = {{g["id"] for g in _MANIFEST["measurementGroups"]
                   if not g.get("required", True)}}


def _group_present(params, group):
    return all(params.get(k) is not None for k in MEASUREMENT_GROUPS[group])


def build_scalars(params):
    """Measurements + derived values as one namespace."""
    m = SimpleNamespace(**{{k: float(params[k])
                          for keys in MEASUREMENT_GROUPS.values()
                          for k in keys if params.get(k) is not None}})
{derived_lines}
    return m


def build_pieces(params):
    """Draft every piece available for the given measurements.
    Returns {{piece_id: SimpleNamespace}}."""
    m = build_scalars(params)
    built = {{}}
{piece_build_lines}
    return built


def _piece_svgs(params, prefix=None):
    seam_allowance = float(params.get("seam_allowance", 0.75))
    white_fill = bool(params.get("white_fill", False))
    built = build_pieces(params)
    out = {{}}
    for piece_id, ns in built.items():
        cfg = settings.PIECES[piece_id]
        outline = ns.outline
        outline_labels = {{n: getattr(ns, n) for n in cfg["outline_labels"]}}
        interior_labels = {{n: getattr(ns, n) for n in cfg["interior_labels"]}}
        if cfg.get("fold_point") and params.get("fold"):
            fold_x = float(getattr(ns, cfg["fold_point"])[0])
            outline = fold_outline(outline, fold_x)
            for name, pt in list(outline_labels.items()):
                if abs(float(pt[0]) - fold_x) > 1e-4:
                    outline_labels[name + "'"] = mirror_point(pt, fold_x)
        annotations = [(text, pos, "#aaa", FONT_SIZE * 2)
                       for text, pos in ns.text_annotations]
        path = f"{{prefix}}_{{piece_id}}.svg" if prefix else None
        result = _write_svg(
            path,
            outline,
            construction_lines=ns.construction_lines,
            dart_lines=ns.dart_lines,
            fill="white" if white_fill else cfg["style"]["fill"],
            stroke=cfg["style"]["stroke"],
            outline_labels=outline_labels,
            interior_labels=interior_labels,
            seam_allowance=seam_allowance,
            label_offsets=cfg["label_offsets"] or None,
            curve_seam_segments=ns.curve_seam_segments or None,
            curve_seam_allowance=(seam_allowance
                                  if ns.curve_seam_segments else None),
            unclipped_construction_lines=(ns.unclipped_construction_lines
                                          or None),
            text_annotations=annotations or None,
        )
        if path is None:
            svg, w, h = result
            out[piece_id] = svg
            out[f"{{piece_id}}_w"] = w
            out[f"{{piece_id}}_h"] = h
    return out


def render_web(params):
    """Web-frontend entry point: params keyed per manifest.json.
    Returns {{piece_id: svg, piece_id_w, piece_id_h, ...}}."""
    return _piece_svgs(params)


def render_files(params, prefix="{id}"):
    """Render every available piece to <prefix>_<piece>.svg files."""
    _piece_svgs(params, prefix=prefix)
'''


def gen_init(project, scal):
    piece_imports = "from . import " + ", ".join(
        p["id"] for p in project["pieces"])

    derived_lines = []
    for d in project.get("derived", []):
        derived_lines.append(
            f"    m.{d['name']} = {pyexpr(d.get('expr', '0'), scal)}")
    if not derived_lines:
        derived_lines.append("    # (no derived values)")

    build_lines = []
    earlier_ids = []
    for piece in project["pieces"]:
        deps = _piece_deps(piece, set(earlier_ids))
        earlier_ids.append(piece["id"])
        call_args = ", ".join(
            ["m"] + [f'{dep}=built["{dep}"]' for dep in deps])
        line = f'    built["{piece["id"]}"] = {piece["id"]}.build({call_args})'
        group = piece.get("group")
        if group:
            build_lines.append(f'    if _group_present(params, "{group}"):')
            build_lines.append("    " + line)
        else:
            build_lines.append(line)

    return _INIT_TEMPLATE.format(
        name=project["name"], id=project["id"],
        piece_imports=piece_imports,
        derived_lines="\n".join(derived_lines),
        piece_build_lines="\n".join(build_lines),
    )


_MAIN_TEMPLATE = '''"""CLI for the {name} pattern (generated by Pattern Studio)."""

import argparse
import json
from pathlib import Path

from . import render_files

_manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())

parser = argparse.ArgumentParser(
    prog="python -m patterns.{id}",
    description=f"Render {{_manifest['name']}} pieces to SVG.")
for grp in _manifest["measurementGroups"]:
    for f in grp["fields"]:
        parser.add_argument(f"--{{f['key']}}", type=float,
                            help=f["label"].lower())
for opt in _manifest.get("options", []):
    parser.add_argument(f"--{{opt['key'].replace('_', '-')}}",
                        action="store_true", help=opt["label"])
parser.add_argument("--prefix", type=str, default="{id}",
                    help="output filename prefix")
parser.add_argument("--seam-allowance", type=float, default=0.75,
                    help="seam allowance in inches (default 0.75)")
args = parser.parse_args()

params = {{"seam_allowance": args.seam_allowance}}
for grp in _manifest["measurementGroups"]:
    keys = [f["key"] for f in grp["fields"]]
    vals = {{k: getattr(args, k) for k in keys}}
    provided = [k for k, v in vals.items() if v is not None]
    if grp.get("required", True) or provided:
        missing = [k for k, v in vals.items() if v is None]
        if missing:
            parser.error(f"group {{grp['id']!r}} is missing: "
                         + ", ".join(f"--{{k}}" for k in missing))
        params.update(vals)
for opt in _manifest.get("options", []):
    params[opt["key"]] = getattr(args, opt["key"])

render_files(params, prefix=args.prefix)
'''


# ── Folder assembly ───────────────────────────────────────────────────────────

def generate(project):
    """Return {filename: content} for the pattern folder."""
    scal = [f["key"] for g in project.get("measurementGroups", [])
            for f in g.get("fields", [])]
    scal += [d["name"] for d in project.get("derived", [])]

    files = {}
    for piece in project["pieces"]:
        files[f"{piece['id']}.py"] = gen_piece(project, piece, scal)
    files["settings.py"] = gen_settings(project)
    files["manifest.json"] = gen_manifest(project)
    files["__init__.py"] = gen_init(project, scal)
    files["__main__.py"] = _MAIN_TEMPLATE.format(name=project["name"],
                                                 id=project["id"])
    files[f"{project['id']}.studio.json"] = (
        json.dumps(project, indent=2, ensure_ascii=False) + "\n")
    return files


def export(project, repo_root, confirm_overwrite=False):
    """Write the pattern folder into patterns/ and mirror it into docs/.

    Returns {"written": [paths]} or {"needs_confirm": reason}.
    Refuses to overwrite a folder that was not studio-generated (no
    .studio.json marker) unless confirm_overwrite is set.
    """
    repo_root = Path(repo_root)
    pattern_id = project["id"]
    target = repo_root / "patterns" / pattern_id
    marker = target / f"{pattern_id}.studio.json"
    if target.exists() and not marker.exists() and not confirm_overwrite:
        return {"needs_confirm":
                f"patterns/{pattern_id}/ exists and was not created by the "
                f"studio — confirm to overwrite it"}

    files = generate(project)
    written = []
    for root in (repo_root / "patterns", repo_root / "docs" / "patterns"):
        folder = root / pattern_id
        folder.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            if root.name != "patterns" and filename == "__main__.py":
                pass  # docs mirror still gets it — harmless and keeps sync simple
            path = folder / filename
            path.write_text(content, encoding="utf-8")
            written.append(str(path.relative_to(repo_root)))

    # register the pattern id in both index.json files
    for index_path in (repo_root / "patterns" / "index.json",
                       repo_root / "docs" / "patterns" / "index.json"):
        index = json.loads(index_path.read_text()) if index_path.exists() \
            else {"patterns": []}
        if pattern_id not in index["patterns"]:
            index["patterns"].append(pattern_id)
            index_path.write_text(json.dumps(index, indent=2) + "\n",
                                  encoding="utf-8")
            written.append(str(index_path.relative_to(repo_root)))

    return {"written": written}
