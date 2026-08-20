"""Basic 2-dart skirt sloper pattern.

Adapted from Pattern Making for Fashion Design, 5th ed., "Skirt Draft"
(p.48-50) — see front_panel.py/back_panel.py for the drafting notes and
approximations. This is the foundation patterns/skirt_aline builds on,
which in turn patterns/culotte builds on.

CLI: python -m patterns.skirt_basic --help
"""

from render import _write_svg, _curve_groups, rectangle_dims
from . import front_panel, back_panel, dart_chart
from . import settings


# ── Build ─────────────────────────────────────────────────────────────────────

def dart_info(waist_arc_front, waist_arc_back, hip_arc_front, hip_arc_back):
    """Resolve the shared dart-chart lookup from all four arc measurements.
    hip_total - waist_total = 2 * [(front hip-waist) + (back hip-waist)],
    since each "arc" measurement is one quarter of the full circumference."""
    diff = 2 * ((hip_arc_front - waist_arc_front) + (hip_arc_back - waist_arc_back))
    return dart_chart.lookup(diff)


def build(waist_arc_front, waist_arc_back, hip_arc_front, hip_arc_back,
          hip_depth_front, hip_depth_back, skirt_length):
    """Draft both panels.  Returns {"front_panel": ns, "back_panel": ns}."""
    info = dart_info(waist_arc_front, waist_arc_back, hip_arc_front, hip_arc_back)
    front = front_panel.build(hip_arc_front, hip_depth_front, skirt_length,
                              info.front_count, info.front_intake)
    back = back_panel.build(hip_arc_back, hip_depth_back, skirt_length,
                            info.back_count, info.back_intake)
    return {"front_panel": front, "back_panel": back}


# ── Piece assembly helpers ────────────────────────────────────────────────────

def _panel_args(ns, corner_labels, interior_labels, style, seam_allowance, white_fill):
    outline_labels = {name: getattr(ns, name) for name in corner_labels}
    outline_labels.update(settings.dart_outline_labels(ns))
    return dict(
        outline=ns.outline,
        construction_lines=ns.construction_lines,
        dart_lines=ns.dart_lines,
        fill="white" if white_fill else style["fill"],
        stroke=style["stroke"],
        outline_labels=outline_labels,
        interior_labels={name: getattr(ns, name) for name in interior_labels},
        seam_allowance=seam_allowance,
        curve_seam_segments=_curve_groups(ns.outline),
        curve_seam_allowance=seam_allowance,
    )


def _all_svg_args(pieces, seam_allowance, white_fill):
    return {
        "front_panel": _panel_args(pieces["front_panel"], settings.FRONT_CORNER_LABELS,
                                   settings.FRONT_INTERIOR_LABELS, settings.FRONT_STYLE,
                                   seam_allowance, white_fill),
        "back_panel": _panel_args(pieces["back_panel"], settings.BACK_CORNER_LABELS,
                                  settings.BACK_INTERIOR_LABELS, settings.BACK_STYLE,
                                  seam_allowance, white_fill),
    }


# ── Render: SVG strings (web interface) ───────────────────────────────────────

def render_web(params):
    """Generic web-frontend entry point (see patterns/bodice/__init__.py)."""
    pieces = build(
        float(params["waist_arc_front"]), float(params["waist_arc_back"]),
        float(params["hip_arc_front"]), float(params["hip_arc_back"]),
        float(params["hip_depth_front"]), float(params["hip_depth_back"]),
        float(params["skirt_length"]),
    )
    args = _all_svg_args(pieces, float(params.get("seam_allowance", 0.75)),
                         bool(params.get("white_fill", False)))
    out = {}
    for piece_id, kw in args.items():
        rect = rectangle_dims(kw["outline"], kw.get("seam_allowance", 0), kw.get("seam_allowance_fn"),
                              kw.get("waist_detect", True), kw.get("merge_consecutive", True))
        svg, w, h = _write_svg(None, kw.pop("outline"), **kw)
        out[piece_id] = svg
        out[f"{piece_id}_w"] = w
        out[f"{piece_id}_h"] = h
        out[f"{piece_id}_rect"] = rect
    return out


# ── Render: SVG files ─────────────────────────────────────────────────────────

def render(waist_arc_front, waist_arc_back, hip_arc_front, hip_arc_back,
           hip_depth_front, hip_depth_back, skirt_length,
           prefix="skirt_basic", seam_allowance=0.75):
    """Render both panels to <prefix>_<piece>.svg files."""
    pieces = build(waist_arc_front, waist_arc_back, hip_arc_front, hip_arc_back,
                   hip_depth_front, hip_depth_back, skirt_length)
    args = _all_svg_args(pieces, seam_allowance, white_fill=False)
    for piece_id, kw in args.items():
        rect = rectangle_dims(kw["outline"], kw.get("seam_allowance", 0), kw.get("seam_allowance_fn"),
                              kw.get("waist_detect", True), kw.get("merge_consecutive", True))
        _write_svg(f"{prefix}_{piece_id}.svg", kw.pop("outline"), **kw)
        if rect:
            print(f"  {piece_id} is a plain rectangle — draft with a ruler: "
                  f"{rect['finished_w']:.2f} x {rect['finished_h']:.2f}in finished, "
                  f"{rect['cut_w']:.2f} x {rect['cut_h']:.2f}in cut")
