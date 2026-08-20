"""Slack pattern — Foundation 3.

Built on patterns/trouser — see front_panel.py/back_panel.py for the
tightening-offset derivation. The waistband recipe (p.588) is identical
across all three from-scratch pant foundations, so it's imported directly
from patterns/trouser rather than duplicated.

CLI: python -m patterns.slack --help
"""

from render import _write_svg, _curve_groups, rectangle_dims
from patterns.trouser import waistband
from . import front_panel, back_panel
from . import settings


# ── Build ─────────────────────────────────────────────────────────────────────

def build(waist_arc_front, waist_arc_back, hip_arc_front, hip_arc_back,
          crotch_depth, pant_length):
    """Draft every piece of the slack.  Returns {piece_id: SimpleNamespace}."""
    front = front_panel.build(hip_arc_front, waist_arc_front, crotch_depth, pant_length)
    back = back_panel.build(hip_arc_back, waist_arc_back, crotch_depth, pant_length)
    waist_total = 2 * (waist_arc_front + waist_arc_back)
    band = waistband.build(waist_total)
    return {"front_panel": front, "back_panel": back, "waistband": band}


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


def _waistband_args(ns, seam_allowance, white_fill):
    return dict(
        outline=ns.outline,
        construction_lines=ns.construction_lines,
        dart_lines=ns.dart_lines,
        fill="white" if white_fill else settings.WAISTBAND_STYLE["fill"],
        stroke=settings.WAISTBAND_STYLE["stroke"],
        outline_labels={name: getattr(ns, name) for name in settings.WAISTBAND_OUTLINE_LABELS},
        interior_labels={},
        seam_allowance=seam_allowance,
        waist_detect=False,
        merge_consecutive=False,
    )


def _all_svg_args(pieces, seam_allowance, white_fill):
    return {
        "front_panel": _panel_args(pieces["front_panel"], settings.FRONT_CORNER_LABELS,
                                   settings.FRONT_INTERIOR_LABELS, settings.FRONT_STYLE,
                                   seam_allowance, white_fill),
        "back_panel": _panel_args(pieces["back_panel"], settings.BACK_CORNER_LABELS,
                                  settings.BACK_INTERIOR_LABELS, settings.BACK_STYLE,
                                  seam_allowance, white_fill),
        "waistband": _waistband_args(pieces["waistband"], seam_allowance, white_fill),
    }


# ── Render: SVG strings (web interface) ───────────────────────────────────────

def render_web(params):
    """Generic web-frontend entry point (see patterns/bodice/__init__.py)."""
    pieces = build(
        float(params["waist_arc_front"]), float(params["waist_arc_back"]),
        float(params["hip_arc_front"]), float(params["hip_arc_back"]),
        float(params["crotch_depth"]), float(params["pant_length"]),
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
           crotch_depth, pant_length, prefix="slack", seam_allowance=0.75):
    """Render every piece of the slack to <prefix>_<piece>.svg files."""
    pieces = build(waist_arc_front, waist_arc_back, hip_arc_front, hip_arc_back,
                   crotch_depth, pant_length)
    args = _all_svg_args(pieces, seam_allowance, white_fill=False)
    for piece_id, kw in args.items():
        rect = rectangle_dims(kw["outline"], kw.get("seam_allowance", 0), kw.get("seam_allowance_fn"),
                              kw.get("waist_detect", True), kw.get("merge_consecutive", True))
        _write_svg(f"{prefix}_{piece_id}.svg", kw.pop("outline"), **kw)
        if rect:
            print(f"  {piece_id} is a plain rectangle — draft with a ruler: "
                  f"{rect['finished_w']:.2f} x {rect['finished_h']:.2f}in finished, "
                  f"{rect['cut_w']:.2f} x {rect['cut_h']:.2f}in cut")
