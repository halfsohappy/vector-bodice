"""INES skirt pattern.

Adapted from the "INES Skirt" tutorial (The Mystical Attic) — see
skirt_panel.py / waistband.py / pocket.py for how each piece's baked-in
seam allowance was reverse-engineered into this repo's finished-size +
render-time-SA convention.

Eight pieces from five drafting files: front/back panel and front/back
waistband are geometrically identical shapes (skirt_panel.build /
waistband.build called twice with different arguments); pocket side 1/2
are mirror images (pocket.build(mirror=...)); tie and pocket_bias are each
a single simple rectangle, cut 2.

CLI: python -m patterns.ines_skirt --help
"""

from render import _write_svg, rectangle_dims
from . import skirt_panel, waistband, pocket, tie, pocket_bias
from . import settings


# ── Build ─────────────────────────────────────────────────────────────────────

def build(waist, skirt_length, hem_allowance):
    """Draft every piece of the skirt.  Returns {piece_id: SimpleNamespace}."""
    front_panel = skirt_panel.build(waist, skirt_length, hem_allowance)
    back_panel  = skirt_panel.build(waist, skirt_length, hem_allowance)

    front_waistband = waistband.build(waistband.front_length(waist))
    back_waistband  = waistband.build(waistband.back_length(waist))

    pocket_side1 = pocket.build(mirror=False)
    pocket_side2 = pocket.build(mirror=True)

    return {
        "front_panel": front_panel, "back_panel": back_panel,
        "front_waistband": front_waistband, "back_waistband": back_waistband,
        "pocket_side1": pocket_side1, "pocket_side2": pocket_side2,
        "tie": tie.build(), "pocket_bias": pocket_bias.build(),
    }


# ── Piece assembly helpers ────────────────────────────────────────────────────

def _labels(ns, names):
    return {name: getattr(ns, name) for name in names}


def _panel_args(ns, style, seam_allowance, white_fill):
    return dict(
        outline=ns.outline,
        construction_lines=ns.construction_lines,
        dart_lines=ns.dart_lines,
        fill="white" if white_fill else style["fill"],
        stroke=style["stroke"],
        outline_labels=_labels(ns, settings.PANEL_OUTLINE_LABELS),
        interior_labels=_labels(ns, settings.PANEL_INTERIOR_LABELS),
        seam_allowance=seam_allowance,
        curve_seam_segments=ns.curve_seam_segments,
        curve_seam_allowance=seam_allowance,
    )


def _waistband_args(ns, seam_allowance, white_fill):
    return dict(
        outline=ns.outline,
        construction_lines=ns.construction_lines,
        dart_lines=ns.dart_lines,
        fill="white" if white_fill else settings.WAISTBAND_STYLE["fill"],
        stroke=settings.WAISTBAND_STYLE["stroke"],
        outline_labels=_labels(ns, settings.WAISTBAND_OUTLINE_LABELS),
        interior_labels={},
        seam_allowance=seam_allowance,
        seam_allowance_fn=settings.waistband_seam_allowance_fn(ns, seam_allowance),
        waist_detect=False,
        merge_consecutive=False,
    )


def _pocket_args(ns, seam_allowance, white_fill):
    return dict(
        outline=ns.outline,
        construction_lines=ns.construction_lines,
        dart_lines=ns.dart_lines,
        fill="white" if white_fill else settings.POCKET_STYLE["fill"],
        stroke=settings.POCKET_STYLE["stroke"],
        outline_labels=_labels(ns, settings.POCKET_OUTLINE_LABELS),
        interior_labels={},
        seam_allowance=seam_allowance,
        waist_detect=False,
    )


def _notion_args(ns, style, labels, white_fill, text_annotations=None):
    return dict(
        outline=ns.outline,
        construction_lines=ns.construction_lines,
        dart_lines=ns.dart_lines,
        fill="white" if white_fill else style["fill"],
        stroke=style["stroke"],
        outline_labels=_labels(ns, labels),
        interior_labels={},
        seam_allowance=0,
        text_annotations=text_annotations,
    )


def _all_svg_args(pieces, seam_allowance, white_fill):
    return {
        "front_panel": _panel_args(pieces["front_panel"], settings.FRONT_PANEL_STYLE,
                                   seam_allowance, white_fill),
        "back_panel":  _panel_args(pieces["back_panel"], settings.BACK_PANEL_STYLE,
                                   seam_allowance, white_fill),
        "front_waistband": _waistband_args(pieces["front_waistband"], seam_allowance, white_fill),
        "back_waistband":  _waistband_args(pieces["back_waistband"], seam_allowance, white_fill),
        "pocket_side1": _pocket_args(pieces["pocket_side1"], seam_allowance, white_fill),
        "pocket_side2": _pocket_args(pieces["pocket_side2"], seam_allowance, white_fill),
        "tie": _notion_args(pieces["tie"], settings.TIE_STYLE,
                            settings.TIE_OUTLINE_LABELS, white_fill),
        "pocket_bias": _notion_args(
            pieces["pocket_bias"], settings.BIAS_STYLE, settings.BIAS_OUTLINE_LABELS,
            white_fill, settings.format_annotations(pieces["pocket_bias"].text_annotations)),
    }


# ── Render: SVG strings (web interface) ───────────────────────────────────────

def _render_all(waist, skirt_length, hem_allowance, seam_allowance, white_fill):
    pieces = build(waist, skirt_length, hem_allowance)
    args = _all_svg_args(pieces, seam_allowance, white_fill)
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


DEFAULT_HEM_ALLOWANCE = 0.625   # 5/8in, ~1.5cm — matches manifest.json's default


def render_web(params):
    """Generic web-frontend entry point (see patterns/bodice/__init__.py)."""
    return _render_all(
        float(params["waist"]), float(params["skirt_length"]),
        float(params.get("hem_allowance", DEFAULT_HEM_ALLOWANCE)),
        float(params.get("seam_allowance", 0.75)),
        bool(params.get("white_fill", False)),
    )


# ── Render: SVG files ─────────────────────────────────────────────────────────

def render(waist, skirt_length, hem_allowance=DEFAULT_HEM_ALLOWANCE,
           prefix="ines_skirt", seam_allowance=0.75):
    """Render every piece of the skirt to <prefix>_<piece>.svg files."""
    pieces = build(waist, skirt_length, hem_allowance)
    args = _all_svg_args(pieces, seam_allowance, white_fill=False)
    for piece_id, kw in args.items():
        rect = rectangle_dims(kw["outline"], kw.get("seam_allowance", 0), kw.get("seam_allowance_fn"),
                              kw.get("waist_detect", True), kw.get("merge_consecutive", True))
        _write_svg(f"{prefix}_{piece_id}.svg", kw.pop("outline"), **kw)
        if rect:
            print(f"  {piece_id} is a plain rectangle — draft with a ruler: "
                  f"{rect['finished_w']:.2f} x {rect['finished_h']:.2f}in finished, "
                  f"{rect['cut_w']:.2f} x {rect['cut_h']:.2f}in cut")
