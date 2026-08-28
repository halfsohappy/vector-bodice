"""Slack pattern — Foundation 3.

Built on patterns/trouser — see front_panel.py/back_panel.py for the
tightening-offset derivation. The waistband recipe (p.588) is identical
across all three from-scratch pant foundations, so it's imported directly
from patterns/trouser rather than duplicated.

CLI: python -m patterns.slack --help
"""

from render import _write_svg, _curve_groups, rectangle_dims
from patterns.trouser import waistband, fly, belt_loops, pocket, legline
from patterns.trouser import back_panel as _t_back
from . import front_panel, back_panel, creaseline
from . import settings


# ── Build ─────────────────────────────────────────────────────────────────────

def build(waist_arc_front, waist_arc_back, hip_arc_front, hip_arc_back,
          crotch_depth, pant_length,
          fly_front=False, belt_loops_opt=False, pockets=False,
          flared_leg=False, flare_position="at_knee",
          creaseline_flare=False):
    """Draft every piece of the slack.  Returns {piece_id: SimpleNamespace}."""
    back = back_panel.build(hip_arc_back, waist_arc_back, crotch_depth, pant_length,
                            flared_leg=flared_leg, flare_position=flare_position)
    front = front_panel.build(hip_arc_front, waist_arc_front, crotch_depth, pant_length,
                              flared_leg=flared_leg, flare_position=flare_position,
                              back_crotch_width=back.crotch_width,
                              creaseline_flare=creaseline_flare)
    band = waistband.build(front.finished_waist, back.finished_waist)
    pieces = {"front_panel": front, "back_panel": back, "waistband": band}
    if creaseline_flare:
        # A seam replaces the back creaseline, so the back becomes two
        # pieces and the plain back panel is dropped (book p.611).
        side, inner = creaseline.back_halves(
            back, _t_back.curve_hip, _t_back.curve_inseam, _t_back.curve_crotch)
        pieces.pop("back_panel")
        pieces["back_side"] = side
        pieces["back_inner"] = inner
    if fly_front:
        pieces["fly"] = fly.build_fly()
        pieces["shield"] = fly.build_shield()
    if belt_loops_opt:
        pieces["belt_loops"] = belt_loops.build()
    if pockets:
        pieces.update(pocket.build(front))
    return pieces


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
        notches=getattr(ns, "notches", None),
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
        notches=getattr(ns, "notches", None),
    )


def _notion_args(ns, style, seam_allowance, white_fill):
    """Render args for a plain rectangular notion (fly, shield, belt loops)."""
    return dict(
        outline=ns.outline,
        construction_lines=ns.construction_lines,
        dart_lines=ns.dart_lines,
        fill="white" if white_fill else style["fill"],
        stroke=style["stroke"],
        outline_labels={n: getattr(ns, n) for n in ("A", "B", "C", "D")},
        interior_labels={},
        seam_allowance=seam_allowance,
        waist_detect=False,
        merge_consecutive=False,
        notches=getattr(ns, "notches", None),
    )


def _addon_args(pieces, args, seam_allowance, white_fill):
    """Attach render args for whichever optional pieces are present."""
    for pid in ("pocket_facing", "pocket_pouch", "pocket_backing", "pocket_lining"):
        if pid in pieces:
            ns = pieces[pid]
            args[pid] = dict(
                outline=ns.outline,
                construction_lines=ns.construction_lines,
                dart_lines=ns.dart_lines,
                fill="white" if white_fill else settings.POCKET_STYLE["fill"],
                stroke=settings.POCKET_STYLE["stroke"],
                outline_labels={n: getattr(ns, n) for n in ("A","B","C","D","E","F","G","X")
                                if hasattr(ns, n)},
                interior_labels={},
                seam_allowance=seam_allowance,
                seam_allowance_fn=pocket.entry_seam_allowance_fn(ns, seam_allowance),
                waist_detect=False,
                merge_consecutive=False,
            )
    for pid, style in (("fly", settings.FLY_STYLE),
                       ("shield", settings.SHIELD_STYLE),
                       ("belt_loops", settings.BELT_LOOP_STYLE)):
        if pid in pieces:
            args[pid] = _notion_args(pieces[pid], style, seam_allowance, white_fill)
    return args


def _all_svg_args(pieces, seam_allowance, white_fill):
    args = {
        "front_panel": _panel_args(pieces["front_panel"], settings.FRONT_CORNER_LABELS,
                                   settings.FRONT_INTERIOR_LABELS, settings.FRONT_STYLE,
                                   seam_allowance, white_fill),
        "waistband": _waistband_args(pieces["waistband"], seam_allowance, white_fill),
    }
    if "back_panel" in pieces:
        args["back_panel"] = _panel_args(
            pieces["back_panel"], settings.BACK_CORNER_LABELS,
            settings.BACK_INTERIOR_LABELS, settings.BACK_STYLE,
            seam_allowance, white_fill)
    for pid, corners in (("back_side", ("O", "C", "A", "waist", "hem_out", "hem_seam")),
                         ("back_inner", ("S", "X", "I", "A", "waist", "hem_out", "hem_seam"))):
        if pid in pieces:
            ns = pieces[pid]
            args[pid] = dict(
                outline=ns.outline,
                construction_lines=ns.construction_lines,
                dart_lines=ns.dart_lines,
                fill="white" if white_fill else settings.BACK_STYLE["fill"],
                stroke=settings.BACK_STYLE["stroke"],
                outline_labels={n: getattr(ns, n) for n in corners if hasattr(ns, n)},
                interior_labels={},
                seam_allowance=seam_allowance,
                curve_seam_segments=_curve_groups(ns.outline),
                curve_seam_allowance=seam_allowance,
                notches=getattr(ns, "notches", None),
            )
    return _addon_args(pieces, args, seam_allowance, white_fill)


# ── Render: SVG strings (web interface) ───────────────────────────────────────

def render_web(params):
    """Generic web-frontend entry point (see patterns/bodice/__init__.py)."""
    pieces = build(
        float(params["waist_arc_front"]), float(params["waist_arc_back"]),
        float(params["hip_arc_front"]), float(params["hip_arc_back"]),
        float(params["crotch_depth"]), float(params["pant_length"]),
        fly_front=bool(params.get("fly_front", False)),
        belt_loops_opt=bool(params.get("belt_loops", False)),
        pockets=bool(params.get("pockets", False)),
        flared_leg=bool(params.get("flared_leg", False)),
        flare_position=str(params.get("flare_position", "at_knee")),
        creaseline_flare=bool(params.get("creaseline_flare", False)),
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
           crotch_depth, pant_length, prefix="slack", seam_allowance=0.75,
           fly_front=False, belt_loops=False, pockets=False,
           flared_leg=False, flare_position="at_knee",
          creaseline_flare=False):
    """Render every piece of the slack to <prefix>_<piece>.svg files."""
    pieces = build(waist_arc_front, waist_arc_back, hip_arc_front, hip_arc_back,
                   crotch_depth, pant_length,
                   fly_front=fly_front, belt_loops_opt=belt_loops, pockets=pockets,
                   flared_leg=flared_leg, flare_position=flare_position,
                   creaseline_flare=creaseline_flare)
    args = _all_svg_args(pieces, seam_allowance, white_fill=False)
    for piece_id, kw in args.items():
        rect = rectangle_dims(kw["outline"], kw.get("seam_allowance", 0), kw.get("seam_allowance_fn"),
                              kw.get("waist_detect", True), kw.get("merge_consecutive", True))
        _write_svg(f"{prefix}_{piece_id}.svg", kw.pop("outline"), **kw)
        if rect:
            print(f"  {piece_id} is a plain rectangle — draft with a ruler: "
                  f"{rect['finished_w']:.2f} x {rect['finished_h']:.2f}in finished, "
                  f"{rect['cut_w']:.2f} x {rect['cut_h']:.2f}in cut")
