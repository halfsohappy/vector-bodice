"""Bodice-with-sleeve pattern.

Three pieces: front bodice, back bodice, sleeve.  Each piece file holds the
drafting math; settings.py holds the render settings; manifest.json
describes the pattern to the web frontend and the CLI (measurements,
options, pieces); this file assembles the pieces and exposes the
build/render entry points.

CLI: python -m patterns.bodice --help
"""

import numpy as np
from types import SimpleNamespace

import render as _render
from render import _write_svg, fold_outline, mirror_point, _curve_groups
from . import front_bodice, back_bodice, sleeve
from . import settings


# ── Build ─────────────────────────────────────────────────────────────────────

def build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
          deepen_bust_dart=False):
    """Draft the front and back bodice pieces and merge them into a single
    namespace (all points, derived measurements, outlines, and construction
    lines).  Returns a SimpleNamespace."""
    front = front_bodice.build(alpha, beta, gamma, delta, epsilon, zeta,
                               eta, theta, deepen_bust_dart=deepen_bust_dart)
    back = back_bodice.build(front)

    merged = {**vars(front), **vars(back)}
    for key in ("outline", "construction_lines", "dart_lines"):
        merged.pop(key, None)

    return SimpleNamespace(
        **merged,
        front_bodice=front.outline,
        back_bodice=back.outline,
        construction_lines=front.construction_lines + back.construction_lines,
        front_dart_lines=front.dart_lines,
        back_dart_lines=back.dart_lines,
    )


build_sleeve = sleeve.build


# ── Piece assembly helpers ────────────────────────────────────────────────────

def _labels(ns, names):
    return {name: getattr(ns, name) for name in names}


def _front_piece_args(bk, fold):
    """Front outline + labels, optionally folded on the center-front line."""
    outline = bk.front_bodice
    labels = _labels(bk, settings.FRONT_OUTLINE_LABELS)
    if fold:
        fold_line_x = bk.M[0]  # = D[0] = center-front x
        outline = fold_outline(bk.front_bodice, fold_line_x)
        # Add primed mirrored labels for every point not sitting on the fold line
        for name, pt in list(labels.items()):
            pt_arr = np.asarray(pt, float)
            if abs(pt_arr[0] - fold_line_x) > 1e-4:
                labels[name + "'"] = mirror_point(pt_arr, fold_line_x)
    return outline, labels


def _bodice_svg_args(bk, fold, seam_allowance, white_fill=False):
    """Build the two _write_svg keyword-argument dicts for the bodice pieces."""
    shared_interior = _labels(bk, settings.SHARED_INTERIOR_LABELS)
    front_outline, front_labels = _front_piece_args(bk, fold)

    back_args = dict(
        outline=bk.back_bodice,
        construction_lines=bk.construction_lines,
        dart_lines=bk.back_dart_lines,
        fill="white" if white_fill else settings.BACK_STYLE["fill"],
        stroke=settings.BACK_STYLE["stroke"],
        outline_labels=_labels(bk, settings.BACK_OUTLINE_LABELS),
        interior_labels=shared_interior,
        seam_allowance=seam_allowance,
        seam_allowance_fn=settings.back_seam_allowance_fn(bk, seam_allowance),
        label_offsets={**settings.INTERIOR_LABEL_OFFSETS,
                       **settings.BACK_LABEL_OFFSETS},
        curve_seam_segments=_curve_groups(bk.back_bodice),
        curve_seam_allowance=seam_allowance,
    )
    front_args = dict(
        outline=front_outline,
        construction_lines=bk.construction_lines,
        dart_lines=bk.front_dart_lines,
        fill="white" if white_fill else settings.FRONT_STYLE["fill"],
        stroke=settings.FRONT_STYLE["stroke"],
        outline_labels=front_labels,
        interior_labels=shared_interior,
        seam_allowance=seam_allowance,
        label_offsets={**settings.INTERIOR_LABEL_OFFSETS,
                       **settings.FRONT_LABEL_OFFSETS},
        curve_seam_segments=_curve_groups(front_outline),
        curve_seam_allowance=seam_allowance,
    )
    return front_args, back_args


def _sleeve_svg_args(sl, seam_allowance, white_fill=False):
    """Build the _write_svg keyword-argument dict for the sleeve piece."""
    return dict(
        outline=sl.outline,
        construction_lines=sl.construction_lines,
        dart_lines=sl.dart_lines,
        fill="white" if white_fill else settings.SLEEVE_STYLE["fill"],
        stroke=settings.SLEEVE_STYLE["stroke"],
        outline_labels=_labels(sl, settings.SLEEVE_OUTLINE_LABELS),
        interior_labels=_labels(sl, settings.SLEEVE_INTERIOR_LABELS),
        seam_allowance=seam_allowance,
        label_offsets={**settings.SLEEVE_INTERIOR_OFFSETS,
                       **settings.SLEEVE_LABEL_OFFSETS},
        curve_seam_segments=[sl.cap_segments],
        curve_seam_allowance=seam_allowance,
        unclipped_construction_lines=sl.unclipped_construction_lines,
        text_annotations=settings.sleeve_text_annotations(sl),
    )


# ── Render: SVG strings (web interface) ───────────────────────────────────────

def render_svgs(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
                fold=False, seam_allowance=0.75, deepen_bust_dart=False,
                white_fill=False):
    """Return {'front': svg_str, 'back': svg_str, ...}.  Used by the web interface."""
    bk = build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
               deepen_bust_dart=deepen_bust_dart)
    front_args, back_args = _bodice_svg_args(bk, fold, seam_allowance, white_fill)

    front_svg, front_w, front_h = _write_svg(None, front_args.pop("outline"), **front_args)
    back_svg,  back_w,  back_h  = _write_svg(None, back_args.pop("outline"),  **back_args)

    return {
        'front': front_svg, 'back': back_svg,
        'front_w': front_w, 'front_h': front_h,
        'back_w': back_w, 'back_h': back_h,
    }


def render_sleeve_svg(sigma, upsilon, omega, xi, psi,
                      seam_allowance=0.75, white_fill=False):
    """Return {'sleeve': svg_str, 'sleeve_w': …, 'sleeve_h': …}."""
    sl = sleeve.build(sigma, upsilon, omega, xi, psi)
    args = _sleeve_svg_args(sl, seam_allowance, white_fill)
    svg, w, h = _write_svg(None, args.pop("outline"), **args)
    return {'sleeve': svg, 'sleeve_w': w, 'sleeve_h': h}


def render_web(params):
    """Generic web-frontend entry point.

    Every pattern module exposes render_web(params): params is a dict keyed
    by the measurement/option keys declared in manifest.json (optional
    groups may be absent), plus the renderer options 'seam_allowance' and
    'white_fill'.  Returns {piece_id: svg_str, piece_id + '_w': inches, …}
    for each piece that could be drafted from the given measurements.
    """
    seam_allowance = float(params.get("seam_allowance", 0.75))
    white_fill = bool(params.get("white_fill", False))

    out = render_svgs(
        params["alpha"], params["beta"], params["gamma"], params["delta"],
        params["epsilon"], params["zeta"], params["eta"], params["theta"],
        fold=bool(params.get("fold", False)),
        seam_allowance=seam_allowance,
        deepen_bust_dart=bool(params.get("deepen_bust_dart", False)),
        white_fill=white_fill,
    )

    sleeve_keys = ("sigma", "upsilon", "omega", "xi", "psi")
    if all(params.get(k) is not None for k in sleeve_keys):
        out.update(render_sleeve_svg(
            *(float(params[k]) for k in sleeve_keys),
            seam_allowance=seam_allowance, white_fill=white_fill,
        ))
    return out


# ── Render: SVG files ─────────────────────────────────────────────────────────

def render(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
           prefix="bodice", fold=False, seam_allowance=0.75,
           deepen_bust_dart=False):
    """Render bodice blocks to SVG files.

    Args:
        fold: If True, mirror front bodice on the M-D line to show full width
        seam_allowance: Seam allowance in inches (default 0.75). The A→GG
                       (center back) seam receives max(seam_allowance, 1.0),
                       except when seam_allowance is exactly 0.
        deepen_bust_dart: If True, add 0.5" to the bust dart depth from Chart 1.
    """
    bk = build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
               deepen_bust_dart=deepen_bust_dart)
    front_args, back_args = _bodice_svg_args(bk, fold, seam_allowance)

    _write_svg(f"{prefix}_back.svg",  back_args.pop("outline"),  **back_args)
    _write_svg(f"{prefix}_front.svg", front_args.pop("outline"), **front_args)


def render_sleeve(sigma, upsilon, omega, xi, psi,
                  prefix="sleeve", seam_allowance=0.75):
    """Render sleeve block to an SVG file."""
    sl = sleeve.build(sigma, upsilon, omega, xi, psi)
    args = _sleeve_svg_args(sl, seam_allowance)
    _write_svg(f"{prefix}.svg", args.pop("outline"), **args)
