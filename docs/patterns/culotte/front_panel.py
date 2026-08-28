"""Front panel of the culotte — a crotch curve cut into the A-line skirt.

Adapted from Pattern Making for Fashion Design, 5th ed., "Culotte —
Foundation 1" (p.571-572), built on patterns/skirt_aline. The book's own
formulas for the crotch curve are self-contained (functions of crotch
depth and hip arc, not of the A-line's internal geometry beyond its
waist/center-front axis), so they're used close to verbatim; the one
approximation is the curve's exact shape, drawn tangent-vertical at the
top (blending into the untouched center-front line above it) and
tangent-horizontal at the crotch point (blending into the inseam below
it) — the same kappa-constant technique already used for the bodice's
armhole/neckline curves, standing in for the book's "touching C and b"
physical-curve-rule description.

Only the center-front side of the A-line panel changes: from the crotch
point down, a new straight inseam replaces the original center-front
edge, carrying the same x-offset down to the (already-flared) hemline.
The waist, any remaining dart, and the side seam are untouched.
"""

import numpy as np
from types import SimpleNamespace

from geometry import cubic_bezier
from patterns.skirt_aline import front_panel as aline

CROTCH_EASE = 0.75    # in, book's "crotch depth plus 3/4"
CROTCH_OUT_EASE = 0.75  # in, book's "less 3/4 inch" on the hip-based crotch width

# Box-Pleated Culotte (book p.585-586)
BOX_PLEAT_SHIFT = 5.0     # in, "shift pattern 5 inches along the guideline for pleat"
BOX_PLEAT_TRACE = 2.5     # in, "trace ... and 2 1/2 inches of waistline" (A to B)
BOX_PLEAT_GUIDE = 5.0     # in, "draw a 5-inch line down from B parallel with CF"
BOX_PLEAT_FLARE = 1.5     # in, optional extra flare swung out at the CF hem


def _shift_cf(ns, dx):
    """Return a copy of the traced A-line namespace with its centre-front
    edge slid outward by dx (the box-pleat insertion).  The side seam,
    hem-at-side and darts are untouched — only the CF column moves."""
    from types import SimpleNamespace as _NS
    out = _NS(**vars(ns))
    shift = np.array([dx, 0.0])
    out.H = ns.H + shift
    out.I = ns.I + shift
    out.J = ns.J + shift
    return out


def curve_crotch(P_start, P_end, t):
    """Cubic Bézier from the crotch-curve top (on the center front line)
    to the crotch point: vertical tangent at the top, horizontal tangent
    at the crotch point — same kappa-constant technique as
    front_bodice.curve_neck."""
    chord = P_end - P_start
    k = 0.5523 * float(np.linalg.norm(chord))
    C1 = P_start + np.array([0.0, -k])
    sign = 1.0 if P_end[0] >= P_start[0] else -1.0
    C2 = P_end - np.array([sign * k, 0.0])
    return cubic_bezier(P_start, C1, C2, P_end, t)


def build(hip_arc_front, hip_depth_front, skirt_length, n_darts, intake_each,
          crotch_depth, box_pleat=False, box_pleat_flare=False):
    """Compute the culotte front panel.  Returns a SimpleNamespace.

    box_pleat adds the book's centre-front box pleat (p.585-586): the
    pattern is shifted 5in along a guideline, which inserts that much
    width at the centre front.  The pleat folds that fullness back out
    before the waistband goes on, so the finished waist is unchanged."""
    base = aline.build(hip_arc_front, hip_depth_front, skirt_length,
                       n_darts, intake_each)

    crotch_base = base.H + np.array([0.0, -(crotch_depth + CROTCH_EASE)])
    X = base.H + np.array(
        [0.0, -(0.5 * (crotch_depth + CROTCH_EASE) - 0.5)])   # crotch-curve tangent point
    D = crotch_base + np.array(
        [0.5 * hip_arc_front - CROTCH_OUT_EASE, 0.0])          # crotch extension point
    E = base.J + np.array([D[0] - base.H[0], 0.0])             # new hem corner, under D

    # ── Box pleat (book p.585-586) ────────────────────────────────────────
    # "Shift pattern 5 inches along the guideline for pleat", then "draw
    # parallel lines from A and C to hem".  Modelled as a straight
    # insertion of BOX_PLEAT_SHIFT at the centre front: H/X/D/E and the
    # crotch side all move outward, the side seam and darts stay put.
    pleats = []
    pleat_intake = 0.0
    if box_pleat:
        # The insertion moves the centre front AWAY from the side seam, so
        # the panel gets BOX_PLEAT_SHIFT wider; the pleat folds that back out.
        shift = np.array([-BOX_PLEAT_SHIFT, 0.0])
        A_pleat = base.H.copy()                      # book's A: original CF waist
        base = _shift_cf(base, -BOX_PLEAT_SHIFT)
        X = X + shift
        D = D + shift
        E = E + shift
        if box_pleat_flare:
            # "swing the pattern 1 1/2 inches away from the centre front at
            # the hem and blend the hemline"
            E = E + np.array([-BOX_PLEAT_FLARE, 0.0])
        # the two fold lines: the new centre front and the original one
        pleats.append((base.H.copy(), A_pleat))
        pleat_intake = BOX_PLEAT_SHIFT

    outline = []
    prev = base.H
    for leg_in, point, leg_out in base.dart_points:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, base.A))
    # reuse the A-line side-seam curve as-is (waist to widened hem)
    _, side_func, _, _ = base.outline[-3]   # the cubic_curve entry, before the 2 hem/inseam lines
    outline.append(("cubic_curve", side_func, base.A, base.B))
    outline.append(("line", base.B, E))
    outline.append(("line", E, D))
    outline.append(("cubic_curve",
                    lambda t: curve_crotch(X, D, 1 - t),
                    D, X))
    outline.append(("line", X, base.H))

    finished_waist_span = float(base.A[0] - base.H[0]) - pleat_intake \
        - sum(float(lo[0] - li[0]) for li, _p, lo in base.dart_points)

    return SimpleNamespace(
        n_darts=base.n_darts, intake_each=intake_each, front_width=base.front_width,
        n_pleats=len(pleats), pleats=pleats, pleat_intake=pleat_intake,
        finished_waist_span=finished_waist_span,
        H=base.H, A=base.A, I=base.I, C=base.C, J=base.J, B=base.B,
        X=X, D=D, E=E,
        dart_points=base.dart_points,
        outline=outline,
        construction_lines=base.construction_lines,
        dart_lines=[(a, np.array([a[0], a[1] - BOX_PLEAT_GUIDE])) for a, _ in pleats]
                   + [(b, np.array([b[0], b[1] - BOX_PLEAT_GUIDE])) for _, b in pleats],
        notches=[a for a, _ in pleats] + [b for _, b in pleats],
    )
