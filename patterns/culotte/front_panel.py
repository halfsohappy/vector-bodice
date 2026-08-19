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
          crotch_depth):
    """Compute the culotte front panel.  Returns a SimpleNamespace."""
    base = aline.build(hip_arc_front, hip_depth_front, skirt_length,
                       n_darts, intake_each)

    crotch_base = base.H + np.array([0.0, -(crotch_depth + CROTCH_EASE)])
    crotch_top = base.H + np.array(
        [0.0, -(0.5 * (crotch_depth + CROTCH_EASE) - 0.5)])
    crotch_out = crotch_base + np.array(
        [0.5 * hip_arc_front - CROTCH_OUT_EASE, 0.0])
    new_hem_center = base.J + np.array([crotch_out[0] - base.H[0], 0.0])

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
    outline.append(("line", base.B, new_hem_center))
    outline.append(("line", new_hem_center, crotch_out))
    outline.append(("cubic_curve",
                    lambda t: curve_crotch(crotch_top, crotch_out, 1 - t),
                    crotch_out, crotch_top))
    outline.append(("line", crotch_top, base.H))

    return SimpleNamespace(
        n_darts=base.n_darts, intake_each=intake_each, front_width=base.front_width,
        H=base.H, A=base.A, I=base.I, C=base.C, J=base.J, B=base.B,
        crotch_top=crotch_top, crotch_out=crotch_out, new_hem_center=new_hem_center,
        dart_points=base.dart_points,
        outline=outline,
        construction_lines=base.construction_lines,
        dart_lines=[],
    )
