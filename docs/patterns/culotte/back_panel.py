"""Back panel of the culotte — a crotch curve cut into the A-line skirt.

See front_panel.py's module docstring for the shared modeling notes. The
back crotch point sits further out than the front's (+0.75in vs -0.75in
on the hip-based offset) — the book's usual "back needs more room" rule —
and its curve is drawn from a proportionally longer diagonal ease (1.75in
vs the front's 1.5in book reference), both giving the back a visibly
deeper/rounder crotch curve than the front, matching the book's Figure 2.
"""

import numpy as np
from types import SimpleNamespace

from patterns.skirt_aline import back_panel as aline
from .front_panel import curve_crotch, CROTCH_EASE

CROTCH_OUT_EASE = 0.75   # in, book's "plus 3/4 inch" on the hip-based crotch width


def build(hip_arc_back, hip_depth_back, skirt_length, n_darts, intake_each,
          crotch_depth):
    """Compute the culotte back panel.  Returns a SimpleNamespace."""
    base = aline.build(hip_arc_back, hip_depth_back, skirt_length,
                       n_darts, intake_each)

    crotch_base = base.D + np.array([0.0, -(crotch_depth + CROTCH_EASE)])
    X = base.D + np.array(
        [0.0, -(0.5 * (crotch_depth + CROTCH_EASE) - 0.5)])   # crotch-curve tangent point
    H = crotch_base + np.array(
        [0.5 * hip_arc_back + CROTCH_OUT_EASE, 0.0])           # crotch extension point
    I = base.F + np.array([H[0] - base.D[0], 0.0])             # new hem corner, under H

    outline = []
    prev = base.D
    for leg_in, point, leg_out in base.dart_points:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, base.A))
    _, side_func, _, _ = base.outline[-3]   # A-line side-seam curve (waist to hem)
    outline.append(("cubic_curve", side_func, base.A, base.B))
    outline.append(("line", base.B, I))
    outline.append(("line", I, H))
    outline.append(("cubic_curve",
                    lambda t: curve_crotch(X, H, 1 - t),
                    H, X))
    outline.append(("line", X, base.D))

    return SimpleNamespace(
        n_darts=base.n_darts, intake_each=intake_each, back_width=base.back_width,
        D=base.D, A=base.A, G=base.G, C=base.C, F=base.F, B=base.B,
        X=X, H=H, I=I,
        dart_points=base.dart_points,
        outline=outline,
        construction_lines=base.construction_lines,
        dart_lines=[],
    )
