"""Front panel of the jean foundation.

Adapted from Pattern Making for Fashion Design, 5th ed., "Jean —
Foundation 4" (p.578-582) — see back_panel.py for the shared rectangle
notes. The front crotch extension is a fixed 2in (not a fraction of the
crotch-level width like the back), and there's no relaxed-fit variant or
pitching on this panel — both are back-only per the book.

Dart placement: the book marks the front dart at a fixed, non-scaling
3 1/4in from the waist-ease-zone start (Q-S) — the same kind of gap
already hit in Trouser's darts. Here it's placed at the midpoint of the
ease zone instead, the same substitute already used for this panel's own
back panel (whose dart position happens to be exact in the book, at
that same midpoint) — keeping front and back consistent.
"""

import numpy as np
from types import SimpleNamespace

from geometry import on_line
from patterns.trouser.back_panel import curve_crotch
from patterns.trouser.front_panel import curve_hip, curve_inseam

HIP_EASE = 0.125            # in, "front hip plus 1/8 inch (ease)"
CENTER_FRONT_OFFSET = 0.5   # in, L-Q
WAIST_DART_EASE = 0.75      # in, Q-R = front waist arc plus 3/4"
DART_INTAKE = 0.5           # in, total (1/4" each side of the dart center)
DART_DEPTH = 2.5            # in, "square down 2 1/2 inches"
CROTCH_EXT_FIXED = 2.0      # in, K-M = 2 inches (fixed, not fractional)
GRAINLINE_MARK = 0.375      # in, D-Y = 3/8"
HEM_HALF = 3.5              # in, half the front leg opening
KNEE_EASE = 1.0


def build(hip_arc, waist_arc, crotch_depth, pant_length):
    """Compute the front panel's points and outline.  Returns a
    SimpleNamespace."""
    full_crotch_depth = crotch_depth
    hip_depth = full_crotch_depth / 3.0
    leg_span = pant_length - full_crotch_depth
    knee_depth = 0.5 * leg_span + KNEE_EASE

    waist_y = 0.0
    crotch_y = -full_crotch_depth
    hip_y = crotch_y + hip_depth
    ankle_y = -pant_length

    front_width = hip_arc + HIP_EASE

    L = np.array([0.0, waist_y])
    K = np.array([0.0, crotch_y])
    C = np.array([front_width, hip_y])
    D = np.array([front_width, crotch_y])
    X = np.array([0.0, 0.5 * crotch_y])   # crotch-curve tangent point: K-X = 1/2 of K-L
    M = np.array([-CROTCH_EXT_FIXED, crotch_y])   # crotch extension point

    Q = np.array([CENTER_FRONT_OFFSET, waist_y])
    R = np.array([CENTER_FRONT_OFFSET + waist_arc + WAIST_DART_EASE, waist_y])   # side waist
    dart_cx = 0.5 * (Q[0] + R[0])
    S = np.array([dart_cx, waist_y - DART_DEPTH])
    # dart legs aren't individually lettered in the book — next free letters
    N = np.array([dart_cx - DART_INTAKE / 2, waist_y])
    O = np.array([dart_cx + DART_INTAKE / 2, waist_y])
    dart_points = [(N, S, O)]

    Y = np.array([D[0] - GRAINLINE_MARK, crotch_y])   # D-Y = 3/8" mark
    Z = np.array([Y[0] - 0.5 * (Y[0] - M[0]), crotch_y])   # Y-Z = 1/2 of Y-M (grainline anchor)
    U = np.array([Z[0] + HEM_HALF, ankle_y])   # ankle, outseam side
    V = np.array([Z[0] - HEM_HALF, ankle_y])   # ankle, inseam side
    knee_y = ankle_y + knee_depth
    T = on_line(C, U, y=knee_y)   # knee, outseam side
    W = on_line(M, V, y=knee_y)   # knee, inseam side

    outline = [("line", L, Q)]
    prev = Q
    for leg_in_, point, leg_out_ in dart_points:
        outline.append(("line", prev, leg_in_))
        outline.append(("dart", leg_in_, point))
        outline.append(("dart", point, leg_out_))
        prev = leg_out_
    outline.append(("line", prev, R))
    outline.append(("cubic_curve", lambda t: curve_hip(R, C, t), R, C))
    outline.append(("line", C, U))
    outline.append(("line", U, V))
    outline.append(("cubic_curve", lambda t: curve_inseam(V, M, t), V, M))
    outline.append(("cubic_curve", lambda t: curve_crotch(X, M, 1 - t), M, X))
    outline.append(("line", X, L))

    return SimpleNamespace(
        front_width=front_width, hip_depth=hip_depth, knee_depth=knee_depth,
        L=L, Q=Q, R=R, K=K, C=C, D=D,
        X=X, M=M, Y=Y, Z=Z, U=U, V=V, T=T, W=W,
        n_darts=1, intake_each=DART_INTAKE, dart_points=dart_points,
        outline=outline,
        construction_lines=[(K, D)],
        dart_lines=[],
    )
