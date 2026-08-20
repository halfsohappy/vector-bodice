"""Front panel of the trouser foundation.

Adapted from Pattern Making for Fashion Design, 5th ed., "Trouser —
Foundation 2" (p.573-576) — see back_panel.py for the shared rectangle
formulas and the dart-placement/waistline-curve approximation notes,
which apply identically here.
"""

import numpy as np
from types import SimpleNamespace

from geometry import cubic_bezier, on_line
from .back_panel import curve_crotch

CROTCH_EASE = 0.75          # in, "crotch depth plus 3/4-inch ease"
HIP_EASE = 0.25             # in, "front hip plus 1/4 inch (ease)"
KNEE_EASE = 1.0             # in, "one-half of B-D plus 1 inch"
WAIST_DART_EASE = 1.25      # in, L-Q = waist arc plus 1 1/4"
DART_COUNT = 2
DART_INTAKE = 0.5           # in, each
DART_DEPTH = 3.0            # in, "square down 3 inches"
CROTCH_EXT_FRAC = 0.25      # K-M = one-fourth of K-D
GRAINLINE_ADJ = 0.25        # in, D-W = one-half of D-M, plus 1/4"
HEM_HALF = 4.0              # in, half the front leg opening
HIP_CURVE_BULGE = 0.3
INSEAM_BULGE = -0.2


def curve_hip(waist_side, hip_side, t):
    """Short, mostly-straight cubic Bezier blending the side-waist point
    down into the hip-row side point (the book's "hip curve")."""
    height = waist_side[1] - hip_side[1]
    C1 = waist_side + np.array([HIP_CURVE_BULGE, -height / 3])
    C2 = hip_side + np.array([HIP_CURVE_BULGE * 0.5, height / 3])
    return cubic_bezier(waist_side, C1, C2, hip_side, t)


def curve_inseam(ankle_inseam, crotch_point, t):
    """Mostly-straight cubic Bezier from the ankle up to the crotch
    point, with a gentle inward blend near the top."""
    height = crotch_point[1] - ankle_inseam[1]
    C1 = ankle_inseam + np.array([INSEAM_BULGE * 0.5, height / 3])
    C2 = crotch_point + np.array([INSEAM_BULGE, -height / 3])
    return cubic_bezier(ankle_inseam, C1, C2, crotch_point, t)


def build(hip_arc, waist_arc, crotch_depth, pant_length):
    """Compute the front panel's points and outline.  Returns a
    SimpleNamespace."""
    full_crotch_depth = crotch_depth + CROTCH_EASE
    hip_depth = full_crotch_depth / 3.0
    leg_span = pant_length - full_crotch_depth
    knee_depth = 0.5 * leg_span + KNEE_EASE

    waist_y = 0.0
    crotch_y = -full_crotch_depth
    hip_y = crotch_y + hip_depth
    ankle_y = -pant_length

    front_width = hip_arc + HIP_EASE

    L = np.array([0.0, waist_y])                      # center front, waist
    K = np.array([0.0, crotch_y])                      # center front, crotch
    C = np.array([front_width, hip_y])                 # side, hip
    D = np.array([front_width, crotch_y])                # side, crotch (spine)

    X = np.array([0.0, waist_y + 0.5 * (crotch_y - waist_y)])    # crotch-curve tangent point
    M = np.array([-CROTCH_EXT_FRAC * front_width, crotch_y])      # crotch extension point

    Q = np.array([waist_arc + WAIST_DART_EASE, waist_y])   # side waist

    # Dart legs/points aren't individually lettered in the book (p.574) —
    # continuing the alphabet from where this panel's own lettering leaves
    # off, first dart (nearer center) then second (nearer side).
    _dart_letters = [("N", "O", "P"), ("R", "S", "T")]
    dart_points = []
    for i in range(DART_COUNT):
        cx = L[0] + (Q[0] - L[0]) * (i + 1) / (DART_COUNT + 1)
        leg_in = np.array([cx - DART_INTAKE / 2, waist_y])
        leg_out = np.array([cx + DART_INTAKE / 2, waist_y])
        point = np.array([cx, waist_y - DART_DEPTH])
        dart_points.append((leg_in, point, leg_out))

    W = np.array([D[0] - (0.5 * (D[0] - M[0]) + GRAINLINE_ADJ), crotch_y])   # grainline anchor
    U = np.array([W[0] + HEM_HALF, ankle_y])   # ankle, outseam side
    V = np.array([W[0] - HEM_HALF, ankle_y])   # ankle, inseam side
    knee_y = ankle_y + knee_depth
    Y = on_line(C, U, y=knee_y)   # knee, outseam side
    Z = on_line(M, V, y=knee_y)   # knee, inseam side

    outline = []
    prev = L
    for leg_in, point, leg_out in dart_points:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, Q))
    outline.append(("cubic_curve", lambda t: curve_hip(Q, C, t), Q, C))
    outline.append(("line", C, U))
    outline.append(("line", U, V))
    outline.append(("cubic_curve", lambda t: curve_inseam(V, M, t), V, M))
    outline.append(("cubic_curve", lambda t: curve_crotch(X, M, 1 - t), M, X))
    outline.append(("line", X, L))

    return SimpleNamespace(
        front_width=front_width, hip_depth=hip_depth, knee_depth=knee_depth,
        L=L, Q=Q, K=K, C=C, D=D,
        X=X, M=M, W=W, U=U, V=V, Y=Y, Z=Z,
        n_darts=DART_COUNT, intake_each=DART_INTAKE, dart_points=dart_points,
        dart_letters=_dart_letters[:DART_COUNT],
        outline=outline,
        construction_lines=[(K, D)],
        dart_lines=[],
    )
