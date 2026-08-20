"""Back panel of the trouser foundation.

Adapted from Pattern Making for Fashion Design, 5th ed., "Trouser —
Foundation 2" (p.573-576). A fresh draft, not built on the skirt chain: a
rectangle from crotch depth/pant length, hip and crotch width from the
back hip arc, two fixed-intake darts, a crotch curve, and mostly-straight
outseam/inseam legs.

Dart placement: the book marks darts at fixed, non-scaling inches within
a "3 inch zone" measured from center back (N-P) — the same kind of
literal worked-example gap already hit in skirt_basic's dart placement.
As there, darts are instead spaced evenly across the available waist
ease-zone (center back to the side-waist point), a standard scalable
convention, rather than reproducing the book's fixed inches.

The waistline is drafted flat (no curve) for the same reason
skirt_basic's is: the book's own waistline here is described as only
"slightly" curved, well within the level of simplification already used
throughout this tool. The hip curve and inseam/outseam legs are smooth
mathematical approximations of the book's hand-drafting curve tool.
"""

import numpy as np
from types import SimpleNamespace

from geometry import cubic_bezier, on_line

CROTCH_EASE = 0.75          # in, "crotch depth plus 3/4-inch ease"
HIP_EASE = 0.25             # in, "back hip plus 1/4 inch (ease)"
KNEE_EASE = 1.0             # in, "one-half of B-D plus 1 inch"
CENTER_BACK_OFFSET = 0.75   # in, H-N
WAIST_DART_EASE = 2.25      # in, N-O = waist arc plus 2 1/4"
DART_COUNT = 2
DART_INTAKE = 1.0           # in, each
DART_DEPTH = 4.5            # in, "square down 4 1/2 inches"
CROTCH_EXT_FRAC = 0.5       # G-I = one-half of G-D
GRAINLINE_ADJ = 0.25        # in, D-V = one-half of D-I, plus 1/4"
HEM_HALF = 4.5              # in, half the back leg opening
HIP_CURVE_BULGE = 0.3       # in, gentle blend from the waist row to the hip row
INSEAM_BULGE = -0.2         # in, gentle inward blend near the crotch point


def curve_hip(waist_side, hip_side, t):
    """Short, mostly-straight cubic Bezier blending the side-waist point
    down into the hip-row side point (the book's "hip curve")."""
    height = waist_side[1] - hip_side[1]
    C1 = waist_side + np.array([HIP_CURVE_BULGE, -height / 3])
    C2 = hip_side + np.array([HIP_CURVE_BULGE * 0.5, height / 3])
    return cubic_bezier(waist_side, C1, C2, hip_side, t)


def curve_inseam(ankle_inseam, crotch_point, t):
    """Mostly-straight cubic Bezier from the ankle up to the crotch
    point, with a gentle inward blend near the top (the book's "inward
    curved lines... blending close to knee level")."""
    height = crotch_point[1] - ankle_inseam[1]
    C1 = ankle_inseam + np.array([INSEAM_BULGE * 0.5, height / 3])
    C2 = crotch_point + np.array([INSEAM_BULGE, -height / 3])
    return cubic_bezier(ankle_inseam, C1, C2, crotch_point, t)


def curve_crotch(P_start, P_end, t):
    """Cubic Bezier from the crotch-curve top (on the center-back line)
    to the crotch point: vertical tangent at the top, horizontal tangent
    at the crotch point — same kappa-constant technique used throughout
    this tool (see culotte.front_panel.curve_crotch)."""
    chord = P_end - P_start
    k = 0.5523 * float(np.linalg.norm(chord))
    C1 = P_start + np.array([0.0, -k])
    sign = 1.0 if P_end[0] >= P_start[0] else -1.0
    C2 = P_end - np.array([sign * k, 0.0])
    return cubic_bezier(P_start, C1, C2, P_end, t)


def build(hip_arc, waist_arc, crotch_depth, pant_length):
    """Compute the back panel's points and outline.  Returns a
    SimpleNamespace."""
    full_crotch_depth = crotch_depth + CROTCH_EASE
    hip_depth = full_crotch_depth / 3.0
    leg_span = pant_length - full_crotch_depth
    knee_depth = 0.5 * leg_span + KNEE_EASE

    waist_y = 0.0
    crotch_y = -full_crotch_depth
    hip_y = crotch_y + hip_depth
    ankle_y = -pant_length

    back_width = hip_arc + HIP_EASE

    H = np.array([0.0, waist_y])                     # center back, waist
    G = np.array([0.0, crotch_y])                     # center back, crotch
    C = np.array([back_width, hip_y])                 # side, hip
    D = np.array([back_width, crotch_y])               # side, crotch (spine)

    X = np.array([0.0, waist_y + 0.5 * (crotch_y - waist_y)])   # crotch-curve tangent point
    I = np.array([-CROTCH_EXT_FRAC * back_width, crotch_y])      # crotch extension point

    N = np.array([CENTER_BACK_OFFSET, waist_y])
    O = np.array([CENTER_BACK_OFFSET + waist_arc + WAIST_DART_EASE, waist_y])   # side waist

    # Dart legs/points aren't individually lettered in the book (p.574) —
    # continuing the alphabet from where this panel's own lettering leaves
    # off, first dart (nearer center) then second (nearer side).
    _dart_letters = [("J", "K", "L"), ("M", "P", "Q")]
    dart_points = []
    for i in range(DART_COUNT):
        cx = N[0] + (O[0] - N[0]) * (i + 1) / (DART_COUNT + 1)
        leg_in = np.array([cx - DART_INTAKE / 2, waist_y])
        leg_out = np.array([cx + DART_INTAKE / 2, waist_y])
        point = np.array([cx, waist_y - DART_DEPTH])
        dart_points.append((leg_in, point, leg_out))

    V = np.array([D[0] - (0.5 * (D[0] - I[0]) + GRAINLINE_ADJ), crotch_y])   # grainline anchor
    Y = np.array([V[0] + HEM_HALF, ankle_y])   # ankle, outseam side
    Z = np.array([V[0] - HEM_HALF, ankle_y])   # ankle, inseam side
    knee_y = ankle_y + knee_depth
    R = on_line(C, Y, y=knee_y)   # knee, outseam side
    S = on_line(I, Z, y=knee_y)   # knee, inseam side

    outline = [("line", H, N)]
    prev = N
    for i, (leg_in, point, leg_out) in enumerate(dart_points):
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, O))
    outline.append(("cubic_curve", lambda t: curve_hip(O, C, t), O, C))
    outline.append(("line", C, Y))
    outline.append(("line", Y, Z))
    outline.append(("cubic_curve", lambda t: curve_inseam(Z, I, t), Z, I))
    outline.append(("cubic_curve", lambda t: curve_crotch(X, I, 1 - t), I, X))
    outline.append(("line", X, H))

    return SimpleNamespace(
        back_width=back_width, hip_depth=hip_depth, knee_depth=knee_depth,
        H=H, N=N, O=O, G=G, C=C, D=D,
        X=X, I=I, V=V, Y=Y, Z=Z, R=R, S=S,
        n_darts=DART_COUNT, intake_each=DART_INTAKE, dart_points=dart_points,
        dart_letters=_dart_letters[:DART_COUNT],
        outline=outline,
        construction_lines=[(G, D)],
        dart_lines=[],
    )
