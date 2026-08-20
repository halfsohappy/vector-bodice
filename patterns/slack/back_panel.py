"""Back panel of the slack foundation — a closer-fitting trim of the trouser.

Adapted from Pattern Making for Fashion Design, 5th ed., "Slack —
Foundation 3" (p.577): "Trace the front and back trouser, omitting darts
closest to the side seam. Modify the pattern using illustration and
measurements as a guide." Unlike Trouser/Jean, the book gives this step
as a traced-and-eyeballed adjustment (illustration + approximate inches),
not a numbered formula list — so the fixed offsets below are a direct
transcription of the figure's labelled inches, applied at trouser's own
named points, rather than a literal reproduction of hand-eyeballing a
traced copy.
"""

import numpy as np
from types import SimpleNamespace

from geometry import on_line
from patterns.trouser import back_panel as trouser

WAIST_TRIM = 0.5    # in, taken in in from the side waist
HIP_TRIM = 0.25     # in, outseam drawn in at hip level
CROTCH_TRIM = 0.75  # in, crotch extension shortened
HEM_HALF = 4.0       # in, half the (narrower) back leg opening


def build(hip_arc, waist_arc, crotch_depth, pant_length):
    """Compute the slack back panel's points and outline.  Returns a
    SimpleNamespace."""
    base = trouser.build(hip_arc, waist_arc, crotch_depth, pant_length)

    # Drop the dart closest to the side seam (last in the list).
    dart_points = base.dart_points[:-1]

    O = base.O - np.array([WAIST_TRIM, 0.0])
    C = base.C - np.array([HIP_TRIM, 0.0])
    I = base.I + np.array([CROTCH_TRIM, 0.0])
    V = np.array([base.D[0] - (0.5 * (base.D[0] - I[0]) + trouser.GRAINLINE_ADJ), base.G[1]])
    ankle_y = base.Y[1]
    Y = np.array([V[0] + HEM_HALF, ankle_y])
    Z = np.array([V[0] - HEM_HALF, ankle_y])
    knee_y = base.R[1]
    R = on_line(C, Y, y=knee_y)
    S = on_line(I, Z, y=knee_y)

    outline = [("line", base.H, base.N)]
    prev = base.N
    for leg_in, point, leg_out in dart_points:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, O))
    outline.append(("cubic_curve", lambda t: trouser.curve_hip(O, C, t), O, C))
    outline.append(("line", C, Y))
    outline.append(("line", Y, Z))
    outline.append(("cubic_curve", lambda t: trouser.curve_inseam(Z, I, t), Z, I))
    outline.append(("cubic_curve", lambda t: trouser.curve_crotch(base.X, I, 1 - t), I, base.X))
    outline.append(("line", base.X, base.H))

    return SimpleNamespace(
        back_width=base.back_width, hip_depth=base.hip_depth, knee_depth=base.knee_depth,
        H=base.H, N=base.N, O=O, G=base.G, C=C, D=base.D,
        X=base.X, I=I, V=V, Y=Y, Z=Z, R=R, S=S,
        n_darts=len(dart_points), intake_each=trouser.DART_INTAKE, dart_points=dart_points,
        dart_letters=base.dart_letters[:len(dart_points)],
        outline=outline,
        construction_lines=[(base.G, base.D)],
        dart_lines=[],
    )
