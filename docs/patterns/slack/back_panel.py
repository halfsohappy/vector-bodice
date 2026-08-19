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
    crotch_point = base.crotch_point + np.array([CROTCH_TRIM, 0.0])
    V_x = base.D[0] - (0.5 * (base.D[0] - crotch_point[0]) + trouser.GRAINLINE_ADJ)
    ankle_y = base.ankle_outseam[1]
    ankle_outseam = np.array([V_x + HEM_HALF, ankle_y])
    ankle_inseam = np.array([V_x - HEM_HALF, ankle_y])
    knee_y = base.knee_outseam[1]
    knee_outseam = on_line(C, ankle_outseam, y=knee_y)
    knee_inseam = on_line(crotch_point, ankle_inseam, y=knee_y)

    outline = [("line", base.H, base.N)]
    prev = base.N
    for leg_in, point, leg_out in dart_points:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, O))
    outline.append(("cubic_curve", lambda t: trouser.curve_hip(O, C, t), O, C))
    outline.append(("line", C, ankle_outseam))
    outline.append(("line", ankle_outseam, ankle_inseam))
    outline.append(("cubic_curve",
                    lambda t: trouser.curve_inseam(ankle_inseam, crotch_point, t),
                    ankle_inseam, crotch_point))
    outline.append(("cubic_curve",
                    lambda t: trouser.curve_crotch(base.crotch_top, crotch_point, 1 - t),
                    crotch_point, base.crotch_top))
    outline.append(("line", base.crotch_top, base.H))

    return SimpleNamespace(
        back_width=base.back_width, hip_depth=base.hip_depth, knee_depth=base.knee_depth,
        H=base.H, N=base.N, O=O, G=base.G, C=C, D=base.D,
        crotch_top=base.crotch_top, crotch_point=crotch_point,
        ankle_outseam=ankle_outseam, ankle_inseam=ankle_inseam,
        knee_outseam=knee_outseam, knee_inseam=knee_inseam,
        n_darts=len(dart_points), intake_each=trouser.DART_INTAKE, dart_points=dart_points,
        outline=outline,
        construction_lines=[(base.G, base.D)],
        dart_lines=[],
    )
