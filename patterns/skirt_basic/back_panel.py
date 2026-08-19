"""Back panel of the basic 2-dart skirt sloper.

Adapted from Pattern Making for Fashion Design, 5th ed., p.48-50 ("Skirt
Draft"). The book draws front and back side by side sharing one rectangle;
here each panel is self-contained, using its own hip arc for the full
panel width (waist row through hem) and its own hip depth for the hip row
— front and back may therefore differ by a small amount at the "shared"
side seam, which is normal and gets trued/blended when the pieces are
actually sewn together, same as the book's own "walk the seams, true"
instructions elsewhere.

Dart count/intake come from the book's Personal Dart Intake Chart
(dart_chart.py), keyed on the total hip-waist difference. Dart LEG length
is fixed at 5.5in for the back (p.50). Dart PLACEMENT is not in the
excerpted chart (the book references a numbered measurement, "dart
placement (20)", whose source formula wasn't located) — darts are placed
by evenly dividing the waist row between center back and the side seam,
a standard, size-scalable patternmaking convention used in place of the
book's literal (non-scaling) worked-example inches.

The waistline is drafted as straight segments between darts (the book's
physical "curve rule" tool isn't reproduced) and the side seam as a
gently blended, mostly-straight curve — both are visually reasonable
approximations of a hand-drafted curve, not literal tool replication.
"""

import numpy as np
from types import SimpleNamespace

from geometry import cubic_bezier

DART_LEG_LENGTH = 5.5   # in, p.50
EASE = 0.5              # in, hip-row ease
SIDE_SEAM_BULGE = 0.25  # in, gentle hip-curve blend


def curve_side_seam(A, B, t):
    """Mostly-straight cubic Bézier from waist (A) to hem (B) with a
    vertical tangent at both ends and a slight outward hip bulge."""
    height = A[1] - B[1]
    C1 = A + np.array([SIDE_SEAM_BULGE, -height / 3])
    C2 = B + np.array([SIDE_SEAM_BULGE * 0.5, height / 3])
    return cubic_bezier(A, C1, C2, B, t)


def build(hip_arc_back, hip_depth_back, skirt_length, n_darts, intake_each):
    """Compute the back panel's points and outline.  Returns a
    SimpleNamespace.  n_darts/intake_each come from dart_chart.lookup(),
    resolved once by the pattern's __init__.py from all four (front+back)
    arc measurements together, since the chart's dart count/intake is
    shared across both panels."""
    back_width = hip_arc_back + EASE
    D = np.array([0.0, 0.0])
    A = np.array([back_width, 0.0])
    G = np.array([0.0, -hip_depth_back])
    C = np.array([back_width, -hip_depth_back])
    F = np.array([0.0, -skirt_length])
    B = np.array([back_width, -skirt_length])

    dart_points = []   # list of (leg_in, point, leg_out) per dart, waist->side order
    for i in range(n_darts):
        cx = back_width * (i + 1) / (n_darts + 1)
        leg_in = np.array([cx - intake_each / 2, 0.0])
        leg_out = np.array([cx + intake_each / 2, 0.0])
        point = np.array([cx, -DART_LEG_LENGTH])
        dart_points.append((leg_in, point, leg_out))

    outline = []
    prev = D
    for leg_in, point, leg_out in dart_points:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, A))
    outline.append(("cubic_curve", lambda t: curve_side_seam(A, B, t), A, B))
    outline.append(("line", B, F))
    outline.append(("line", F, D))

    return SimpleNamespace(
        n_darts=n_darts, intake_each=intake_each, back_width=back_width,
        D=D, A=A, G=G, C=C, F=F, B=B,
        dart_points=dart_points,
        outline=outline,
        construction_lines=[(G, C)],
        dart_lines=[],
    )
