"""Front panel of the basic 2-dart skirt sloper.

See back_panel.py's module docstring for the shared modeling notes (dart
chart, dart-placement approximation, curve approximations). Only the
front-specific constants differ: dart leg length 3.5in (p.50), waist ease
1/4in, chart lookup uses the front columns.
"""

import numpy as np
from types import SimpleNamespace

from geometry import cubic_bezier

DART_LEG_LENGTH = 3.5   # in, p.50
EASE = 0.5              # in, hip-row ease
SIDE_SEAM_BULGE = 0.25  # in, gentle hip-curve blend


def curve_side_seam(A, B, t):
    """Mostly-straight cubic Bézier from waist (A) to hem (B) with a
    vertical tangent at both ends and a slight outward hip bulge."""
    height = A[1] - B[1]
    C1 = A + np.array([SIDE_SEAM_BULGE, -height / 3])
    C2 = B + np.array([SIDE_SEAM_BULGE * 0.5, height / 3])
    return cubic_bezier(A, C1, C2, B, t)


def build(hip_arc_front, hip_depth_front, skirt_length, n_darts, intake_each):
    """Compute the front panel's points and outline.  Returns a
    SimpleNamespace.  n_darts/intake_each come from dart_chart.lookup(),
    resolved once by the pattern's __init__.py from all four (front+back)
    arc measurements together, since the chart's dart count/intake is
    shared across both panels.

    Points run center front (x=0) to side seam (x=front_width) — the
    mirror image of back_panel.py, whose side seam sits at x=back_width
    with center back at x=0."""
    front_width = hip_arc_front + EASE
    H = np.array([0.0, 0.0])                    # waist, center front
    A = np.array([front_width, 0.0])             # waist, side seam
    I = np.array([0.0, -hip_depth_front])        # hip level, center front
    C = np.array([front_width, -hip_depth_front])   # hip level, side seam
    J = np.array([0.0, -skirt_length])           # hem, center front
    B = np.array([front_width, -skirt_length])   # hem, side seam

    dart_points = []   # list of (leg_in, point, leg_out) per dart, center->side order
    for i in range(n_darts):
        cx = front_width * (i + 1) / (n_darts + 1)
        leg_in = np.array([cx - intake_each / 2, 0.0])
        leg_out = np.array([cx + intake_each / 2, 0.0])
        point = np.array([cx, -DART_LEG_LENGTH])
        dart_points.append((leg_in, point, leg_out))

    outline = []
    prev = H
    for leg_in, point, leg_out in dart_points:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, A))
    outline.append(("cubic_curve", lambda t: curve_side_seam(A, B, t), A, B))
    outline.append(("line", B, J))
    outline.append(("line", J, H))

    return SimpleNamespace(
        n_darts=n_darts, intake_each=intake_each, front_width=front_width,
        H=H, A=A, I=I, C=C, J=J, B=B,
        dart_points=dart_points,
        outline=outline,
        construction_lines=[(I, C)],
        dart_lines=[],
    )
