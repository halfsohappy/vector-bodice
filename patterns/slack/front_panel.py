"""Front panel of the slack foundation — see back_panel.py for notes."""

import numpy as np
from types import SimpleNamespace

from geometry import on_line
from patterns.trouser import front_panel as trouser

WAIST_TRIM = 0.25   # in, taken in from the side waist
HIP_TRIM = 0.25     # in, outseam drawn in at hip level
CROTCH_TRIM = 0.25  # in, crotch extension shortened
HEM_HALF = 3.5        # in, half the (narrower) front leg opening


def build(hip_arc, waist_arc, crotch_depth, pant_length):
    """Compute the slack front panel's points and outline.  Returns a
    SimpleNamespace."""
    base = trouser.build(hip_arc, waist_arc, crotch_depth, pant_length)

    dart_points = base.dart_points[:-1]

    Q = base.Q - np.array([WAIST_TRIM, 0.0])
    C = base.C - np.array([HIP_TRIM, 0.0])
    M = base.M + np.array([CROTCH_TRIM, 0.0])
    W = np.array([base.D[0] - (0.5 * (base.D[0] - M[0]) + trouser.GRAINLINE_ADJ), base.K[1]])
    ankle_y = base.U[1]
    U = np.array([W[0] + HEM_HALF, ankle_y])
    V = np.array([W[0] - HEM_HALF, ankle_y])
    knee_y = base.Y[1]
    Y = on_line(C, U, y=knee_y)
    Z = on_line(M, V, y=knee_y)

    outline = []
    prev = base.L
    for leg_in, point, leg_out in dart_points:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, Q))
    outline.append(("cubic_curve", lambda t: trouser.curve_hip(Q, C, t), Q, C))
    outline.append(("line", C, U))
    outline.append(("line", U, V))
    outline.append(("cubic_curve", lambda t: trouser.curve_inseam(V, M, t), V, M))
    outline.append(("cubic_curve", lambda t: trouser.curve_crotch(base.X, M, 1 - t), M, base.X))
    outline.append(("line", base.X, base.L))

    return SimpleNamespace(
        front_width=base.front_width, hip_depth=base.hip_depth, knee_depth=base.knee_depth,
        L=base.L, Q=Q, K=base.K, C=C, D=base.D,
        X=base.X, M=M, W=W, U=U, V=V, Y=Y, Z=Z,
        n_darts=len(dart_points), intake_each=trouser.DART_INTAKE, dart_points=dart_points,
        dart_letters=base.dart_letters[:len(dart_points)],
        outline=outline,
        construction_lines=[(base.K, base.D)],
        dart_lines=[],
    )
