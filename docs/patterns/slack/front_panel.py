"""Front panel of the slack foundation — see back_panel.py for notes."""

import numpy as np
from types import SimpleNamespace

from geometry import on_line
from patterns.trouser import front_panel as trouser
from patterns.trouser import legline
from . import creaseline

CF_TRIM = 0.25      # in, taken in at centre front
WAIST_TRIM = 0.25   # in, taken in at the side waist
HIP_TRIM = 0.25     # in, outseam drawn in at hip level
CROTCH_TRIM = 0.25  # in, crotch extension shortened
HEM_HALF = 3.5        # in, half the (narrower) front leg opening


def build(hip_arc, waist_arc, crotch_depth, pant_length,
          flared_leg=False, flare_position="at_knee", back_crotch_width=None,
          creaseline_flare=False):
    """Compute the slack front panel's points and outline.  Returns a
    SimpleNamespace."""
    base = trouser.build(hip_arc, waist_arc, crotch_depth, pant_length)

    LL = base.LL + np.array([CF_TRIM, 0.0])
    Q = base.Q - np.array([WAIST_TRIM, 0.0])
    C = base.C - np.array([HIP_TRIM, 0.0])
    M = base.M + np.array([CROTCH_TRIM, 0.0])

    def waist_at(x):
        return float(on_line(LL, Q, x=x)[1])

    n_darts = base.n_darts - 1
    intake = trouser.DART_INTAKE
    dart_points = []
    for i in range(n_darts):
        cx = LL[0] + (Q[0] - LL[0]) * (i + 1) / (n_darts + 1)
        leg_in = np.array([cx - intake / 2, waist_at(cx - intake / 2)])
        leg_out = np.array([cx + intake / 2, waist_at(cx + intake / 2)])
        point = np.array([cx, waist_at(cx) - trouser.DART_DEPTH])
        dart_points.append((leg_in, point, leg_out))

    W = np.array([base.D[0] - (0.5 * (base.D[0] - M[0]) + trouser.GRAINLINE_ADJ),
                  base.K[1]])
    ankle_y = base.U[1]
    knee_y = base.UU[1]

    leg = None
    if flared_leg:
        if back_crotch_width is None:
            raise ValueError("flared_leg needs back_crotch_width")
        leg = legline.build(C, M, W[0], HEM_HALF, ankle_y, knee_y,
                            flare_half=back_crotch_width / 2.0 - legline.FRONT_REDUCTION,
                            flare_position=flare_position)
        U, V, UU, VV = leg.ankle_out, leg.ankle_in, leg.knee_out, leg.knee_in
    else:
        U = np.array([W[0] + HEM_HALF, ankle_y])
        V = np.array([W[0] - HEM_HALF, ankle_y])
        if creaseline_flare:
            # Figure 3: extend 1 1/2in at side and inseam, drop 1/2in at
            # each end, blend an inward curve at the hemline.
            U, V = creaseline.front_hem(SimpleNamespace(U=U, V=V))
        UU = on_line(C, U, y=knee_y)
        VV = on_line(M, V, y=knee_y)

    outline = []
    finished_waist = 0.0
    prev = LL
    for leg_in, point, leg_out in dart_points:
        outline.append(("line", prev, leg_in))
        finished_waist += float(np.linalg.norm(leg_in - prev))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, Q))
    finished_waist += float(np.linalg.norm(Q - prev))
    outline.append(("cubic_curve", lambda t: trouser.curve_hip(Q, C, t), Q, C))
    if leg is not None:
        outline.append(("line", C, leg.flare_out))
        outline.append(("line", leg.flare_out, U))
        outline.append(("line", U, V))
        outline.append(("line", V, leg.flare_in))
        outline.append(("cubic_curve", lambda t: trouser.curve_inseam(leg.flare_in, M, t),
                        leg.flare_in, M))
    else:
        outline.append(("line", C, U))
        outline.append(("line", U, V))
        outline.append(("cubic_curve", lambda t: trouser.curve_inseam(V, M, t), V, M))
    outline.append(("cubic_curve", lambda t: trouser.curve_crotch(base.X, M, 1 - t), M, base.X))
    outline.append(("line", base.X, LL))

    return SimpleNamespace(
        front_width=base.front_width, hip_depth=base.hip_depth, knee_depth=base.knee_depth,
        finished_waist=finished_waist,
        L=base.L, LL=LL, Q=Q, K=base.K, C=C, D=base.D,
        side_waist=Q, cf_waist=LL,
        X=base.X, M=M, W=W, U=U, V=V, UU=UU, VV=VV,
        n_darts=n_darts, intake_each=intake, dart_points=dart_points,
        dart_letters=base.dart_letters[:n_darts],
        outline=outline,
        construction_lines=[(base.K, base.D)],
        dart_lines=[],
        notches=[C, UU, VV],
    )
