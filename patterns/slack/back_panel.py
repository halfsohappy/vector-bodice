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

Dropping the side-most dart returns its intake to the waist, so the book
trims the same amount back off at BOTH ends of the waistline (1/2 inch at
centre back and 1/2 inch at the side) — which is what keeps the slack's
finished waist equal to the trouser's rather than a dart-width larger.
"""

import numpy as np
from types import SimpleNamespace

from geometry import on_line
from patterns.trouser import back_panel as trouser
from patterns.trouser import legline

CB_TRIM = 0.5       # in, taken in at centre back
WAIST_TRIM = 0.5    # in, taken in at the side waist
HIP_TRIM = 0.25     # in, outseam drawn in at hip level
CROTCH_TRIM = 0.75  # in, crotch extension shortened
HEM_HALF = 4.0       # in, half the (narrower) back leg opening


def build(hip_arc, waist_arc, crotch_depth, pant_length,
          flared_leg=False, flare_position="at_knee"):
    """Compute the slack back panel's points and outline.  Returns a
    SimpleNamespace."""
    base = trouser.build(hip_arc, waist_arc, crotch_depth, pant_length)

    S = base.S + np.array([CB_TRIM, 0.0])
    O = base.O - np.array([WAIST_TRIM, 0.0])
    C = base.C - np.array([HIP_TRIM, 0.0])
    I = base.I + np.array([CROTCH_TRIM, 0.0])

    def waist_at(x):
        return float(on_line(S, O, x=x)[1])

    # One dart survives (the side-most is dropped), re-centred on the
    # trimmed waistline.
    n_darts = base.n_darts - 1
    intake = trouser.DART_INTAKE
    dart_points = []
    for i in range(n_darts):
        cx = S[0] + (O[0] - S[0]) * (i + 1) / (n_darts + 1)
        leg_in = np.array([cx - intake / 2, waist_at(cx - intake / 2)])
        leg_out = np.array([cx + intake / 2, waist_at(cx + intake / 2)])
        point = np.array([cx, waist_at(cx) - trouser.DART_DEPTH])
        dart_points.append((leg_in, point, leg_out))

    V = np.array([base.D[0] - (0.5 * (base.D[0] - I[0]) + trouser.GRAINLINE_ADJ),
                  base.G[1]])
    ankle_y = base.Y[1]
    knee_y = base.YY[1]
    crotch_width = float(base.D[0] - I[0])

    leg = None
    if flared_leg:
        leg = legline.build(C, I, V[0], HEM_HALF, ankle_y, knee_y,
                            flare_half=crotch_width / 2.0,
                            flare_position=flare_position)
        Y, Z, YY, ZZ = leg.ankle_out, leg.ankle_in, leg.knee_out, leg.knee_in
    else:
        Y = np.array([V[0] + HEM_HALF, ankle_y])
        Z = np.array([V[0] - HEM_HALF, ankle_y])
        YY = on_line(C, Y, y=knee_y)
        ZZ = on_line(I, Z, y=knee_y)

    outline = []
    finished_waist = 0.0
    prev = S
    for leg_in, point, leg_out in dart_points:
        outline.append(("line", prev, leg_in))
        finished_waist += float(np.linalg.norm(leg_in - prev))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, O))
    finished_waist += float(np.linalg.norm(O - prev))
    outline.append(("cubic_curve", lambda t: trouser.curve_hip(O, C, t), O, C))
    if leg is not None:
        outline.append(("line", C, leg.flare_out))
        outline.append(("line", leg.flare_out, Y))
        outline.append(("line", Y, Z))
        outline.append(("line", Z, leg.flare_in))
        outline.append(("cubic_curve", lambda t: trouser.curve_inseam(leg.flare_in, I, t),
                        leg.flare_in, I))
    else:
        outline.append(("line", C, Y))
        outline.append(("line", Y, Z))
        outline.append(("cubic_curve", lambda t: trouser.curve_inseam(Z, I, t), Z, I))
    outline.append(("cubic_curve", lambda t: trouser.curve_crotch(base.X, I, 1 - t), I, base.X))
    outline.append(("line", base.X, S))

    return SimpleNamespace(
        back_width=base.back_width, hip_depth=base.hip_depth, knee_depth=base.knee_depth,
        finished_waist=finished_waist, crotch_width=crotch_width,
        H=base.H, N=base.N, S=S, O=O, G=base.G, C=C, D=base.D,
        X=base.X, I=I, V=V, Y=Y, Z=Z, YY=YY, ZZ=ZZ,
        n_darts=n_darts, intake_each=intake, dart_points=dart_points,
        dart_letters=base.dart_letters[:n_darts],
        outline=outline,
        construction_lines=[(base.G, base.D)],
        dart_lines=[],
        notches=[C, YY, ZZ],
    )
