"""Back panel of the jean foundation.

Adapted from Pattern Making for Fashion Design, 5th ed., "Jean —
Foundation 4" (p.578-582). A fresh draft with its own rectangle/dart/
crotch-curve constants (tighter than Trouser's), a single waist dart
(exact midpoint placement, no approximation gap this time), and two
jean-specific mechanics:

- Fit variant: the book gives the back crotch extension as G-I = 1/4 of
  hip width for a "contour fit," or that same value +1in for a "relaxed
  fit" — exposed here as relaxed_fit, the pattern's own manifest option.

- Centre-back pitch: the jean's shorter crotch extension is compensated
  by pitching the centre back both IN (H-N = 1 3/4in) and UP (N-T = 1in),
  so the finished waistline runs T->O and the centre back runs T->X. That
  1in lift is the book's built-in compensation. The separate "Pitching the
  Back Pattern" procedure on p.582 is an optional personal-fit correction
  for someone who still needs more crotch length — it is not part of the
  base draft, so it is not applied automatically here.

Dart placement is exact this time (the book gives it as literally the
midpoint of the waist ease-zone — no non-scaling worked-example gap to
approximate around, unlike Trouser's darts).
"""

import numpy as np
from types import SimpleNamespace

from geometry import on_line
from patterns.trouser.back_panel import curve_crotch, curve_hip, curve_inseam
from patterns.trouser import legline

HIP_EASE = 0.125            # in, "back hip plus 1/8 inch (ease)"
CENTER_BACK_OFFSET = 1.75   # in, H-N (centre back pitched in by this much)
CB_PITCH_UP = 1.0           # in, N-T squared up (book Fig 4)
WAIST_DART_EASE = 1.0       # in, N-O = back waist arc plus 1"
DART_INTAKE = 0.75          # in, total (3/8" each side of the dart center)
DART_DEPTH = 3.5            # in, "square down 3 1/2 inches"
CROTCH_EXT_FRAC = 0.25      # G-I = one-fourth of G-D, contour fit
RELAXED_EXTRA = 1.0         # in, added to the extension for a relaxed fit
GRAINLINE_MARK = 0.375      # in, D-V = 3/8"
HEM_HALF = 4.5              # in, half the back leg opening
KNEE_EASE = 1.0


def build(hip_arc, waist_arc, crotch_depth, pant_length, relaxed_fit=False,
          flared_leg=False, flare_position="at_knee"):
    """Compute the back panel's points and outline.  Returns a
    SimpleNamespace (with .pitch, the computed lift amount, for
    reference/testing)."""
    full_crotch_depth = crotch_depth   # no added ease, unlike Trouser
    hip_depth = full_crotch_depth / 3.0
    leg_span = pant_length - full_crotch_depth
    knee_depth = 0.5 * leg_span + KNEE_EASE

    crotch_y = -full_crotch_depth
    hip_y = crotch_y + hip_depth
    ankle_y = -pant_length

    back_width = hip_arc + HIP_EASE

    G = np.array([0.0, crotch_y])
    C = np.array([back_width, hip_y])
    D = np.array([back_width, crotch_y])
    X = np.array([0.0, 0.5 * crotch_y])   # crotch-curve tangent point: G-X = 1/2 of G-H

    ext_amount = CROTCH_EXT_FRAC * back_width + (RELAXED_EXTRA if relaxed_fit else 0.0)
    I = np.array([-ext_amount, crotch_y])   # crotch extension point

    waist_y = 0.0
    H = np.array([0.0, waist_y])                      # rectangle corner (construction only)
    N = np.array([CENTER_BACK_OFFSET, waist_y])       # H-N mark, centre back pitched in
    # T is the pitched centre-back waist point (book Fig 4: "N-T = 1 inch
    # squared up from N.  Draw line from T through X to crotch level").
    # The waistline runs T->O and the centre back runs T->X; the H-N-T
    # corner is construction only.  This 1in lift IS the book's built-in
    # compensation for the jean's short crotch extension — the separate
    # "Pitching the Back Pattern" adjustment on p.582 is an optional
    # personal-fit correction on top, not part of the base draft, so it is
    # deliberately not applied automatically here.
    T = N + np.array([0.0, CB_PITCH_UP])
    O = np.array([CENTER_BACK_OFFSET + waist_arc + WAIST_DART_EASE, waist_y])   # side waist

    def waist_at(x):
        """y on the sloping T->O waistline at the given x."""
        return float(on_line(T, O, x=x)[1])

    dart_cx = 0.5 * (T[0] + O[0])   # N-P = one-half of N-O
    P = np.array([dart_cx, waist_at(dart_cx) - DART_DEPTH])
    # dart legs aren't individually lettered in the book — next free letters
    J = np.array([dart_cx - DART_INTAKE / 2, waist_at(dart_cx - DART_INTAKE / 2)])
    K = np.array([dart_cx + DART_INTAKE / 2, waist_at(dart_cx + DART_INTAKE / 2)])
    dart_points = [(J, P, K)]

    V = np.array([D[0] - GRAINLINE_MARK, crotch_y])   # D-V = 3/8" mark
    W = np.array([V[0] - 0.5 * (V[0] - I[0]), crotch_y])   # V-W = 1/2 of V-I (grainline anchor)
    knee_y = ankle_y + knee_depth
    crotch_width = float(D[0] - I[0])

    leg = None
    if flared_leg:
        leg = legline.build(C, I, W[0], HEM_HALF, ankle_y, knee_y,
                            flare_half=crotch_width / 2.0,
                            flare_position=flare_position)
        Y, Z, YY, ZZ = leg.ankle_out, leg.ankle_in, leg.knee_out, leg.knee_in
    else:
        Y = np.array([W[0] + HEM_HALF, ankle_y])   # ankle, outseam side
        Z = np.array([W[0] - HEM_HALF, ankle_y])   # ankle, inseam side
        YY = on_line(C, Y, y=knee_y)   # knee, outseam side
        ZZ = on_line(I, Z, y=knee_y)   # knee, inseam side

    outline = []
    finished_waist = 0.0
    prev = T
    for leg_in_, point, leg_out_ in dart_points:
        outline.append(("line", prev, leg_in_))
        finished_waist += float(np.linalg.norm(leg_in_ - prev))
        outline.append(("dart", leg_in_, point))
        outline.append(("dart", point, leg_out_))
        prev = leg_out_
    outline.append(("line", prev, O))
    finished_waist += float(np.linalg.norm(O - prev))
    outline.append(("cubic_curve", lambda t: curve_hip(O, C, t), O, C))
    if leg is not None:
        outline.append(("line", C, leg.flare_out))
        outline.append(("line", leg.flare_out, Y))
        outline.append(("line", Y, Z))
        outline.append(("line", Z, leg.flare_in))
        outline.append(("cubic_curve", lambda t: curve_inseam(leg.flare_in, I, t), leg.flare_in, I))
    else:
        outline.append(("line", C, Y))
        outline.append(("line", Y, Z))
        outline.append(("cubic_curve", lambda t: curve_inseam(Z, I, t), Z, I))
    outline.append(("cubic_curve", lambda t: curve_crotch(X, I, 1 - t), I, X))
    outline.append(("line", X, T))

    return SimpleNamespace(
        back_width=back_width, hip_depth=hip_depth, knee_depth=knee_depth,
        pitch=CB_PITCH_UP, finished_waist=finished_waist, crotch_width=crotch_width,
        H=H, N=N, T=T, O=O, G=G, C=C, D=D,
        X=X, I=I, V=V, W=W, Y=Y, Z=Z, YY=YY, ZZ=ZZ,
        n_darts=1, intake_each=DART_INTAKE, dart_points=dart_points,
        outline=outline,
        construction_lines=[(G, D), (H, N)],
        dart_lines=[],
        notches=[C, YY, ZZ],
    )
