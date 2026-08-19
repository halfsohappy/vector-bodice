"""Back panel of the jean foundation.

Adapted from Pattern Making for Fashion Design, 5th ed., "Jean —
Foundation 4" (p.578-582). A fresh draft with its own rectangle/dart/
crotch-curve constants (tighter than Trouser's), a single waist dart
(exact midpoint placement, no approximation gap this time), and two
jean-specific mechanics:

- Fit variant: the book gives the back crotch extension as G-I = 1/4 of
  hip width for a "contour fit," or that same value +1in for a "relaxed
  fit" — exposed here as relaxed_fit, the pattern's own manifest option.

- Back pitching (p.582): the jean's shorter back crotch extension loses
  crotch length compared to a normal (Trouser-style) extension, so the
  book lifts the center-back point above the waist row to recover it.
  Modeled here in closed form: build a reference crotch curve using
  Trouser's own (longer) back extension formula at this same hip
  arc/crotch depth, compare its arc length to this draft's actual
  (shorter) crotch curve, and lift the whole waist row — center back,
  the small offset point, the side-waist point, and the dart — by the
  difference. This is the first piece in this tool whose outline extends
  above its own waist row.

Dart placement is exact this time (the book gives it as literally the
midpoint of the waist ease-zone — no non-scaling worked-example gap to
approximate around, unlike Trouser's darts).
"""

import numpy as np
from types import SimpleNamespace

from geometry import on_line, curve_length
from patterns.trouser import back_panel as trouser
from patterns.trouser.back_panel import curve_crotch, curve_hip, curve_inseam

HIP_EASE = 0.125            # in, "back hip plus 1/8 inch (ease)"
CENTER_BACK_OFFSET = 1.75   # in, H-N
WAIST_DART_EASE = 1.0       # in, N-O = back waist arc plus 1"
DART_INTAKE = 0.75          # in, total (3/8" each side of the dart center)
DART_DEPTH = 3.5            # in, "square down 3 1/2 inches"
CROTCH_EXT_FRAC = 0.25      # G-I = one-fourth of G-D, contour fit
RELAXED_EXTRA = 1.0         # in, added to the extension for a relaxed fit
GRAINLINE_MARK = 0.375      # in, D-V = 3/8"
HEM_HALF = 4.5              # in, half the back leg opening
KNEE_EASE = 1.0


def build(hip_arc, waist_arc, crotch_depth, pant_length, relaxed_fit=False):
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
    crotch_top = np.array([0.0, 0.5 * crotch_y])   # X: G-X = 1/2 of G-H

    ext_amount = CROTCH_EXT_FRAC * back_width + (RELAXED_EXTRA if relaxed_fit else 0.0)
    crotch_point = np.array([-ext_amount, crotch_y])   # I

    # Reference (Trouser-style) back crotch curve at this same hip
    # arc/crotch depth, to measure how much crotch length this jean's
    # shorter extension loses.
    ref_ext = trouser.CROTCH_EXT_FRAC * back_width
    ref_crotch_point = np.array([-ref_ext, crotch_y])
    ref_length = curve_length(lambda t: curve_crotch(crotch_top, ref_crotch_point, 1 - t))
    actual_length = curve_length(lambda t: curve_crotch(crotch_top, crotch_point, 1 - t))
    pitch = max(0.0, ref_length - actual_length)

    waist_y = pitch   # the whole waist row is lifted by the pitch amount
    H = np.array([0.0, waist_y])
    N = np.array([CENTER_BACK_OFFSET, waist_y])
    O = np.array([CENTER_BACK_OFFSET + waist_arc + WAIST_DART_EASE, waist_y])   # side waist
    dart_cx = 0.5 * (N[0] + O[0])   # N-P = one-half of N-O
    dart_point = np.array([dart_cx, waist_y - DART_DEPTH])
    leg_in = np.array([dart_cx - DART_INTAKE / 2, waist_y])
    leg_out = np.array([dart_cx + DART_INTAKE / 2, waist_y])
    dart_points = [(leg_in, dart_point, leg_out)]

    V = np.array([D[0] - GRAINLINE_MARK, crotch_y])
    W_x = V[0] - 0.5 * (V[0] - crotch_point[0])
    ankle_outseam = np.array([W_x + HEM_HALF, ankle_y])
    ankle_inseam = np.array([W_x - HEM_HALF, ankle_y])
    knee_y = ankle_y + knee_depth
    knee_outseam = on_line(C, ankle_outseam, y=knee_y)
    knee_inseam = on_line(crotch_point, ankle_inseam, y=knee_y)

    outline = [("line", H, N)]
    prev = N
    for leg_in_, point, leg_out_ in dart_points:
        outline.append(("line", prev, leg_in_))
        outline.append(("dart", leg_in_, point))
        outline.append(("dart", point, leg_out_))
        prev = leg_out_
    outline.append(("line", prev, O))
    outline.append(("cubic_curve", lambda t: curve_hip(O, C, t), O, C))
    outline.append(("line", C, ankle_outseam))
    outline.append(("line", ankle_outseam, ankle_inseam))
    outline.append(("cubic_curve",
                    lambda t: curve_inseam(ankle_inseam, crotch_point, t),
                    ankle_inseam, crotch_point))
    outline.append(("cubic_curve",
                    lambda t: curve_crotch(crotch_top, crotch_point, 1 - t),
                    crotch_point, crotch_top))
    outline.append(("line", crotch_top, H))

    return SimpleNamespace(
        back_width=back_width, hip_depth=hip_depth, knee_depth=knee_depth, pitch=pitch,
        H=H, N=N, O=O, G=G, C=C, D=D,
        crotch_top=crotch_top, crotch_point=crotch_point,
        ankle_outseam=ankle_outseam, ankle_inseam=ankle_inseam,
        knee_outseam=knee_outseam, knee_inseam=knee_inseam,
        n_darts=1, intake_each=DART_INTAKE, dart_points=dart_points,
        outline=outline,
        construction_lines=[(G, D)],
        dart_lines=[],
    )
