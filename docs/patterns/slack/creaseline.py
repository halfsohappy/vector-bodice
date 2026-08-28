"""Contour pant with creaseline flare — the slack back split in two.

Adapted from Pattern Making for Fashion Design, 5th ed., "Contour Pant
with Creaseline Flare" (p.611):

    The contour pant design is achieved by removing excess from under the
    buttocks via a seam replacing the back creaseline.  The pant hemline
    flares out at the side and back seamlines, dropping low at the center
    back and rising at center front.  The design is based on the slack
    foundation.

Figure 1 traces the back "from side seam to creaseline", Figure 2 "from
inseam to grainline" — so the back becomes TWO pieces meeting at a new
seam where the creaseline used to be.  Both halves get the same shaping,
mirrored, so the two edges still sew together:

    Shift dart to the creaseline to remove through the styleline.
    Measure in 1/2 inch from the creaseline at crotch level.  Label A.
    Buttocks contour: from point A, draw an inward curve blending with an
    outward curve to the dart point.  Continue the curve from A to knee
    level.
    Hemline flare: extend hemline 1 1/2 inches.  Measure down 1 1/2 inches
    from the creaseline.  Measure down 1/2 inch at the side seam.  Draw
    straight lines from marks to knee levels.  Draw a curved hemline.

Figure 3 gives the front the matching hem treatment (extended 1 1/2in at
side and inseam, dropped 1/2in at each end, blended with an inward curve).

The contour curve is built with geometry.catmull_rom_chain through the
book's own landmarks (waist, dart point, A, knee, hem) rather than
reproducing a hand-drawn French curve — the same class of approximation
used for every other curve in this tool.
"""

import numpy as np
from types import SimpleNamespace

from geometry import catmull_rom_chain, cubic_bezier, on_line

CONTOUR_IN = 0.5        # in, "measure in 1/2 inch from the creaseline at crotch level"
HEM_EXTEND = 1.5        # in, "extend hemline 1 1/2 inches"
HEM_DROP_SEAM = 1.5     # in, dropped this much at the creaseline seam
HEM_DROP_SIDE = 0.5     # in, dropped this much at the side seam / inseam


def _hem_curve(P_start, P_end, bow, t):
    """Gently curved hemline between two points, bowing by *bow* inches."""
    mid_dip = np.array([0.0, -bow])
    C1 = P_start + (P_end - P_start) / 3.0 + mid_dip
    C2 = P_start + 2.0 * (P_end - P_start) / 3.0 + mid_dip
    return cubic_bezier(P_start, C1, C2, P_end, t)


def _contour_edge(waist_pt, dart_pt, A, knee_pt, hem_pt):
    """The new creaseline seam, waist down to hem, through the book's
    landmarks.  Catmull-Rom keeps it C1-continuous, so the 'inward curve
    blending with an outward curve' reads as one smooth line."""
    return catmull_rom_chain([waist_pt, dart_pt, A, knee_pt, hem_pt])


def back_halves(back, curve_hip, curve_inseam, curve_crotch):
    """Split a slack back panel into (side half, inner half).

    back      — the slack back panel namespace
    curve_*   — the panel's own curve functions, passed in so this module
                does not have to reach back into patterns.trouser
    """
    crease_x = float(back.V[0])
    knee_y = float(back.YY[1])
    ankle_y = float(back.Y[1])
    crotch_y = float(back.G[1])

    # The dart is absorbed into the new seam: each half gives up half its
    # intake at the waist, so the two edges still meet when sewn.
    if back.dart_points:
        (leg_in, dart_tip, leg_out) = back.dart_points[0]
        intake = float(leg_out[0] - leg_in[0])
        dart_depth_y = float(dart_tip[1])
    else:
        intake, dart_depth_y = 0.0, crotch_y

    waist_y_at_crease = float(on_line(back.S, back.O, x=crease_x)[1])
    half = intake / 2.0

    # Hem: drops HEM_DROP_SEAM at the new seam, HEM_DROP_SIDE at the
    # side seam / inseam, and extends HEM_EXTEND outward at each.
    hem_seam_y = ankle_y - HEM_DROP_SEAM
    hem_out_y = ankle_y - HEM_DROP_SIDE

    side_hem_out = np.array([float(back.Y[0]) + HEM_EXTEND, hem_out_y])
    inner_hem_out = np.array([float(back.Z[0]) - HEM_EXTEND, hem_out_y])
    side_hem_seam = np.array([crease_x, hem_seam_y])
    inner_hem_seam = np.array([crease_x, hem_seam_y])

    def make_half(is_side):
        sgn = 1.0 if is_side else -1.0
        waist_pt = np.array([crease_x + sgn * half, waist_y_at_crease])
        dart_pt = np.array([crease_x + sgn * half * 0.5, dart_depth_y])
        A = np.array([crease_x + sgn * CONTOUR_IN, crotch_y])
        knee_pt = np.array([crease_x, knee_y])
        hem_pt = side_hem_seam if is_side else inner_hem_seam
        return waist_pt, _contour_edge(waist_pt, dart_pt, A, knee_pt, hem_pt), A

    side_waist, side_edge, side_A = make_half(True)
    inner_waist, inner_edge, inner_A = make_half(False)

    # ── side half: creaseline seam -> waist -> outseam -> hem ────────────
    side_outline = [("line", side_waist, back.O)]
    side_outline.append(("cubic_curve", lambda t: curve_hip(back.O, back.C, t),
                         back.O, back.C))
    side_outline.append(("line", back.C, side_hem_out))
    side_outline.append(("cubic_curve",
                         lambda t: _hem_curve(side_hem_out, side_hem_seam, 0.25, t),
                         side_hem_out, side_hem_seam))
    side_outline.extend(reversed([("cubic_curve",
                                   (lambda f: lambda t: f(1 - t))(seg[1]),
                                   seg[3], seg[2]) for seg in side_edge]))

    side = SimpleNamespace(
        O=back.O, C=back.C, A=side_A,
        waist=side_waist, hem_out=side_hem_out, hem_seam=side_hem_seam,
        outline=side_outline,
        construction_lines=[],
        dart_lines=[],
        notches=[back.C, np.array([crease_x, knee_y])],
    )

    # ── inner half: centre back -> waist -> creaseline seam -> hem ───────
    inner_outline = [("line", back.S, inner_waist)]
    inner_outline.extend(inner_edge)
    inner_outline.append(("cubic_curve",
                          lambda t: _hem_curve(inner_hem_seam, inner_hem_out, 0.25, t),
                          inner_hem_seam, inner_hem_out))
    inner_outline.append(("cubic_curve",
                          lambda t: curve_inseam(inner_hem_out, back.I, t),
                          inner_hem_out, back.I))
    inner_outline.append(("cubic_curve",
                          lambda t: curve_crotch(back.X, back.I, 1 - t),
                          back.I, back.X))
    inner_outline.append(("line", back.X, back.S))

    inner = SimpleNamespace(
        S=back.S, X=back.X, I=back.I, A=inner_A,
        waist=inner_waist, hem_out=inner_hem_out, hem_seam=inner_hem_seam,
        outline=inner_outline,
        construction_lines=[],
        dart_lines=[],
        notches=[np.array([crease_x, knee_y])],
    )
    return side, inner


def front_hem(front):
    """Figure 3's matching front hem: extended 1 1/2in at side and inseam,
    dropped 1/2in at each end, blended with an inward curve.  Returns the
    new (outseam ankle, inseam ankle) points."""
    y = float(front.U[1]) - HEM_DROP_SIDE
    return (np.array([float(front.U[0]) + HEM_EXTEND, y]),
            np.array([float(front.V[0]) - HEM_EXTEND, y]))
