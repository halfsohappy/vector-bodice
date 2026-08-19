"""Front panel of the A-line skirt — built on skirt_basic's sloper.

Adapted from Pattern Making for Fashion Design, 5th ed., "A-Line Flared
Skirt" (p.239-240). The book achieves the flare with two separate paper
operations: closing the dart nearest the side seam (which swings the hem
outward as the dart's rotation carries down the slash line to the hem),
then an *additional* side-seam widening on top ("X-Y = one-half of A-B
space"). Per-user decision: both are folded into one closed-form flare
magnitude (not a literal dart-rotation simulation) — see module-level
comment on `flare_amount` below.

The waist/hip area is left exactly as drafted by skirt_basic (any darts
other than the transferred one stay put); only the side seam and hem
change, matching the book's own "the center front/back line is unchanged"
convention for A-line flare.
"""

import math

import numpy as np
from types import SimpleNamespace

from geometry import cubic_bezier
from patterns.skirt_basic import front_panel as basic


def flare_amount(dart_leg_length, intake, dart_point_to_hem):
    """Combined closed-form flare magnitude for one transferred dart:
    the chord swing at the hem from rotating the dart closed (standard
    isoceles-dart trigonometry), times 1.5 to fold in the book's separate
    side-seam-widening step (its own instruction adds "one-half of A-B
    space" again on top of the dart-transfer amount)."""
    half_angle = math.asin(min(1.0, (intake / 2) / dart_leg_length))
    hem_swing = 2 * dart_point_to_hem * math.sin(half_angle)
    return 1.5 * hem_swing


def curve_aline_side_seam(A, new_B, hip_frac, t):
    """Cubic Bézier from waist (A) to the widened hem (new_B): stays
    close to straight/vertical through the hip (fraction hip_frac of the
    total height), then swings out to the wider hem."""
    height = A[1] - new_B[1]
    C1 = A + np.array([0.0, -height * hip_frac])
    C2 = new_B + np.array([0.0, height * (1 - hip_frac) * 0.5])
    return cubic_bezier(A, C1, C2, new_B, t)


def build(hip_arc_front, hip_depth_front, skirt_length, n_darts, intake_each):
    """Compute the A-line front panel.  Returns a SimpleNamespace.
    Same signature as skirt_basic.front_panel.build()."""
    base = basic.build(hip_arc_front, hip_depth_front, skirt_length,
                       n_darts, intake_each)

    remaining_darts = list(base.dart_points)
    new_B = base.B.copy()
    if remaining_darts:
        # the dart closest to the side seam transfers to the hem
        _, transfer_point, transfer_leg_out = remaining_darts.pop()
        dart_point_to_hem = skirt_length - basic.DART_LEG_LENGTH
        flare = flare_amount(basic.DART_LEG_LENGTH, intake_each, dart_point_to_hem)
        new_B = base.B + np.array([flare, 0.0])

    hip_frac = hip_depth_front / skirt_length

    outline = []
    prev = base.H
    for leg_in, point, leg_out in remaining_darts:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, base.A))
    outline.append(("cubic_curve",
                    lambda t: curve_aline_side_seam(base.A, new_B, hip_frac, t),
                    base.A, new_B))
    outline.append(("line", new_B, base.J))
    outline.append(("line", base.J, base.H))

    return SimpleNamespace(
        n_darts=len(remaining_darts), intake_each=intake_each,
        front_width=base.front_width,
        H=base.H, A=base.A, I=base.I, C=base.C, J=base.J, B=new_B,
        dart_points=remaining_darts,
        outline=outline,
        construction_lines=base.construction_lines,
        dart_lines=[],
    )
