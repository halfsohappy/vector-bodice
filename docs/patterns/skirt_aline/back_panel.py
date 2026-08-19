"""Back panel of the A-line skirt — built on skirt_basic's sloper.

See front_panel.py's module docstring for the shared modeling notes
(combined closed-form flare, unchanged waist/hip area).
"""

import numpy as np
from types import SimpleNamespace

from patterns.skirt_basic import back_panel as basic
from .front_panel import flare_amount, curve_aline_side_seam


def build(hip_arc_back, hip_depth_back, skirt_length, n_darts, intake_each):
    """Compute the A-line back panel.  Returns a SimpleNamespace.  Same
    signature as skirt_basic.back_panel.build()."""
    base = basic.build(hip_arc_back, hip_depth_back, skirt_length,
                       n_darts, intake_each)

    remaining_darts = list(base.dart_points)
    new_B = base.B.copy()
    if remaining_darts:
        # the dart closest to the side seam transfers to the hem
        _, transfer_point, transfer_leg_out = remaining_darts.pop()
        dart_point_to_hem = skirt_length - basic.DART_LEG_LENGTH
        flare = flare_amount(basic.DART_LEG_LENGTH, intake_each, dart_point_to_hem)
        new_B = base.B + np.array([flare, 0.0])

    hip_frac = hip_depth_back / skirt_length

    outline = []
    prev = base.D
    for leg_in, point, leg_out in remaining_darts:
        outline.append(("line", prev, leg_in))
        outline.append(("dart", leg_in, point))
        outline.append(("dart", point, leg_out))
        prev = leg_out
    outline.append(("line", prev, base.A))
    outline.append(("cubic_curve",
                    lambda t: curve_aline_side_seam(base.A, new_B, hip_frac, t),
                    base.A, new_B))
    outline.append(("line", new_B, base.F))
    outline.append(("line", base.F, base.D))

    return SimpleNamespace(
        n_darts=len(remaining_darts), intake_each=intake_each,
        back_width=base.back_width,
        D=base.D, A=base.A, G=base.G, C=base.C, F=base.F, B=new_B,
        dart_points=remaining_darts,
        outline=outline,
        construction_lines=base.construction_lines,
        dart_lines=[],
    )
