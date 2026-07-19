"""Waist tie strip — a plain rectangle, folded and turned during sewing.

Constants are the tutorial's cm values converted through a precise 1/2.54
factor (see skirt_panel.py's module docstring). Drafted at the tutorial's
literal given cut dimensions (100 x 8cm). This is a flat notion consumed
entirely by folding-and-turning, not a shaped seam — there's nothing useful
for a seam-allowance overlay to show here, so it renders with
seam_allowance=0 (see settings.py). Cut 2.
"""

import numpy as np
from types import SimpleNamespace

CM_TO_IN = 1 / 2.54

LENGTH = 100.0 * CM_TO_IN
WIDTH  = 8.0 * CM_TO_IN


def build():
    """Compute the tie's points and outline.  Returns a SimpleNamespace."""
    A = np.array([0.0,    0.0])
    B = np.array([LENGTH, 0.0])
    C = np.array([LENGTH, WIDTH])
    D = np.array([0.0,    WIDTH])

    outline = [
        ("line", A, B),
        ("line", B, C),
        ("line", C, D),
        ("line", D, A),
    ]

    return SimpleNamespace(
        A=A, B=B, C=C, D=D,
        outline=outline,
        construction_lines=[],
        dart_lines=[],
    )
