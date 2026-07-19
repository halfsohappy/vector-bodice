"""Bias strip for finishing the pocket opening — a plain rectangle.

Constants are the tutorial's cm values converted through a precise 1/2.54
factor (see skirt_panel.py's module docstring). Drafted at the tutorial's
literal given cut dimensions (26 x 2cm). Cut on the bias (45°), unlike every
other piece in this pattern (straight/cross grain) — flagged as a text
annotation since there's no other grain-line indicator for a piece this
small. Flat notion, no seam-allowance overlay (seam_allowance=0, see
settings.py). Cut 2.
"""

import numpy as np
from types import SimpleNamespace

CM_TO_IN = 1 / 2.54

LENGTH = 26.0 * CM_TO_IN
WIDTH  = 2.0 * CM_TO_IN


def build():
    """Compute the bias strip's points and outline.  Returns a
    SimpleNamespace."""
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
        text_annotations=[("cut on the bias (45°)",
                           np.array([LENGTH / 2, WIDTH / 2]))],
    )
