"""In-seam pocket bag half — a rectangle with one corner cut on the diagonal.

Two mirror-image halves (side 1, side 2) make one pocket bag; build() is
called with mirror=False / mirror=True for the two manifest pieces.

All constants are the tutorial's cm values converted through a precise
1/2.54 factor — see skirt_panel.py's module docstring for why. Drafted at
the tutorial's literal given cut dimensions (33 x 17cm rectangle, corner cut
from (0, 12) to (10, 33) — verified against their stated 23cm diagonal
length and 7cm remaining top segment) rather than reduced for baked-in SA.

The tutorial actually mixes seam allowances across this piece's 5 edges
(0.7cm on the internally French-seamed and bias-bound edges, 1.5cm on the
edge caught in the side seam) — reconstructing that precisely from a
hand-drawn diagram is more risk than it's worth for a functional inner
piece. Per-piece decision: treat the given dimensions as the seamline and
apply one uniform, standard seam allowance to every edge (see settings.py)
— always at least as generous as the tutorial's smaller values, trimmable
if it turns out too generous anywhere.
"""

import numpy as np
from types import SimpleNamespace

CM_TO_IN = 1 / 2.54

WIDTH   = 17.0 * CM_TO_IN   # pocket rectangle width
HEIGHT  = 33.0 * CM_TO_IN   # pocket rectangle height
NOTCH_Y = 12.0 * CM_TO_IN   # left edge — where the diagonal mouth starts
TOP_X   = WIDTH - 7.0 * CM_TO_IN   # top edge — where the diagonal mouth ends


def build(mirror=False):
    """Compute one pocket-half's points and outline.  Returns a
    SimpleNamespace."""
    A = np.array([0.0,    0.0])
    B = np.array([WIDTH,  0.0])
    C = np.array([WIDTH,  HEIGHT])
    D = np.array([TOP_X,  HEIGHT])
    E = np.array([0.0,    NOTCH_Y])
    pts = [A, B, C, D, E]

    if mirror:
        pts = [np.array([WIDTH - p[0], p[1]]) for p in reversed(pts)]
    A, B, C, D, E = pts

    outline = [
        ("line", A, B),   # bottom / side — internal seam to the other half
        ("line", B, C),   # attach edge — caught in the panel's side seam
        ("line", C, D),   # mouth, flat segment — bias-bound
        ("line", D, E),   # mouth, diagonal — bias-bound
        ("line", E, A),   # side — internal seam to the other half
    ]

    return SimpleNamespace(
        A=A, B=B, C=C, D=D, E=E,
        outline=outline,
        construction_lines=[],
        dart_lines=[],
    )
