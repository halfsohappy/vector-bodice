"""Waistband piece — a folded, self-facing strip.

Front and back waistbands are the same flat rectangle shape, differing only
in length; build() is called twice with each length.

All constants are the tutorial's cm values converted through a precise
1/2.54 factor — see skirt_panel.py's module docstring for why.

The tutorial cuts a 10cm-tall strip that folds in half lengthwise to enclose
the skirt's raw waist edge (its own construction, p.13-15): 1.5cm SA on the
bottom long edge (sewn to the skirt), 1.5cm on the top long edge (folded
under and topstitched), leaving a finished flat height of 10 - 1.5 - 1.5 =
7cm (3.5cm per layer once folded). The two SHORT ends get a tighter 0.7cm
seam allowance for boxing/tunnel-closing the corner — not the standard
1.5cm — so this piece needs a per-edge seam_allowance_fn override, exactly
like patterns/bodice/settings.py's back_seam_allowance_fn for the center
back seam.
"""

import numpy as np
from types import SimpleNamespace

CM_TO_IN = 1 / 2.54

END_SA = 0.7 * CM_TO_IN            # fixed, technique-driven (boxed/tunnel corner)
CUT_HEIGHT = 10.0 * CM_TO_IN        # tutorial's raw strip height, for reference only
LONG_SA_BAKED = 1.5 * CM_TO_IN      # SA baked into CUT_HEIGHT on each long edge, removed below
HEIGHT = CUT_HEIGHT - 2 * LONG_SA_BAKED   # finished flat height (~2.76in = 7cm)

FRONT_WIDTH_CONST = 10.0 * CM_TO_IN   # tutorial's "waist - 10cm"
BACK_WIDTH_CONST  = 13.0 * CM_TO_IN   # tutorial's "waist + 13cm"


def front_length(waist):
    return (waist - FRONT_WIDTH_CONST) - 2 * END_SA


def back_length(waist):
    return (waist + BACK_WIDTH_CONST) - 2 * END_SA


def build(length):
    """Compute the waistband's points and outline for the given finished
    length (already net of the two 0.7cm-equivalent end allowances).
    Returns a SimpleNamespace."""
    A = np.array([0.0,    0.0])
    B = np.array([length, 0.0])
    C = np.array([length, HEIGHT])
    D = np.array([0.0,    HEIGHT])

    outline = [
        ("line", A, B),   # bottom — sewn to skirt waist arc
        ("line", B, C),   # short end — boxed/tunnel corner
        ("line", C, D),   # top — folded under, self-facing
        ("line", D, A),   # short end — boxed/tunnel corner
    ]

    fold_line = (np.array([0.0, HEIGHT / 2]), np.array([length, HEIGHT / 2]))

    return SimpleNamespace(
        length=length, height=HEIGHT,
        A=A, B=B, C=C, D=D,
        outline=outline,
        construction_lines=[fold_line],
        dart_lines=[],
        end_points=(B, C, D, A),   # short-end runs, for the SA override
    )
