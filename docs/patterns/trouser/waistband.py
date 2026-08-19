"""Trouser waistband — a plain, self-facing strip with a button extension.

Adapted from Pattern Making for Fashion Design, 5th ed., "Waist Band"
(p.588): length = waist + 1/2in ease + 1 1/4in extension, cut 2.5in wide
(finishes 1.25in once folded). This same recipe is cited by all three
pant foundations built from scratch (Trouser, Slack, Jean); patterns/slack
and patterns/jean import this module directly rather than duplicating it.

All 4 edges are plain "line" segments with no darts/curves between them,
so render.py's default seam-allowance heuristic would merge them into one
wraparound run and then silently drop the bottom edge (it sits at the
piece's bounding-box minimum y) — the same shape ines_skirt's waistband
already has to work around. Callers must pass waist_detect=False (and
merge_consecutive=False) when rendering this piece.
"""

import numpy as np
from types import SimpleNamespace

EASE = 0.5              # in, "waist measurement plus 1/2-inch ease"
EXTENSION = 1.25        # in, button/buttonhole extension
CUT_WIDTH = 2.5          # in, finishes to 1.25in once folded


def build(waist_total):
    """Compute the waistband's points and outline.  Returns a
    SimpleNamespace.  waist_total is the full body waist circumference."""
    length = waist_total + EASE + EXTENSION

    A = np.array([0.0, 0.0])
    B = np.array([length, 0.0])
    C = np.array([length, CUT_WIDTH])
    D = np.array([0.0, CUT_WIDTH])

    outline = [
        ("line", A, B),
        ("line", B, C),
        ("line", C, D),
        ("line", D, A),
    ]

    fold_line = (np.array([0.0, CUT_WIDTH / 2]), np.array([length, CUT_WIDTH / 2]))

    return SimpleNamespace(
        length=length, width=CUT_WIDTH,
        A=A, B=B, C=C, D=D,
        outline=outline,
        construction_lines=[fold_line],
        dart_lines=[],
    )
