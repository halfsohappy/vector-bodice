"""Culotte waistband — a folded, self-facing strip with a button extension.

Adapted from Pattern Making for Fashion Design, 5th ed., "Basic
Waistband" (p.237-238): length = waist − 0.5in, +1in extension for the
button/buttonhole, cut 2.5in wide (finishes 1.25in once folded). Unlike
patterns/ines_skirt's waistband, every edge here gets the same seam
allowance (the book states one uniform 0.5in for this piece) — no
per-edge override needed.
"""

import numpy as np
from types import SimpleNamespace

EXTENSION = 1.0        # in, button/buttonhole extension
CUT_WIDTH = 2.5         # in, finishes to 1.25in once folded


def build(waist_total):
    """Compute the waistband's points and outline.  Returns a
    SimpleNamespace.  waist_total is the full body waist circumference."""
    length = (waist_total - 0.5) + EXTENSION

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
