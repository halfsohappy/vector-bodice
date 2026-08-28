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

EXTENSION = 1.25        # in, button/buttonhole extension
CUT_WIDTH = 2.5          # in, finishes to 1.25in once folded


def build(front_waist, back_waist):
    """Compute the waistband's points and outline.  Returns a
    SimpleNamespace.

    front_waist/back_waist are the *drafted finished* waist lengths of one
    front and one back panel (i.e. after the darts fold out), not the raw
    body measurement.  Sizing the band from the draft rather than from the
    body guarantees it matches the pant it is being sewn to — the book's
    own "waist + 1/2 inch" band and its "waist arc + 1/4 inch per quarter"
    pant draft disagree by half an inch, which would otherwise show up as
    a band that will not reach.

    Notches mark where the two side seams and the centre back fall, so the
    band can be pinned to the assembled pant without measuring.
    """
    span = 2 * (front_waist + back_waist)   # the part that circles the waist
    length = span + EXTENSION

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

    # Walking from the centre-front edge (x=0): front panel, side seam,
    # back panel, centre back, back panel, side seam, front panel, then the
    # button extension runs past the far centre front.
    side1 = np.array([front_waist, 0.0])
    cb    = np.array([front_waist + back_waist, 0.0])
    side2 = np.array([front_waist + 2 * back_waist, 0.0])
    cf_end = np.array([span, 0.0])

    return SimpleNamespace(
        length=length, span=span, width=CUT_WIDTH,
        A=A, B=B, C=C, D=D,
        side1=side1, cb=cb, side2=side2, cf_end=cf_end,
        outline=outline,
        construction_lines=[fold_line],
        dart_lines=[],
        notches=[side1, cb, side2, cf_end],
    )
