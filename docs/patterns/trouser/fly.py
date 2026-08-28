"""Fly and shield — the two pieces of a trouser fly-front zipper closure.

Adapted from Pattern Making for Fashion Design, 5th ed., p.584 and p.588
(Fig 2a, 2c). The fly is stitched to the right pant and the shield to the
left; both are plain rectangles, so render.rectangle_dims() gives them a
"draft with a ruler" callout instead of requiring a printed pattern.

The book states these two ways, and they agree: on p.584 in terms of the
zipper ("fly 1 1/2 inches and 1/2 inch fold-back longer than the zipper;
the shield 2 1/2 inches wide and 1 inch longer than the zipper"), and on
p.588 as finished numbers for the standard 7-inch pant zipper (fly
1 1/2 x 7 1/2in, shield 2 1/2 x 8 1/2in). The p.588 numbers are used
directly; a zipper_length argument re-derives them for any other zipper.

This module lives in patterns/trouser because that is where the shared
pant pieces live (see waistband.py) — patterns/slack and patterns/jean
import it rather than duplicating it.
"""

import numpy as np
from types import SimpleNamespace

STD_ZIPPER = 7.0        # in, the zipper the book's stated sizes assume

FLY_WIDTH = 1.5         # in, p.588 Fig 2a
FLY_OVER_ZIP = 0.5      # in, fly runs this much longer than the zipper

SHIELD_WIDTH = 2.5      # in, p.588 Fig 2c
SHIELD_OVER_ZIP = 1.5   # in, shield runs this much longer than the zipper


def _rect(width, height):
    A = np.array([0.0, 0.0])
    B = np.array([width, 0.0])
    C = np.array([width, height])
    D = np.array([0.0, height])
    return A, B, C, D, [
        ("line", A, B),
        ("line", B, C),
        ("line", C, D),
        ("line", D, A),
    ]


def build_fly(zipper_length=STD_ZIPPER):
    """Fly piece — stitched to the right pant.  Returns a SimpleNamespace."""
    height = zipper_length + FLY_OVER_ZIP
    A, B, C, D, outline = _rect(FLY_WIDTH, height)
    return SimpleNamespace(
        width=FLY_WIDTH, height=height,
        A=A, B=B, C=C, D=D,
        outline=outline,
        construction_lines=[],
        dart_lines=[],
    )


def build_shield(zipper_length=STD_ZIPPER):
    """Shield piece — stitched to the left pant, behind the zipper."""
    height = zipper_length + SHIELD_OVER_ZIP
    A, B, C, D, outline = _rect(SHIELD_WIDTH, height)
    # The shield folds in half lengthwise to enclose the zipper tape.
    fold = (np.array([SHIELD_WIDTH / 2, 0.0]), np.array([SHIELD_WIDTH / 2, height]))
    return SimpleNamespace(
        width=SHIELD_WIDTH, height=height,
        A=A, B=B, C=C, D=D,
        outline=outline,
        construction_lines=[fold],
        dart_lines=[],
    )
