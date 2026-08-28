"""Belt-loop strip — one strip cut into six loops.

Adapted from Pattern Making for Fashion Design, 5th ed., p.588: "Cut a
pattern strip 3/4 inch wide and 13 1/2 inches long for six loops."

A plain rectangle, so render.rectangle_dims() reports it as a ruler-drafted
piece. Notches mark where to cut the strip into the six individual loops.

Shared with patterns/slack and patterns/jean, same as waistband.py.
"""

import numpy as np
from types import SimpleNamespace

WIDTH = 0.75        # in, p.588
LENGTH = 13.5       # in, p.588
LOOP_COUNT = 6


def build():
    """Compute the belt-loop strip.  Returns a SimpleNamespace."""
    A = np.array([0.0, 0.0])
    B = np.array([LENGTH, 0.0])
    C = np.array([LENGTH, WIDTH])
    D = np.array([0.0, WIDTH])

    outline = [
        ("line", A, B),
        ("line", B, C),
        ("line", C, D),
        ("line", D, A),
    ]

    # Cut marks between loops (5 internal divisions for 6 loops).
    step = LENGTH / LOOP_COUNT
    notches = [np.array([step * i, 0.0]) for i in range(1, LOOP_COUNT)]
    cut_lines = [(np.array([step * i, 0.0]), np.array([step * i, WIDTH]))
                 for i in range(1, LOOP_COUNT)]

    return SimpleNamespace(
        length=LENGTH, width=WIDTH, loop_count=LOOP_COUNT,
        loop_length=step,
        A=A, B=B, C=C, D=D,
        outline=outline,
        construction_lines=cut_lines,
        dart_lines=[],
        notches=notches,
    )
