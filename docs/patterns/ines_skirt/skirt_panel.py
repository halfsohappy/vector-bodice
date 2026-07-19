"""Skirt panel piece — a quarter-circle annulus.

Both the front and back skirt panel are geometrically identical (a pivot
corner, a waist arc at radius, a hem arc further out, joined by two straight
radial edges); build() is called twice with the same measurements to produce
the two manifest pieces. They differ only in which pieces get attached to
them during sewing (pockets on the back, a different waistband length on
each) — not in this drafted shape.

Adapted from "INES Skirt" tutorial (The Mystical Attic), which is metric.
This repo's engine and every existing pattern are inch-based (render.py's
SCALE is a fixed px-per-inch), so every constant here is the tutorial's cm
value converted through a single precise factor (1/2.54), not the
tutorial's own separately-rounded inch call-outs — chaining those would
compound rounding error into the geometry.

The tutorial bakes a 1.5cm seam allowance directly into its measurements
and cuts the waist curve at exactly the radius it computes, with no
separate SA offset. Here the drafted radius is treated as the finished
SEAMLINE instead, and the renderer adds a proper seam allowance offset
outward for cutting — the hem, however, is not a seam (it's a narrow-hem
turn-under), so its allowance is baked directly into the drafted radius and
excluded from the render overlay.
"""

import math

import numpy as np
from types import SimpleNamespace

from geometry import cubic_bezier

CM_TO_IN = 1 / 2.54


def curve_quarter_arc(P_start, P_end, center, t):
    """Cubic Bézier approximation of a 90° circular arc, P_start on the +x
    axis from center, P_end on the -y axis from center (same radius).
    Same kappa-constant technique as front_bodice.curve_neck."""
    r = np.linalg.norm(P_start - center)
    k = 0.5523 * r
    C1 = P_start + np.array([0.0, -k])
    C2 = P_end   + np.array([k,    0.0])
    return cubic_bezier(P_start, C1, C2, P_end, t)


def build(waist, skirt_length, hem_allowance):
    """Compute the panel's points and outline.  Returns a SimpleNamespace.

    waist:         waist circumference (in)
    skirt_length:  desired finished length, waist seamline to hem (in)
    hem_allowance: extra fabric folded under for the narrow hem (in) —
                   baked directly into the drafted hem radius, not a
                   render-time seam allowance (see module docstring)
    """
    overlap_ease   = 16 * CM_TO_IN            # tutorial's "+16cm" wrap-overlap ease (not SA)
    adjusted_waist = waist + overlap_ease
    waist_radius   = adjusted_waist / math.pi      # = (adjusted_waist/(2*pi))*2
    panel_length   = skirt_length + hem_allowance
    hem_radius     = waist_radius + panel_length

    O = np.array([0.0, 0.0])                 # pivot — construction reference only
    K = np.array([waist_radius, 0.0])
    L = np.array([0.0, -waist_radius])
    M = np.array([hem_radius, 0.0])
    N = np.array([0.0, -hem_radius])

    outline = [
        ("line", K, M),                                                        # side/pocket edge
        ("cubic_curve", lambda t: curve_quarter_arc(M, N, O, t), M, N),        # hem (no SA overlay)
        ("line", N, L),                                                        # side/pocket edge
        ("cubic_curve", lambda t: curve_quarter_arc(K, L, O, 1-t), L, K),      # waist (gets SA overlay)
    ]

    construction_lines = [(O, K), (O, L)]    # pivot reference axes

    return SimpleNamespace(
        waist_radius=waist_radius, hem_radius=hem_radius, panel_length=panel_length,
        O=O, K=K, L=L, M=M, N=N,
        outline=outline,
        construction_lines=construction_lines,
        dart_lines=[],
        # waist arc only — the hem is excluded from the rendered SA overlay
        curve_seam_segments=[outline[3]],
    )
