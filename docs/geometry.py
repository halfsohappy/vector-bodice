"""Shared drafting-curve primitives.

Pattern pieces use these to turn drafting points into curve segments.
Kept separate from render.py so drafting math never depends on rendering.
"""

import numpy as np


def cubic_bezier(P0, P1, P2, P3, t):
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)[:, None]
    pts = ((1-t)**3 * P0 + 3*(1-t)**2*t * P1 +
           3*(1-t)*t**2 * P2 + t**3 * P3)
    return pts[0] if scalar else pts


def xon_line(P1, P2, x):
    """Given an x value, find the point on the line P1–P2 with that x value."""
    m = (P1[1] - P2[1]) / (P1[0] - P2[0])
    b = P2[1] - m * P2[0]
    y = m * x + b
    return np.array([x, y])


def catmull_rom_segment(P_prev, P0, P1, P_next, t):
    """Cubic Bézier from P0→P1 using Catmull-Rom tangents from neighbours."""
    CP1 = P0 + (P1 - P_prev) / 6.0
    CP2 = P1 - (P_next - P0) / 6.0
    return cubic_bezier(P0, CP1, CP2, P1, t)


def catmull_rom_chain(points):
    """Build cubic_curve outline entries through an ordered list of points.

    Uses Catmull-Rom interpolation so the resulting curve is C¹-continuous.
    Virtual points are reflected at the endpoints to produce natural tangents.
    Returns a list of ("cubic_curve", func, P0, P1) outline segments.
    """
    P_before = points[0] - (points[1] - points[0])    # virtual before first
    P_after  = points[-1] - (points[-2] - points[-1])  # virtual after last
    extended = [P_before] + list(points) + [P_after]

    segments = []
    for i in range(len(points) - 1):
        pp, ps, pe, pn = (extended[i], extended[i+1],
                          extended[i+2], extended[i+3])
        # capture by value via default args
        def _curve(t, _pp=pp, _ps=ps, _pe=pe, _pn=pn):
            return catmull_rom_segment(_pp, _ps, _pe, _pn, t)
        segments.append(("cubic_curve", _curve, ps, pe))
    return segments
