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


# ── Construction operations ───────────────────────────────────────────────────
# Drafting-instruction primitives, shared by hand-written pattern pieces,
# the pattern studio evaluator, and studio-generated code.

def liney(P1, P2, x):
    """y of the line P1–P2 at the given x."""
    return float(xon_line(P1, P2, x)[1])


def linex(P1, P2, y):
    """x of the line P1–P2 at the given y."""
    m = (P1[0] - P2[0]) / (P1[1] - P2[1])
    b = P2[0] - m * P2[1]
    return float(m * y + b)


def on_line(P1, P2, x=None, y=None):
    """Point on the line P1–P2 at the given x (or y)."""
    if x is not None:
        return np.array([x, liney(P1, P2, x)])
    return np.array([linex(P1, P2, y), y])


def circle_h(center, radius, y, branch="left"):
    """Intersection of a circle with the horizontal line at *y*.
    branch: "left" or "right" of the center."""
    dx2 = radius**2 - (y - center[1])**2
    if dx2 < 0:
        raise ValueError(
            f"circle of radius {radius:g} around ({center[0]:g}, {center[1]:g}) "
            f"does not reach the line y = {y:g}")
    dx = np.sqrt(dx2)
    x = center[0] - dx if branch == "left" else center[0] + dx
    return np.array([x, y])


def circle_v(center, radius, x, branch="down"):
    """Intersection of a circle with the vertical line at *x*.
    branch: "down" or "up" of the center."""
    dy2 = radius**2 - (x - center[0])**2
    if dy2 < 0:
        raise ValueError(
            f"circle of radius {radius:g} around ({center[0]:g}, {center[1]:g}) "
            f"does not reach the line x = {x:g}")
    dy = np.sqrt(dy2)
    y = center[1] - dy if branch == "down" else center[1] + dy
    return np.array([x, y])


def along(start, toward, dist):
    """Point at *dist* from *start* in the direction of *toward*."""
    direction = (toward - start) / np.linalg.norm(toward - start)
    return start + dist * direction


def curve_length(func, n=200):
    """Numerical arc length of a parametric curve func(t), t in [0, 1]."""
    ts = np.linspace(0, 1, n)
    pts = np.array([func(t) for t in ts])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def intersect_lines(A1, A2, B1, B2):
    """Intersection of the (infinite) lines A1–A2 and B1–B2."""
    d1 = A2 - A1
    d2 = B2 - B1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        raise ValueError("lines are parallel — no intersection")
    t = ((B1[0] - A1[0]) * d2[1] - (B1[1] - A1[1]) * d2[0]) / denom
    return A1 + t * d1


def cubic_from_tangents(P0, P1, dir0, len0, dir1, len1, t):
    """Cubic Bézier from P0→P1 with tangent directions at both ends.

    dir0/dir1 are direction-of-travel vectors at the start/end (need not be
    unit length); len0/len1 are the control-handle lengths.
        CP1 = P0 + len0 · unit(dir0)
        CP2 = P1 − len1 · unit(dir1)
    The existing curve_* constructions (armhole, neck) are special cases.
    """
    d0 = np.asarray(dir0, float)
    d1 = np.asarray(dir1, float)
    CP1 = P0 + len0 * d0 / np.linalg.norm(d0)
    CP2 = P1 - len1 * d1 / np.linalg.norm(d1)
    return cubic_bezier(P0, CP1, CP2, P1, t)


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
