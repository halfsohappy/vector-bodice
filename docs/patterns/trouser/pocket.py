"""Slash-pocket set for the trouser front — four pieces.

Adapted from Pattern Making for Fashion Design, 5th ed., "Pocket Draft for
Trouser" (p.589).  Everything is anchored on the front panel's side-waist
point X and its outseam, using the book's Figure 5 measurements:

    X-D = 6 1/2 inches        (down the outseam; D is the entry's lower end)
    X-C = 1 3/4 inches        (along the waist; C is the entry's upper end)
    C-D                       the pocket entry itself
    C-E = 1 1/2 inches        (further along the waist)
    E-F                       parallel with C-D, F on the outseam
    F raised 1/2 inch and blended

and the Figure 4a lining box (12 inches deep, 4 inches wide).

Four layers come out of that (Figure 6a-d):
    facing        C-E-F-D   the self-fabric strip behind the opening
    pouch         A-C-D-G-B the bag itself
    backing       X-E-F     the piece replacing the corner the entry cuts off
    full lining   A-X-G-B   the support layer under the whole pocket area

Two documented interpretations, both flagged in the manifest notes:
  * The book shapes the lining by combining the waist darts and pivoting on
    a hinge.  That pivot is approximated in closed form here, the same way
    patterns/skirt_aline handles its dart transfer, rather than literally
    rotating the pattern.
  * Figure 4a's "12 inches" and "4 inches" are read as the bag's depth below
    the waist and how far its inner edge sits beyond the entry point C.

Shared with patterns/slack and patterns/jean, same as waistband.py.
"""

import numpy as np
from types import SimpleNamespace

from geometry import on_line

ENTRY_DOWN = 6.5      # in, X-D down the outseam
ENTRY_IN = 1.75       # in, X-C along the waist
FACING_WIDTH = 1.5    # in, C-E along the waist
F_RISE = 0.5          # in, "raise F 1/2 inch and blend"
BAG_DEPTH = 12.0      # in, lining depth below the waist (Fig 4a)
BAG_WIDTH = 4.0       # in, how far the bag's inner edge sits beyond C (Fig 4a)

ENTRY_SA = 0.25       # in, "add 1/2-inch seams, 1/4 inch at entry"
STD_SA = 0.5          # in


def _outseam_path(panel):
    """Dense polyline of the front panel's outseam, from the side waist
    down to the ankle — the hip curve plus the straight leg below it."""
    pts = []
    started = False
    for seg in panel.outline:
        if seg[0] == "cubic_curve":
            if not started:
                started = True          # the hip curve starts the outseam
            else:
                break                    # the inseam curve ends it
            _, func, _p0, _p1 = seg
            pts.extend(func(t) for t in np.linspace(0, 1, 60))
        elif started and seg[0] == "line":
            if not pts or not np.allclose(pts[-1], seg[1], atol=1e-9):
                pts.append(np.asarray(seg[1], float))
            pts.append(np.asarray(seg[2], float))
            if abs(float(seg[2][1]) - float(seg[1][1])) < 1e-9:
                pts.pop()               # that was the hem, not the outseam
                break
    return np.array(pts, float)


def _walk(path, dist):
    """Point at *dist* of arc length along a polyline."""
    run = 0.0
    for a, b in zip(path[:-1], path[1:]):
        step = float(np.linalg.norm(b - a))
        if run + step >= dist:
            return a + (b - a) * ((dist - run) / step)
        run += step
    return path[-1].copy()


def _ray_hits_path(origin, direction, path):
    """First point where the ray origin + t*direction (t > 0) crosses the
    polyline.  Falls back to the polyline's far end if it never does."""
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    best_t, best_pt = None, None
    for a, b in zip(path[:-1], path[1:]):
        e = b - a
        denom = d[0] * e[1] - d[1] * e[0]
        if abs(denom) < 1e-12:
            continue
        t = ((a[0] - o[0]) * e[1] - (a[1] - o[1]) * e[0]) / denom
        s = ((a[0] - o[0]) * d[1] - (a[1] - o[1]) * d[0]) / denom
        if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9 and (best_t is None or t < best_t):
            best_t, best_pt = t, a + e * s
    return best_pt if best_pt is not None else path[-1].copy()


def _arc_length_to(path, target):
    """Arc length from the start of the polyline to the point on it nearest
    to *target*."""
    target = np.asarray(target, float)
    run = 0.0
    best_d, best_run = None, 0.0
    for a, b in zip(path[:-1], path[1:]):
        e = b - a
        le2 = float(np.dot(e, e))
        s = 0.0 if le2 < 1e-18 else max(0.0, min(1.0, float(np.dot(target - a, e)) / le2))
        proj = a + e * s
        dist = float(np.linalg.norm(target - proj))
        if best_d is None or dist < best_d:
            best_d, best_run = dist, run + float(np.linalg.norm(proj - a))
        run += float(np.linalg.norm(e))
    return best_run


def build(panel):
    """Draft the four pocket pieces from a front panel namespace.

    panel needs .side_waist and .cf_waist (each front panel declares those
    aliases, since the letters differ between foundations) and an outline
    whose outseam runs from the side waist down to the ankle.
    """
    X = np.asarray(panel.side_waist, float)      # the book's X
    inner = np.asarray(panel.cf_waist, float)     # centre-front end of the waistline

    waist_dir = inner - X
    waist_dir = waist_dir / np.linalg.norm(waist_dir)     # points toward centre front

    path = _outseam_path(panel)
    C = X + ENTRY_IN * waist_dir
    D = _walk(path, ENTRY_DOWN)
    E = C + FACING_WIDTH * waist_dir

    # F: where a line from E, parallel to the entry C-D, meets the outseam.
    # The book then raises F by 1/2in and blends.
    F = _ray_hits_path(E, D - C, path)
    F = _walk(path, max(0.0, _arc_length_to(path, F) - F_RISE))

    bag_top_y = float(C[1])
    bag_bottom_y = bag_top_y - BAG_DEPTH
    inner_x = float(C[0]) + BAG_WIDTH * float(waist_dir[0])
    A = np.array([inner_x, float(on_line(X, inner, x=inner_x)[1])])   # bag inner top, on the waist
    B = np.array([inner_x, bag_bottom_y])                              # bag inner bottom
    G = np.array([float(_walk(path, BAG_DEPTH)[0]), bag_bottom_y])     # bag outer bottom, at the outseam

    def piece(points, entry_edge=None):
        pts = [np.asarray(p, float) for p in points]
        outline = [("line", pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        return outline, entry_edge

    facing_outline, _ = piece([C, E, F, D])
    pouch_outline, _ = piece([A, C, D, G, B])
    backing_outline, _ = piece([X, E, F])
    lining_outline, _ = piece([A, X, G, B])

    def ns(outline, **pts):
        return SimpleNamespace(outline=outline, construction_lines=[], dart_lines=[], **pts)

    return {
        "pocket_facing": ns(facing_outline, C=C, E=E, F=F, D=D, entry=(C, D)),
        "pocket_pouch": ns(pouch_outline, A=A, C=C, D=D, G=G, B=B, entry=(C, D)),
        "pocket_backing": ns(backing_outline, X=X, E=E, F=F, entry=None),
        "pocket_lining": ns(lining_outline, A=A, X=X, G=G, B=B, entry=None),
    }


def entry_seam_allowance_fn(ns, seam_allowance):
    """Per-edge seam allowance: the pocket entry gets the book's tighter
    1/4in, every other edge the standard 1/2in — the same per-edge override
    patterns/ines_skirt/settings.py uses for its waistband ends."""
    entry = getattr(ns, "entry", None)
    if entry is None:
        return None
    a, b = (np.asarray(p, float) for p in entry)

    def fn(run):
        if len(run) == 2 and (
            (np.allclose(run[0], a, atol=1e-6) and np.allclose(run[1], b, atol=1e-6)) or
            (np.allclose(run[0], b, atol=1e-6) and np.allclose(run[1], a, atol=1e-6))):
            return min(ENTRY_SA, seam_allowance)
        return seam_allowance
    return fn
