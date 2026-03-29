import argparse
import numpy as np
from block import build

SCALE         = 300          # px per inch (300 DPI)
MARGIN_INCHES = 0.5          # half-inch whitespace border around bodice
LABEL_OFFSET  = 87           # px — fallback inward push toward interior
FONT_SIZE     = 34           # px
FONT_FAMILY   = "monospace"

# ── Per-label offset tables ───────────────────────────────────────────────────
# All values in model-space inches (dx rightward, dy upward).
# SVG y is flipped in _label_elements, so dy upward → negative SVG dy.
# Back and front use separate tables because shared points (e.g. O) sit on
# opposite sides of their respective pieces.

_BACK_LABEL_OFFSETS = {
    "A":   ( 0.71, -0.71),  # nape (315°)
    "GG":  ( 0.71,  0.71),  # bottom-left corner (45°)
    "AA":  ( 0.42, -0.91),  # above nape, neck RHS (295°)
    "DD":  (-0.87, -0.50),  # shoulder tip (210°)
    "BB":  (-1.00,  0.00),  # upper armhole (180°)
    "O":   (-0.77, -0.64),  # side seam top (220°)
    "FF":  (-0.64,  0.77),  # side seam bottom (130°)
    "XX":  (-0.71,  0.71),  # back dart left base (135°)
    "YY":  ( 0.71,  0.71),  # back dart right base (45°)
    "ZZ":  ( 0.45,  0.00),  # dart tip → right (already optimal)
}

_FRONT_LABEL_OFFSETS = {
    "M":   (-0.71, -0.71),  # CF neck corner (225°)
    "D":   (-0.45,  0.38),  # CF waist corner (already optimal)
    "K":   (-0.57, -0.82),  # front neck curve (235°)
    "N":   ( 0.91, -0.42),  # shoulder tip (335°)
    "P":   ( 1.00,  0.00),  # lower armhole transition (0°)
    "O":   ( 0.71, -0.71),  # side seam top (315°)
    "Q":   ( 0.64,  0.77),  # side seam base (50°)
    "V":   ( 0.87, -0.50),  # bust dart lower (330°)
    "T":   ( 0.87,  0.50),  # bust dart upper (30°)
    "UU":  ( 0.64,  0.77),  # bust dart upper base (50°)
    "VV":  (-0.64,  0.77),  # waist dart left base (130°)
    "WW":  ( 0.64,  0.77),  # waist dart right base (50°)
    "W":   (-0.45,  0.00),  # waist dart tip (already optimal)
    "S":   ( 0.40,  0.00),  # bust point (already optimal)
}

_INTERIOR_LABEL_OFFSETS = {
    # Construction-rectangle reference points. Offsets chosen so text lands
    # inside whichever piece the dot appears in.
    "B":   ( 0.40,  0.35),  # bottom-left corner → right and up
    "C":   (-0.40, -0.35),  # top-right corner → left and down
    "E":   (-0.35, -0.40),  # top of vertical centerline → left and down
    "F":   ( 0.35,  0.40),  # bottom of vertical centerline → right and up
    "G":   ( 0.40,  0.00),  # left of upper horizontal → right
    "H":   (-0.40,  0.00),  # right of upper horizontal → left
    "I":   ( 0.40,  0.00),  # left of middle horizontal → right
    "J":   (-0.40,  0.00),  # right of middle horizontal → left
    "R":   ( 0.40, -0.25),  # shoulder midpoint → right and slightly down
    "U":   ( 0.40,  0.00),  # bust dart apex → right
    "EE":  ( 0.40,  0.35),  # waist curve reference → right and up
    "CC":  (-0.40, -0.35),  # armhole reference (above BB) → left and down
}


# ── Label-tightening helpers ─────────────────────────────────────────────────

def _pip(pt, poly):
    """Ray-casting point-in-polygon test. poly is Nx2 array."""
    x, y   = float(pt[0]), float(pt[1])
    inside = False
    n      = len(poly)
    px, py = float(poly[-1, 0]), float(poly[-1, 1])
    for i in range(n):
        cx, cy = float(poly[i, 0]), float(poly[i, 1])
        if ((cy > y) != (py > y)) and x < (px - cx) * (y - cy) / (py - cy) + cx:
            inside = not inside
        px, py = cx, cy
    return inside


def _seg_dist(pt, p0, p1):
    """Distance from pt to segment p0→p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    pt     = np.asarray(pt,    float)
    v      = p1 - p0
    l2     = float(np.dot(v, v))
    if l2 < 1e-12:
        return float(np.linalg.norm(pt - p0))
    t = max(0.0, min(1.0, float(np.dot(pt - p0, v)) / l2))
    return float(np.linalg.norm(pt - (p0 + t * v)))


def _poly_dist(pt, poly):
    """Minimum distance from pt to the polygon boundary (Nx2)."""
    n = len(poly)
    return min(_seg_dist(pt, poly[i], poly[(i + 1) % n]) for i in range(n))


def _tighten_offset(pt, dx_in, dy_in, poly, name=""):
    """Shrink the offset to the minimum magnitude (same direction) where the
    text centre is inside the polygon and clear of the boundary by the text
    bounding-box half-diagonal.
    poly: Nx2 model-space polygon (from _sample_outline).
    """
    mag = float(np.hypot(dx_in, dy_in))
    if mag < 1e-8:
        return dx_in, dy_in
    udx, udy = dx_in / mag, dy_in / mag
    # Clearance = text bounding-box half-diagonal + 2 px gap
    nchars    = max(len(name), 1)
    text_w_px = FONT_SIZE * 0.6 * nchars
    text_h_px = FONT_SIZE
    clearance = float(np.hypot(text_w_px / 2, text_h_px / 2)) / SCALE + 2.0 / SCALE
    lo, hi = clearance, mag
    for _ in range(24):
        m    = (lo + hi) * 0.5
        test = np.array([float(pt[0]) + udx * m, float(pt[1]) + udy * m])
        if _pip(test, poly) and _poly_dist(test, poly) >= clearance:
            hi = m   # can get closer
        else:
            lo = m   # need more distance
    return udx * hi, udy * hi


def _seam_runs(segments):
    """Find consecutive runs of seam-eligible edges in a closed outline.

    Only plain "line" edges (not "dart", "quadratic", or "cubic_curve") receive
    seam allowance.  Darts, neck curves, and armhole curves are treated as
    no-seam gaps.  The outline is traversed as a closed polygon; runs that
    cross the wrap-around join (e.g. back waist XX→GG→A) are detected and
    returned as a single run.

    Returns a list of open polylines (each an np.array of model-space points).
    """
    # Build flat (is_seam, p0, p1) edge list — curves collapsed to endpoints
    flat = []
    for seg in segments:
        t = seg[0]
        if t == "line":
            flat.append((True,  np.asarray(seg[1], float), np.asarray(seg[2], float)))
        elif t == "dart":
            flat.append((False, np.asarray(seg[1], float), np.asarray(seg[2], float)))
        elif t == "quadratic":
            flat.append((False, np.asarray(seg[1], float), np.asarray(seg[3], float)))
        elif t == "cubic_curve":
            _, func, p0r, p1r = seg
            flat.append((False, np.asarray(p0r, float), np.asarray(p1r, float)))

    n = len(flat)
    if not n:
        return []

    # Find first no-seam edge; start traversal after it to avoid splitting
    # a run that wraps around the polygon boundary
    start = 0
    for j in range(n):
        if not flat[j][0]:
            start = (j + 1) % n
            break
    else:
        # Every edge is a seam — whole outline is one run
        pts = [flat[0][1]] + [e[2] for e in flat]
        return [np.array(pts)]

    runs, current = [], []
    for step in range(n):
        is_seam, p0, p1 = flat[(start + step) % n]
        if is_seam:
            if not current:
                current = [p0.copy()]
            current.append(p1.copy())
        else:
            if current:
                runs.append(np.array(current))
                current = []
    if current:
        runs.append(np.array(current))
    return runs


def _seam_runs_no_waist(segments, seam_allowance, seam_allowance_fn):
    """Like _seam_runs but omits bottom-horizontal (waist) edges.
    The back waist (FF→YY, XX→GG) and front waist (D→WW, VV→Q) lines receive
    no seam allowance — they are the hem/cut edge.

    Returns list of (run, sa) pairs: run is np.array of model-space points,
    sa is the float seam allowance for that run (already > 0).
    """
    pts_bbox = _sample_bbox(segments)
    min_y    = float(pts_bbox[:, 1].min())
    WAIST_DY   = 0.05   # max y-variation for a "horizontal" edge (inches)
    WAIST_BAND = 0.25   # waist must be within 0.25" of the outline minimum y

    result = []
    for run in _seam_runs(segments):
        n         = len(run)
        sub_start = 0
        for i in range(n - 1):
            p0y = float(run[i,     1])
            p1y = float(run[i + 1, 1])
            is_waist = (abs(p1y - p0y) < WAIST_DY and
                        max(p0y, p1y)  < min_y + WAIST_BAND)
            if is_waist:
                if i > sub_start:          # flush sub-run before waist edge
                    sub = run[sub_start : i + 1]
                    sa  = seam_allowance_fn(sub) if seam_allowance_fn else seam_allowance
                    if sa > 1e-6:
                        result.append((sub, sa))
                sub_start = i + 1          # resume after waist edge
        # tail
        if n - 1 > sub_start:
            sub = run[sub_start:]
            sa  = seam_allowance_fn(sub) if seam_allowance_fn else seam_allowance
            if sa > 1e-6:
                result.append((sub, sa))
    return result


def _offset_open_polyline(pts, distance, centroid):
    """Outward parallel offset of an open polyline.

    Interior vertices use a miter bisector (capped at MITER_LIMIT).
    Endpoint vertices use the perpendicular of their single adjacent edge.
    Returns an np.array of offset points, same count as *pts*.
    """
    centroid = np.asarray(centroid, float)
    n = len(pts)
    MITER_LIMIT = 4.0

    def _out_normal(ev, le, mid):
        nv = np.array([-ev[1] / le, ev[0] / le])
        if np.linalg.norm(mid + nv - centroid) < np.linalg.norm(mid - centroid):
            nv = -nv
        return nv

    offset = []
    for i in range(n):
        p = pts[i]
        if i == 0:
            e = pts[1] - pts[0]; le = np.linalg.norm(e)
            offset.append(p + _out_normal(e, le, (pts[0]+pts[1])/2) * distance
                          if le > 1e-8 else p.copy())
        elif i == n - 1:
            e = pts[-1] - pts[-2]; le = np.linalg.norm(e)
            offset.append(p + _out_normal(e, le, (pts[-2]+pts[-1])/2) * distance
                          if le > 1e-8 else p.copy())
        else:
            e_in = pts[i] - pts[i-1]; li = np.linalg.norm(e_in)
            e_out = pts[i+1] - pts[i]; lo = np.linalg.norm(e_out)
            if li < 1e-8 or lo < 1e-8:
                v = p - centroid; vl = np.linalg.norm(v)
                offset.append(p + (v/vl if vl > 1e-8 else np.array([1.,0.])) * distance)
                continue
            ni = _out_normal(e_in,  li, (pts[i-1]+pts[i])/2)
            no = _out_normal(e_out, lo, (pts[i]+pts[i+1])/2)
            bis = ni + no; bl = np.linalg.norm(bis)
            if bl < 1e-8:
                offset.append(p + ni * distance)
                continue
            bis /= bl
            sin_h = max(np.dot(bis, ni), 1.0 / MITER_LIMIT)
            offset.append(p + bis * (distance / sin_h))
    return np.array(offset)


def _mirror_point(pt, fold_line_x):
    """Mirror a point across a vertical fold line at x = fold_line_x."""
    pt = np.asarray(pt, float)
    mirrored = pt.copy()
    mirrored[0] = 2 * fold_line_x - pt[0]
    return mirrored


def _mirror_segment(seg, fold_line_x):
    """Mirror a segment across a vertical fold line. Returns mirrored segment."""
    if seg[0] == "line":
        _, p0, p1 = seg
        return ("line", _mirror_point(p1, fold_line_x), _mirror_point(p0, fold_line_x))
    elif seg[0] == "dart":
        _, p0, p1 = seg
        return ("dart", _mirror_point(p1, fold_line_x), _mirror_point(p0, fold_line_x))
    elif seg[0] == "quadratic":
        _, p0, cp, p3 = seg
        return ("quadratic", _mirror_point(p3, fold_line_x), _mirror_point(cp, fold_line_x), _mirror_point(p0, fold_line_x))
    elif seg[0] == "cubic_curve":
        _, func, p0, p1 = seg
        # Create a mirrored function
        def mirrored_func(t):
            pt = func(1 - t)  # reverse parameterization
            return _mirror_point(pt, fold_line_x)
        return ("cubic_curve", mirrored_func, _mirror_point(p1, fold_line_x), _mirror_point(p0, fold_line_x))
    else:
        return seg


def _fold_front_bodice(front_bodice, fold_line_x):
    """Create a full-width front bodice by folding/mirroring.
    Assumes front_bodice goes from fold line to edge. Returns full-width outline."""
    # Mirror all segments and reverse their order
    mirrored = [_mirror_segment(seg, fold_line_x) for seg in reversed(front_bodice)]
    # Remove the last segment of mirrored (which is the M-D fold line)
    # and first segment of original (which is also M-D)
    return mirrored[:-1] + front_bodice[1:]


def _sample_bbox(segments):
    """Return points sufficient to compute a bounding box (endpoints + ctrl pt)."""
    pts = []
    for seg in segments:
        if seg[0] in ("line", "dart"):
            pts += [seg[1], seg[2]]
        elif seg[0] == "quadratic":  # quadratic: (type, P0, CP, P3)
            pts += [seg[1], seg[2], seg[3]]
        elif seg[0] == "cubic_curve":  # cubic_curve: (type, func, P0, P1)
            # Sample the curve to get representative points for bbox
            _, func, p0, p1 = seg
            pts += [p0, p1]
            # Add midpoint sample
            mid = func(0.5)
            pts.append(mid)
    return np.array(pts)


def _make_converter(all_pts):
    min_x = all_pts[:, 0].min()
    max_x = all_pts[:, 0].max()
    min_y = all_pts[:, 1].min()
    max_y = all_pts[:, 1].max()
    content_w = max_x - min_x   # inches
    content_h = max_y - min_y   # inches
    # Smallest integer-inch canvas containing content + 0.5" margin on each side
    w_in = int(np.ceil(content_w + 2 * MARGIN_INCHES))
    h_in = int(np.ceil(content_h + 2 * MARGIN_INCHES))
    # Center content within the integer canvas
    pad_x = (w_in - content_w) / 2   # inches
    pad_y = (h_in - content_h) / 2   # inches
    w = w_in * SCALE
    h = h_in * SCALE

    def convert(pts):
        pts = np.atleast_2d(pts)
        return np.column_stack([
            (pts[:, 0] - min_x + pad_x) * SCALE,
            (max_y - pts[:, 1] + pad_y) * SCALE,
        ])

    return convert, w, h, w_in, h_in


def _outline_to_svg_path(segments, convert):
    """Full closed path (fill + clip). Dart and curve segments treated as lines/polylines."""
    parts = []
    for idx, seg in enumerate(segments):
        if seg[0] in ("line", "dart"):
            p0 = convert(seg[1])[0]
            p1 = convert(seg[2])[0]
            if idx == 0:
                parts.append(f"M {p0[0]:.2f},{p0[1]:.2f}")
            parts.append(f"L {p1[0]:.2f},{p1[1]:.2f}")
        elif seg[0] == "quadratic":  # quadratic: (type, P0, CP, P3)
            _, p0, cp, p3 = seg
            s0  = convert(p0)[0]
            sc  = convert(cp)[0]
            s3  = convert(p3)[0]
            if idx == 0:
                parts.append(f"M {s0[0]:.2f},{s0[1]:.2f}")
            parts.append(f"Q {sc[0]:.2f},{sc[1]:.2f} {s3[0]:.2f},{s3[1]:.2f}")
        elif seg[0] == "cubic_curve":  # cubic_curve: (type, func, P0, P1)
            # Sample the curve and create a polyline approximation
            _, func, p0, p1 = seg
            ts = np.linspace(0, 1, 40)  # 40-point sample for smooth curve
            pts_model = np.array([func(t) for t in ts])
            pts_svg = convert(pts_model)
            if idx == 0:
                parts.append(f"M {pts_svg[0, 0]:.2f},{pts_svg[0, 1]:.2f}")
            for pt in pts_svg[1:]:
                parts.append(f"L {pt[0]:.2f},{pt[1]:.2f}")
    parts.append("Z")
    return " ".join(parts)


def _outline_stroke_paths(segments, convert):
    """Return (seam_d, dart_d) — separate path strings for stroke rendering."""
    seam, dart = [], []
    for seg in segments:
        if seg[0] in ("line", "dart"):
            p0 = convert(seg[1])[0]
            p1 = convert(seg[2])[0]
            chunk = f"M {p0[0]:.2f},{p0[1]:.2f} L {p1[0]:.2f},{p1[1]:.2f}"
            (dart if seg[0] == "dart" else seam).append(chunk)
        elif seg[0] == "quadratic":  # quadratic → always a seam
            _, p0, cp, p3 = seg
            s0, sc, s3 = convert(p0)[0], convert(cp)[0], convert(p3)[0]
            seam.append(f"M {s0[0]:.2f},{s0[1]:.2f}"
                        f" Q {sc[0]:.2f},{sc[1]:.2f} {s3[0]:.2f},{s3[1]:.2f}")
        elif seg[0] == "cubic_curve":  # cubic_curve → seam (sample curve)
            _, func, p0, p1 = seg
            ts = np.linspace(0, 1, 40)
            pts_model = np.array([func(t) for t in ts])
            pts_svg = convert(pts_model)
            path = f"M {pts_svg[0, 0]:.2f},{pts_svg[0, 1]:.2f}"
            for pt in pts_svg[1:]:
                path += f" L {pt[0]:.2f},{pt[1]:.2f}"
            seam.append(path)
    return " ".join(seam), " ".join(dart)


def _sample_outline(segments, n=80):
    """Dense point sample — used to compute the centroid."""
    pts = []
    for idx, seg in enumerate(segments):
        if seg[0] in ("line", "dart"):
            if idx == 0:
                pts.append(seg[1])
            pts.append(seg[2])
        elif seg[0] == "quadratic":  # quadratic: (type, P0, CP, P3)
            _, p0, cp, p3 = seg
            ts  = np.linspace(0, 1, n)[:, None]
            smp = (1-ts)**2 * p0 + 2*(1-ts)*ts * cp + ts**2 * p3
            pts.extend(smp if idx == 0 else smp[1:])
        elif seg[0] == "cubic_curve":  # cubic_curve: (type, func, P0, P1)
            _, func, p0, p1 = seg
            ts = np.linspace(0, 1, n)
            smp = np.array([func(t) for t in ts])
            pts.extend(smp if idx == 0 else smp[1:])
    return np.array(pts)


def _inward_dir(pt, outline, centroid_model):
    """Return a unit vector (model space) pointing inward from pt.

    For outline vertices: angle bisector of the two adjacent edge directions,
    verified to point toward the interior.  For bezier endpoints the adjacent
    control point is used so the bisector is tangent-aware.
    Falls back to the centroid direction for interior / unmatched points.
    """
    pt = np.asarray(pt, float)
    prev_n = next_n = None

    for seg in outline:
        if seg[0] in ("line", "dart"):
            p0, p1 = np.asarray(seg[1], float), np.asarray(seg[2], float)
            if np.allclose(p0, pt, atol=1e-4): next_n = p1
            if np.allclose(p1, pt, atol=1e-4): prev_n = p0
        elif seg[0] == "quadratic":
            p0, cp, p3 = (np.asarray(seg[i], float) for i in range(1, 4))
            if np.allclose(p0, pt, atol=1e-4): next_n = cp    # tangent leaves toward cp
            if np.allclose(p3, pt, atol=1e-4): prev_n = cp    # tangent arrived from cp
        elif seg[0] == "cubic_curve":
            _, func, p0, p1 = seg
            p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
            if np.allclose(p0, pt, atol=1e-4): 
                # Tangent at start: direction from p0 toward nearby point on curve
                next_pt = func(0.1)
                next_n = np.asarray(next_pt, float)
            if np.allclose(p1, pt, atol=1e-4):
                # Tangent at end: direction from nearby point back to p1
                prev_pt = func(0.9)
                prev_n = np.asarray(prev_pt, float)

    c = np.asarray(centroid_model, float)

    # Centroid direction — always reliably inward
    diff = c - pt
    cdn = np.linalg.norm(diff)
    centroid_dir = diff / cdn if cdn > 1e-8 else np.array([1.0, 0.0])

    if prev_n is not None and next_n is not None:
        v1 = np.asarray(prev_n, float) - pt
        v2 = np.asarray(next_n, float) - pt
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-8 and n2 > 1e-8:
            b = v1 / n1 + v2 / n2
            nb = np.linalg.norm(b)
            if nb > 1e-8:
                b /= nb
                if np.dot(b, c - pt) < 0:
                    b = -b
                # Always blend 35% bisector + 65% centroid direction so that
                # concave corners (like O and V) are pushed reliably inward.
                b = 0.35 * b + 0.65 * centroid_dir
                b /= np.linalg.norm(b)
                return b

    return centroid_dir


def _label_elements(convert, outline, centroid_model, name, pt, filled,
                    label_offsets=None, outline_poly=None):
    """Return (dot_lines, text_lines) for one labeled point.
    Dots go inside the clip group; text goes outside so it is never cut off.
    filled=True  → solid black dot, dark text  (outline point)
    filled=False → hollow circle, gray text    (interior / construction point)
    outline_poly: Nx2 model-space polygon; when supplied, offsets for outline
                  labels are tightened to the minimum safe clearance.
    """
    sx, sy = convert(np.array([pt]))[0]

    # Resolve offset: use hardcoded table when available, else centroid direction.
    base   = name.rstrip("'")
    primes = len(name) - len(base)
    if label_offsets is not None and base in label_offsets:
        dx_in, dy_in = label_offsets[base]
        if primes % 2 == 1:
            dx_in = -dx_in          # mirror x for fold-primed labels
        if outline_poly is not None and filled:
            dx_in, dy_in = _tighten_offset(
                np.asarray(pt, float), dx_in, dy_in, outline_poly, name)
        dx =  dx_in * SCALE
        dy = -dy_in * SCALE         # model y-up → SVG y-down
    else:
        d = _inward_dir(pt, outline, centroid_model)
        dx, dy = d[0] * LABEL_OFFSET, -d[1] * LABEL_OFFSET

    if filled:
        dot   = f'    <circle cx="{sx:.1f}" cy="{sy:.1f}" r="8" fill="black"/>'
        color = "#222"
    else:
        dot   = (f'    <circle cx="{sx:.1f}" cy="{sy:.1f}" r="9"'
                 f'            fill="white" stroke="#888" stroke-width="3"/>')
        color = "#888"

    text = (f'  <text x="{sx+dx:.1f}" y="{sy+dy:.1f}"'
            f'        font-family="{FONT_FAMILY}" font-size="{FONT_SIZE}"'
            f'        fill="{color}" font-weight="bold"'
            f'        text-anchor="middle" dominant-baseline="middle">{name}</text>')
    return [dot], [text]


def _write_svg(path, outline, construction_lines, dart_lines, fill, stroke,
               outline_labels, interior_labels, seam_allowance=0,
               seam_allowance_fn=None, label_offsets=None):
    # Compute seam allowance runs (open polylines) in model space for bbox.
    # seam_allowance_fn, if provided, takes a run (np.array of points) and returns
    # the SA for that run, allowing per-run overrides.
    # Waist (bottom horizontal) edges are always excluded from seam allowance.
    outline_poly  = _sample_outline(outline)          # dense Nx2, reused below
    centroid_temp = outline_poly.mean(axis=0)
    seam_offset_runs = []   # list of np.array (model-space offset polyline per run)
    for run, sa in _seam_runs_no_waist(outline, seam_allowance, seam_allowance_fn):
        seam_offset_runs.append(_offset_open_polyline(run, sa, centroid_temp))

    # bounding box: outline + seam offset only.
    # Construction lines span the full grid rectangle and must not inflate the canvas.
    all_pt_arrays = ([_sample_bbox(outline)] + seam_offset_runs)
    all_pts = np.vstack(all_pt_arrays)

    convert, w, h, w_in, h_in = _make_converter(all_pts)
    centroid_model = outline_poly.mean(axis=0)                    # model space
    clip_id        = "bodice-clip"
    path_d   = _outline_to_svg_path(outline, convert)
    seam_d, dart_d = _outline_stroke_paths(outline, convert)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{w:.0f}" height="{h:.0f}"',
        f'     viewBox="0 0 {w:.0f} {h:.0f}">',
        '  <rect width="100%" height="100%" fill="white"/>',
        '  <defs>',
        f'    <clipPath id="{clip_id}">',
        f'      <path d="{path_d}"/>',
        '    </clipPath>',
        '  </defs>',
    ]
    
    # seam allowance — one open dashed path per seam run
    for off_pts in seam_offset_runs:
        svg_pts = convert(off_pts)
        d_parts = [f"M {svg_pts[0,0]:.2f},{svg_pts[0,1]:.2f}"]
        for sp in svg_pts[1:]:
            d_parts.append(f"L {sp[0]:.2f},{sp[1]:.2f}")
        lines.append(f'  <path d="{" ".join(d_parts)}" fill="none" stroke="{stroke}"'
                     f'        stroke-width="5" stroke-dasharray="6 6" opacity="0.5"'
                     f'        stroke-linecap="round"/>')
    
    lines += [
        # bodice fill
        f'  <path d="{path_d}" fill="{fill}" stroke="none" opacity="0.85"/>',
        # construction lines clipped to interior
        f'  <g clip-path="url(#{clip_id})">',
    ]
    for p0, p1 in construction_lines:
        x0, y0 = convert(p0)[0]
        x1, y1 = convert(p1)[0]
        lines.append(
            f'    <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"'
            f'          stroke="#aaa" stroke-width="5" stroke-dasharray="12 9"/>'
        )
    lines.append('  </g>')

    # dart construction lines — rendered unclipped so they show inside dart notches
    lines.append('  <g>')
    for p0, p1 in dart_lines:
        x0, y0 = convert(p0)[0]
        x1, y1 = convert(p1)[0]
        lines.append(
            f'    <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"'
            f'          stroke="#bbb" stroke-width="5" stroke-dasharray="12 9"/>'
        )
    lines.append('  </g>')

    lines += [
        # seam stroke
        f'  <path d="{seam_d}" fill="none" stroke="{stroke}"'
        f'        stroke-width="5" stroke-linejoin="round"/>',
        # dart stroke in light gray
        f'  <path d="{dart_d}" fill="none" stroke="#bbb"'
        f'        stroke-width="3" stroke-linejoin="round"/>',
    ]

    # Collect dots (clipped) and text (unclipped) separately.
    # Keeping text outside the clip path prevents concave-corner labels
    # like O and V from being cut off by the bodice outline.
    all_label_items = (
        [(name, pt, False) for name, pt in interior_labels.items()] +
        [(name, pt, True)  for name, pt in outline_labels.items()]
    )
    all_dots  = []
    all_texts = []
    for name, pt, filled in all_label_items:
        dot, text = _label_elements(convert, outline, centroid_model, name, pt, filled,
                                     label_offsets, outline_poly)
        all_dots.append(dot)
        if filled:
            all_texts.append(text)

    lines.append(f'  <g clip-path="url(#{clip_id})">')
    for d in all_dots:
        lines += d
    lines.append('  </g>')
    # Text rendered after and outside the clip so it is never cut off
    for t in all_texts:
        lines += t
    lines.append('</svg>')

    svg_string = "\n".join(lines)
    if path is None:
        return svg_string, w_in, h_in
    with open(path, "w") as f:
        f.write(svg_string)
    print(f"Saved {path}  ({w_in} \u00d7 {h_in} in  /  {w:.0f} \u00d7 {h:.0f} px)")


def render_svgs(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
               fold=False, seam_allowance=0.75, deepen_bust_dart=False,
               white_fill=False):
    """Return {'front': svg_str, 'back': svg_str, ...}.  Used by the web interface."""
    bk = build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
               deepen_bust_dart=deepen_bust_dart)

    # A→GG (center back seam) must never have SA between 0 and 1 exclusive.
    center_back_sa = 0.0 if seam_allowance == 0 else max(seam_allowance, 1.0)
    a_pt = bk.A
    def _back_sa_fn(run):
        # The center-back run [XX, GG, A] ends at A
        if np.allclose(run[-1], a_pt, atol=1e-4):
            return center_back_sa
        return seam_allowance

    shared_interior = {
        "B":  bk.B,  "C":  bk.C,
        "E":  bk.E,  "F":  bk.F,  "G":  bk.G,
        "H":  bk.H,  "I":  bk.I,  "J":  bk.J,
        "R":  bk.R,
        "U":  bk.U,  "EE": bk.EE, "CC": bk.CC,
    }

    front_outline = bk.front_bodice
    front_labels = {
        "D":  bk.D,  "M":  bk.M,  "K":  bk.K,  "N":  bk.N,
        "P":  bk.P,  "O":  bk.O,  "Q":  bk.Q,
        "T":  bk.T,  "V":  bk.V,  "W":  bk.W,
        "UU": bk.UU, "VV": bk.VV, "WW": bk.WW,
        "S":  bk.S,
    }

    if fold:
        fold_line_x = bk.M[0]
        front_outline = _fold_front_bodice(bk.front_bodice, fold_line_x)
        front_labels = {
            "M":  bk.M,  "D":  bk.D,
            "K":  bk.K,  "N":  bk.N,
            "P":  bk.P,  "O":  bk.O,  "Q":  bk.Q,
            "T":  bk.T,  "V":  bk.V,  "W":  bk.W,
            "UU": bk.UU, "VV": bk.VV, "WW": bk.WW,
            "S":  bk.S,
        }
        for name, pt in list(front_labels.items()):
            pt_arr = np.asarray(pt, float)
            if abs(pt_arr[0] - fold_line_x) > 1e-4:
                front_labels[name + "'"] = _mirror_point(pt_arr, fold_line_x)

    back_svg, back_w, back_h = _write_svg(
        None,
        bk.back_bodice,
        construction_lines=bk.construction_lines,
        dart_lines=bk.back_dart_lines,
        fill="white" if white_fill else "#dce8f5", stroke="#2255aa",
        outline_labels={
            "A":  bk.A,  "GG": bk.GG, "AA": bk.AA, "DD": bk.DD,
            "BB": bk.BB, "O":  bk.O,  "FF": bk.FF,
            "XX": bk.XX, "YY": bk.YY, "ZZ": bk.ZZ,
        },
        interior_labels=shared_interior,
        seam_allowance=seam_allowance,
        seam_allowance_fn=_back_sa_fn,
        label_offsets={**_INTERIOR_LABEL_OFFSETS, **_BACK_LABEL_OFFSETS},
    )

    front_svg, front_w, front_h = _write_svg(
        None,
        front_outline,
        construction_lines=bk.construction_lines,
        dart_lines=bk.front_dart_lines,
        fill="white" if white_fill else "#dac7ff", stroke="#7a6f8a",
        outline_labels=front_labels,
        interior_labels=shared_interior,
        seam_allowance=seam_allowance,
        label_offsets={**_INTERIOR_LABEL_OFFSETS, **_FRONT_LABEL_OFFSETS},
    )

    return {
        'front': front_svg, 'back': back_svg,
        'front_w': front_w, 'front_h': front_h,
        'back_w': back_w, 'back_h': back_h,
    }


def render(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
           prefix="bodice", fold=False, seam_allowance=0.75, deepen_bust_dart=False):
    """Render bodice blocks to SVG.
    
    Args:
        fold: If True, mirror front bodice on the M-D line to show full width
        seam_allowance: Seam allowance in inches (default 0.75). The A→GG
                       (center back) seam receives max(seam_allowance, 1.0),
                       except when seam_allowance is exactly 0.
        deepen_bust_dart: If True, add 0.5" to the bust dart depth from Chart 1.
    """
    bk = build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
               deepen_bust_dart=deepen_bust_dart)
    
    # A→GG (center back seam) must never have SA between 0 and 1 exclusive.
    center_back_sa = 0.0 if seam_allowance == 0 else max(seam_allowance, 1.0)
    a_pt = bk.A
    def _back_sa_fn(run):
        if np.allclose(run[-1], a_pt, atol=1e-4):
            return center_back_sa
        return seam_allowance

    # Points shared by both views (construction rectangle corners + reference pts)
    shared_interior = {
        "B":  bk.B,  "C":  bk.C,
        "E":  bk.E,  "F":  bk.F,  "G":  bk.G,
        "H":  bk.H,  "I":  bk.I,  "J":  bk.J,
        "R":  bk.R,
        "U":  bk.U,  "EE": bk.EE, "CC": bk.CC,
    }
    
    # Determine front bodice outline
    front_outline = bk.front_bodice
    front_labels = {
        "D":  bk.D,  "M":  bk.M,  "K":  bk.K,  "N":  bk.N,
        "P":  bk.P,  "O":  bk.O,  "Q":  bk.Q,
        "T":  bk.T,  "V":  bk.V,  "W":  bk.W,
        "UU": bk.UU, "VV": bk.VV, "WW": bk.WW,
        "S":  bk.S,
    }
    
    if fold:
        fold_line_x = bk.M[0]  # = D[0] = center-front x
        front_outline = _fold_front_bodice(bk.front_bodice, fold_line_x)
        front_labels = {
            "M":  bk.M,  "D":  bk.D,
            "K":  bk.K,  "N":  bk.N,
            "P":  bk.P,  "O":  bk.O,  "Q":  bk.Q,
            "T":  bk.T,  "V":  bk.V,  "W":  bk.W,
            "UU": bk.UU, "VV": bk.VV, "WW": bk.WW,
            "S":  bk.S,
        }
        # Add primed mirrored labels for every point not sitting on the fold line
        for name, pt in list(front_labels.items()):
            pt_arr = np.asarray(pt, float)
            if abs(pt_arr[0] - fold_line_x) > 1e-4:
                front_labels[name + "'"] = _mirror_point(pt_arr, fold_line_x)

    _write_svg(
        f"{prefix}_back.svg",
        bk.back_bodice,
        construction_lines=bk.construction_lines,
        dart_lines=bk.back_dart_lines,
        fill="#dce8f5", stroke="#2255aa",
        outline_labels={
            "A":  bk.A,  "GG": bk.GG, "AA": bk.AA, "DD": bk.DD,
            "BB": bk.BB, "O":  bk.O,  "FF": bk.FF,
            "XX": bk.XX, "YY": bk.YY, "ZZ": bk.ZZ,
        },
        interior_labels=shared_interior,
        seam_allowance=seam_allowance,
        seam_allowance_fn=_back_sa_fn,
        label_offsets={**_INTERIOR_LABEL_OFFSETS, **_BACK_LABEL_OFFSETS},
    )

    _write_svg(
        f"{prefix}_front.svg",
        front_outline,
        construction_lines=bk.construction_lines,
        dart_lines=bk.front_dart_lines,
        fill="#dac7ff", stroke="#7a6f8a",
        outline_labels=front_labels,
        interior_labels=shared_interior,
        seam_allowance=seam_allowance,
        label_offsets={**_INTERIOR_LABEL_OFFSETS, **_FRONT_LABEL_OFFSETS},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a bodice block to SVG.")
    parser.add_argument("--alpha",   type=float, required=True, help="waist")
    parser.add_argument("--beta",    type=float, required=True, help="bust")
    parser.add_argument("--gamma",   type=float, required=True, help="back nape to waist")
    parser.add_argument("--delta",   type=float, required=True, help="neck to shoulder")
    parser.add_argument("--epsilon", type=float, required=True, help="shoulder center to bust point")
    parser.add_argument("--zeta",    type=float, required=True, help="bust point to bust point")
    parser.add_argument("--eta",     type=float, required=True, help="front width")
    parser.add_argument("--theta",   type=float, required=True, help="back width")
    parser.add_argument("--prefix",  type=str,   default="bodice", help="output filename prefix")
    parser.add_argument("--fold",    action="store_true", help="render front bodice on fold (mirrored, full width)")
    parser.add_argument("--seam-allowance", type=float, default=0.5, help="seam allowance in inches (default 0.75)")
    args = parser.parse_args()

    render(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        delta=args.delta, epsilon=args.epsilon, zeta=args.zeta,
        eta=args.eta, theta=args.theta, prefix=args.prefix,
        fold=args.fold, seam_allowance=args.seam_allowance,
    )
