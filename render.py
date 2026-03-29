import argparse
import numpy as np
from block import build

SCALE        = 96   # px per inch
MARGIN       = 60   # px
LABEL_OFFSET = 20   # px — inward push along bisector
FONT_SIZE    = 11   # px


def _offset_outline_outward(segments, distance, centroid):
    """Create an outward-offset outline for seam allowance.
    Simplistic approach: offset line endpoints perpendicular to adjacent edges.
    For curves, sample and offset the samples."""
    if distance < 1e-6:
        return segments  # no offset needed
    
    centroid = np.asarray(centroid, float)
    offset_segs = []
    
    for idx, seg in enumerate(segments):
        if seg[0] == "line":
            _, p0, p1 = seg
            p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
            
            # Get perpendicular direction pointing outward
            edge_dir = p1 - p0
            perp = np.array([-edge_dir[1], edge_dir[0]])
            perp = perp / np.linalg.norm(perp)
            
            # Check which direction points outward from centroid
            edge_mid = (p0 + p1) / 2
            outward_test = edge_mid + perp
            if np.linalg.norm(outward_test - centroid) > np.linalg.norm(edge_mid - centroid):
                # perp points outward
                offset_p0 = p0 + perp * distance
                offset_p1 = p1 + perp * distance
            else:
                # -perp points outward
                offset_p0 = p0 - perp * distance
                offset_p1 = p1 - perp * distance
            
            offset_segs.append(("line", offset_p0, offset_p1))
        
        elif seg[0] == "dart":
            _, p0, p1 = seg
            p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
            
            # Darts typically don't get seam allowance, but we'll offset anyway
            edge_dir = p1 - p0
            perp = np.array([-edge_dir[1], edge_dir[0]])
            perp = perp / np.linalg.norm(perp)
            
            edge_mid = (p0 + p1) / 2
            if np.linalg.norm(edge_mid + perp - centroid) > np.linalg.norm(edge_mid - centroid):
                offset_p0 = p0 + perp * distance
                offset_p1 = p1 + perp * distance
            else:
                offset_p0 = p0 - perp * distance
                offset_p1 = p1 - perp * distance
            
            offset_segs.append(("dart", offset_p0, offset_p1))
        
        elif seg[0] == "quadratic":
            _, p0, cp, p3 = seg
            # For quadratic, we'll sample and create a polyline
            ts = np.linspace(0, 1, 15)
            pts_model = np.array([(1-t)**2 * p0 + 2*(1-t)*t * cp + t**2 * p3 for t in ts])
            
            # Offset each point outward (simplified: use centroid direction)
            offset_pts = []
            for pt in pts_model:
                outdir = pt - centroid
                outdir = outdir / np.linalg.norm(outdir) if np.linalg.norm(outdir) > 1e-8 else np.array([1.0, 0.0])
                offset_pts.append(pt + outdir * distance)
            
            # Convert to line segments
            for i in range(len(offset_pts) - 1):
                offset_segs.append(("line", offset_pts[i], offset_pts[i+1]))
        
        elif seg[0] == "cubic_curve":
            _, func, p0, p1 = seg
            ts = np.linspace(0, 1, 40)
            pts_model = np.array([func(t) for t in ts])
            
            offset_pts = []
            for i, pt in enumerate(pts_model):
                # Calculate local tangent
                if i == 0:
                    tangent = pts_model[1] - pts_model[0]
                elif i == len(pts_model) - 1:
                    tangent = pts_model[-1] - pts_model[-2]
                else:
                    tangent = pts_model[i+1] - pts_model[i-1]
                
                tangent_len = np.linalg.norm(tangent)
                if tangent_len > 1e-8:
                    tangent = tangent / tangent_len
                    # Get perpendicular pointing outward
                    perp = np.array([-tangent[1], tangent[0]])
                    # Test orientation vs centroid
                    outward_test = pt + perp
                    if np.linalg.norm(outward_test - centroid) < np.linalg.norm(pt - centroid):
                        perp = -perp
                    offset_pts.append(pt + perp * distance)
                else:
                    # fallback to centroid logic
                    outdir = pt - centroid
                    outdir = outdir / np.linalg.norm(outdir) if np.linalg.norm(outdir) > 1e-8 else np.array([1.0, 0.0])
                    offset_pts.append(pt + outdir * distance)
            
            for i in range(len(offset_pts) - 1):
                offset_segs.append(("line", offset_pts[i], offset_pts[i+1]))
    
    return offset_segs


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
    min_x  = all_pts[:, 0].min()
    max_y  = all_pts[:, 1].max()
    width  = (all_pts[:, 0].max() - min_x) * SCALE + 2 * MARGIN
    height = (max_y - all_pts[:, 1].min()) * SCALE + 2 * MARGIN

    def convert(pts):
        pts = np.atleast_2d(pts)
        return np.column_stack([
            (pts[:, 0] - min_x) * SCALE + MARGIN,
            (max_y    - pts[:, 1]) * SCALE + MARGIN,
        ])

    return convert, width, height


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
                # Ensure minimum inward component: if point is near vertical edge,
                # boost horizontal component toward centroid
                centroid_dir = c - pt
                if np.linalg.norm(centroid_dir) > 1e-8:
                    centroid_dir = centroid_dir / np.linalg.norm(centroid_dir)
                    # Blend with centroid direction if inward component is weak
                    inward_component = np.dot(b, centroid_dir)
                    if inward_component < 0.3:  # weak inward pointing
                        b = 0.5 * b + 0.5 * centroid_dir
                        b = b / np.linalg.norm(b)
                return b

    # fallback: centroid direction
    diff = c - pt
    n = np.linalg.norm(diff)
    return diff / n if n > 1e-8 else np.array([1.0, 0.0])


def _label_elements(convert, outline, centroid_model, name, pt, filled):
    """Return (dot_lines, text_lines) for one labeled point.
    Dots go inside the clip group; text goes outside so it is never cut off.
    filled=True  → solid black dot, dark text  (outline point)
    filled=False → hollow circle, gray text    (interior / construction point)
    """
    sx, sy = convert(np.array([pt]))[0]

    d = _inward_dir(pt, outline, centroid_model)
    # model space is y-up; SVG is y-down — flip the y component
    dx, dy = d[0] * LABEL_OFFSET, -d[1] * LABEL_OFFSET

    if filled:
        dot   = f'    <circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.5" fill="black"/>'
        color = "#222"
    else:
        dot   = (f'    <circle cx="{sx:.1f}" cy="{sy:.1f}" r="3"'
                 f'            fill="white" stroke="#888" stroke-width="1"/>')
        color = "#888"

    text = (f'  <text x="{sx+dx:.1f}" y="{sy+dy:.1f}"'
            f'        font-family="monospace" font-size="{FONT_SIZE}"'
            f'        fill="{color}" font-weight="bold"'
            f'        text-anchor="middle" dominant-baseline="middle">{name}</text>')
    return [dot], [text]


def _write_svg(path, outline, construction_lines, dart_lines, fill, stroke,
               outline_labels, interior_labels, seam_allowance=0):
    # Create seam allowance outline if specified
    seam_outline = None
    if seam_allowance > 0:
        centroid_temp = _sample_outline(outline).mean(axis=0)
        seam_outline = _offset_outline_outward(outline, seam_allowance, centroid_temp)
    
    # bounding box: outline control pts + seam allowance + construction lines + dart lines + all label pts
    bbox_segs = [outline]
    if seam_outline:
        bbox_segs.append(seam_outline)
    
    extra = (  [p for seg in construction_lines for p in seg]
             + [p for seg in dart_lines         for p in seg]
             + list(outline_labels.values())
             + list(interior_labels.values()) )
    
    all_pts = np.vstack(
        [_sample_bbox(seg_list) for seg_list in bbox_segs] + 
        [np.atleast_2d(p) for p in extra]
    )

    convert, w, h  = _make_converter(all_pts)
    centroid_model = _sample_outline(outline).mean(axis=0)        # model space
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
    
    # seam allowance (if present) — drawn with dashed outline
    if seam_outline:
        seam_allow_path_d = _outline_to_svg_path(seam_outline, convert)
        seam_allow_d, _ = _outline_stroke_paths(seam_outline, convert)
        lines.append(f'  <path d="{seam_allow_d}" fill="none" stroke="{stroke}"'
                     f'        stroke-width="0.75" stroke-dasharray="2 2" opacity="0.5"/>')
    
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
            f'          stroke="#aaa" stroke-width="0.75" stroke-dasharray="4 3"/>'
        )
    lines.append('  </g>')

    # dart construction lines — rendered unclipped so they show inside dart notches
    lines.append('  <g>')
    for p0, p1 in dart_lines:
        x0, y0 = convert(p0)[0]
        x1, y1 = convert(p1)[0]
        lines.append(
            f'    <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"'
            f'          stroke="#bbb" stroke-width="0.75" stroke-dasharray="4 3"/>'
        )
    lines.append('  </g>')

    lines += [
        # seam stroke
        f'  <path d="{seam_d}" fill="none" stroke="{stroke}"'
        f'        stroke-width="1.5" stroke-linejoin="round"/>',
        # dart stroke in light gray
        f'  <path d="{dart_d}" fill="none" stroke="#bbb"'
        f'        stroke-width="1" stroke-linejoin="round"/>',
    ]

    # dots clipped to bodice interior, text rendered outside clip so it's never cut off
    all_label_items = (
        [(name, pt, False) for name, pt in interior_labels.items()] +
        [(name, pt, True)  for name, pt in outline_labels.items()]
    )

    lines.append(f'  <g clip-path="url(#{clip_id})">')
    text_lines = []
    for name, pt, filled in all_label_items:
        dot, text = _label_elements(convert, outline, centroid_model, name, pt, filled)
        lines += dot
        if filled:
            text_lines += text
    lines.append('  </g>')

    # text on top, unclipped
    lines += text_lines
    lines.append('</svg>')

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}  ({w:.0f} × {h:.0f} px)")


def render(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
           prefix="bodice", fold=False, seam_allowance=0.75):
    """Render bodice blocks to SVG.
    
    Args:
        fold: If True, mirror front bodice on the M-D line to show full width
        seam_allowance: Seam allowance in inches (default 0.75). Special handling:
                       if the center-back seam (A-B line) is shorter than 1 inch,
                       seam_allowance is set to 0 for that edge to avoid bunching.
    """
    bk = build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta)
    
    # Check center-back seam length (A to B)
    ab_length = np.linalg.norm(bk.B - bk.A)
    center_back_seam_allow = seam_allowance if ab_length >= 1.0 else 0
    
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
        # Mirror front bodice on the M-D line (fold line at x = c + 0.5, which is M[0] = D[0])
        front_outline = _fold_front_bodice(bk.front_bodice, bk.M[0])
        # Adjust labels for folded view: remove center points that don't appear in mirror
        front_labels = {
            "M":  bk.M,  "D":  bk.D,
            "K":  bk.K,  "N":  bk.N,
            "P":  bk.P,  "O":  bk.O,  "Q":  bk.Q,
            "T":  bk.T,  "V":  bk.V,  "W":  bk.W,
            "UU": bk.UU, "VV": bk.VV, "WW": bk.WW,
            "S":  bk.S,
        }

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
        seam_allowance=center_back_seam_allow,
    )

    _write_svg(
        f"{prefix}_front.svg",
        front_outline,
        construction_lines=bk.construction_lines,
        dart_lines=bk.front_dart_lines,
        fill="#fdeede", stroke="#aa5522",
        outline_labels=front_labels,
        interior_labels=shared_interior,
        seam_allowance=seam_allowance,
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
    parser.add_argument("--seam-allowance", type=float, default=0.75, help="seam allowance in inches (default 0.75)")
    args = parser.parse_args()

    render(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        delta=args.delta, epsilon=args.epsilon, zeta=args.zeta,
        eta=args.eta, theta=args.theta, prefix=args.prefix,
        fold=args.fold, seam_allowance=args.seam_allowance,
    )
