import argparse
import numpy as np
from block import build

SCALE        = 96   # px per inch
MARGIN       = 60   # px
LABEL_OFFSET = 20   # px — inward push along bisector
FONT_SIZE    = 11   # px


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
               outline_labels, interior_labels):
    # bounding box: outline control pts + construction lines + dart lines + all label pts
    extra = (  [p for seg in construction_lines for p in seg]
             + [p for seg in dart_lines         for p in seg]
             + list(outline_labels.values())
             + list(interior_labels.values()) )
    all_pts = np.vstack(
        [_sample_bbox(outline)] + [np.atleast_2d(p) for p in extra]
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
           prefix="bodice"):
    bk = build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta)

    # Points shared by both views (construction rectangle corners + reference pts)
    shared_interior = {
        "B":  bk.B,  "C":  bk.C,
        "E":  bk.E,  "F":  bk.F,  "G":  bk.G,
        "H":  bk.H,  "I":  bk.I,  "J":  bk.J,
        "L":  bk.L,  "R":  bk.R,
        "U":  bk.U,  "EE": bk.EE, "CC": bk.CC,
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
    )

    _write_svg(
        f"{prefix}_front.svg",
        bk.front_bodice,
        construction_lines=bk.construction_lines,
        dart_lines=bk.front_dart_lines,
        fill="#fdeede", stroke="#aa5522",
        outline_labels={
            "D":  bk.D,  "M":  bk.M,  "K":  bk.K,  "N":  bk.N,
            "P":  bk.P,  "O":  bk.O,  "Q":  bk.Q,
            "T":  bk.T,  "V":  bk.V,  "W":  bk.W,
            "UU": bk.UU, "VV": bk.VV, "WW": bk.WW,
            "S":  bk.S,
        },
        interior_labels=shared_interior,
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
    args = parser.parse_args()

    render(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        delta=args.delta, epsilon=args.epsilon, zeta=args.zeta,
        eta=args.eta, theta=args.theta, prefix=args.prefix,
    )
