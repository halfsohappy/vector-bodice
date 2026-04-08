import argparse
import numpy as np
from types import SimpleNamespace


# ── Nontrivial point solvers ───────────────────────────────────────────────────

def xon_line(P1, P2, x):
    """Given an x value, find the point on the line P1–P2 with that x value."""
    m = (P1[1] - P2[1]) / (P1[0] - P2[0])
    b = P2[1] - m * P2[0]
    y = m * x + b
    return np.array([x, y])


# ── Curves ────────────────────────────────────────────────────────────────────

def cubic_bezier(P0, P1, P2, P3, t):
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)[:, None]
    pts = ((1-t)**3 * P0 + 3*(1-t)**2*t * P1 +
           3*(1-t)*t**2 * P2 + t**3 * P3)
    return pts[0] if scalar else pts


def _catmull_rom_segment(P_prev, P0, P1, P_next, t):
    """Cubic Bézier from P0→P1 using Catmull-Rom tangents from neighbours."""
    CP1 = P0 + (P1 - P_prev) / 6.0
    CP2 = P1 - (P_next - P0) / 6.0
    return cubic_bezier(P0, CP1, CP2, P1, t)


def _make_cap_segments(cap_pts):
    """Build cubic_curve outline entries for the sleeve cap.

    Uses Catmull-Rom interpolation through the ordered cap points
    so the resulting curve is C¹-continuous.
    Virtual points are reflected at the endpoints to produce natural tangents.
    """
    P_before = cap_pts[0] - (cap_pts[1] - cap_pts[0])   # virtual before first
    P_after  = cap_pts[-1] - (cap_pts[-2] - cap_pts[-1]) # virtual after last
    extended = [P_before] + list(cap_pts) + [P_after]

    segments = []
    for i in range(len(cap_pts) - 1):
        pp, ps, pe, pn = (extended[i], extended[i+1],
                          extended[i+2], extended[i+3])
        # capture by value via default args
        def _curve(t, _pp=pp, _ps=ps, _pe=pe, _pn=pn):
            return _catmull_rom_segment(_pp, _ps, _pe, _pn, t)
        segments.append(("cubic_curve", _curve, ps, pe))
    return segments


# ── Builder ───────────────────────────────────────────────────────────────────

def build(sigma, upsilon, omega, xi, psi):
    """Compute all derived measurements, points, and outlines for a sleeve.

    Measurements:
        sigma:   shoulder to wrist
        upsilon: underarm to elbow
        omega:   underarm to wrist
        xi:      armscye
        psi:     wrist

    Returns a SimpleNamespace.
    """

    # Derived measurements
    chi = xi - 3

    # Construction rectangle
    B = np.array([0, 0])
    A = np.array([0, sigma])
    C = np.array([chi, sigma])
    D = np.array([chi, 0])
    I = np.array([0, omega - upsilon])
    J = np.array([chi, omega - upsilon])

    # Outline of sleeve
    K = np.array([1.5, 0])
    G = np.array([0, omega])
    E = np.array([chi / 2, sigma])
    H = np.array([chi, omega])
    L = np.array([chi - 1.5, 0])

    # F = midpoint of K and L (bottom center)
    F = (K + L) / 2

    # Curve points at top of sleeve (sleeve cap)
    # Left side: inflection at midpoint of E–G line
    EG_mid = (E + G) / 2
    M = np.array([EG_mid[0], EG_mid[1] - 0.75])
    N = np.array([chi / 3,   xon_line(E, G, chi / 3)[1] + 0.75])
    # Right side: inflection stays at P (5/6 of chi)
    O = np.array([2 * chi / 3, xon_line(H, E, 2 * chi / 3)[1] + 1.0])
    P = np.array([5 * chi / 6, xon_line(H, E, 5 * chi / 6)[1]])

    # Wrist opening (placket slit)
    Q = np.array([chi - 1.5 - (psi / 2), 0])
    R = np.array([chi - 1.5 - (psi / 2), 4])

    # ── Sleeve cap curve (Catmull-Rom through G, M, N, E, O, P, H) ───────────
    cap_pts = [G, M, N, E, O, P, H]
    cap_segments = _make_cap_segments(cap_pts)

    # ── Outline ───────────────────────────────────────────────────────────────
    # K → G → [cap: G→M→N→E→O→P→H] → H → L → Q → R → Q → K
    sleeve_outline = (
        [("line", K, G)]                          # left edge
        + cap_segments                            # sleeve cap curve
        + [("line",  H, L),                       # right edge
           ("line",  L, Q),                       # bottom right to slit
           ("dart",  Q, R),                       # slit (up)
           ("dart",  R, Q),                       # slit (down)
           ("line",  Q, K)]                       # bottom left
    )

    # ── Construction lines ────────────────────────────────────────────────────
    # EG and EH are kept separate so they render unclipped (visible beyond body)
    construction_lines = [
        (A, B),    # rectangle left side
        (B, D),    # rectangle bottom
        (D, C),    # rectangle right side
        (C, A),    # rectangle top
        (G, H),    # underarm horizontal
        (E, F),    # vertical center: cap peak to bottom center
        (I, J),    # elbow line
    ]

    unclipped_construction_lines = [
        (G, E),    # left reference line (GE)
        (E, H),    # right reference line (EH)
    ]

    dart_lines = [
        (Q, R),    # slit opening reference
    ]

    return SimpleNamespace(
        # measurements
        sigma=sigma, upsilon=upsilon, omega=omega, xi=xi, psi=psi,
        # derived
        chi=chi,
        # points
        A=A, B=B, C=C, D=D,
        E=E, F=F, G=G, H=H, I=I, J=J,
        K=K, L=L, M=M, N=N, O=O, P=P,
        Q=Q, R=R,
        # cap curve segments (for curve seam allowance)
        cap_segments=cap_segments,
        # outlines and construction
        sleeve_outline=sleeve_outline,
        construction_lines=construction_lines,
        unclipped_construction_lines=unclipped_construction_lines,
        dart_lines=dart_lines,
    )


# ── Per-label offset tables ──────────────────────────────────────────────────
# All values in model-space inches (dx rightward, dy upward).

_SLEEVE_LABEL_OFFSETS = {
    "K":  ( 0.50,  0.71),   # bottom left → inward
    "G":  ( 0.71,  0.00),   # top left → right (into sleeve)
    "M":  ( 0.71,  0.00),   # cap, left side → right
    "N":  ( 0.00, -0.80),   # cap, left-center → down (into sleeve)
    "E":  ( 0.00, -0.80),   # cap peak → down (into sleeve)
    "O":  ( 0.00, -0.80),   # cap, right-center → down (into sleeve)
    "P":  (-0.71,  0.00),   # cap, right side → left
    "H":  (-0.71,  0.00),   # top right → left (into sleeve)
    "L":  (-0.50,  0.71),   # bottom right → inward
    "Q":  (-0.71,  0.71),   # slit base → inward
    "R":  (-0.71,  0.00),   # slit top → left (into sleeve)
}

_SLEEVE_INTERIOR_OFFSETS = {
    "A":  ( 0.40, -0.35),   # top-left rectangle corner
    "B":  ( 0.40,  0.35),   # bottom-left rectangle corner
    "C":  (-0.40, -0.35),   # top-right rectangle corner
    "D":  (-0.40,  0.35),   # bottom-right rectangle corner
    "F":  ( 0.35,  0.40),   # bottom center
    "I":  ( 0.40,  0.00),   # elbow line left
    "J":  (-0.40,  0.00),   # elbow line right
}


# ── Render ────────────────────────────────────────────────────────────────────

def render_svg(sigma, upsilon, omega, xi, psi,
               seam_allowance=0.75, white_fill=False):
    """Return {'sleeve': svg_str, 'sleeve_w': …, 'sleeve_h': …}."""
    from render import _write_svg, FONT_SIZE

    sl = build(sigma, upsilon, omega, xi, psi)

    outline_labels = {
        "K": sl.K, "G": sl.G, "M": sl.M, "N": sl.N,
        "E": sl.E, "O": sl.O, "P": sl.P, "H": sl.H,
        "L": sl.L, "Q": sl.Q, "R": sl.R,
    }
    interior_labels = {
        "A": sl.A, "B": sl.B, "C": sl.C, "D": sl.D,
        "F": sl.F, "I": sl.I, "J": sl.J,
    }

    # Text annotations: large labels in construction-line color
    ann_color = "#aaa"
    ann_size = FONT_SIZE * 2
    ef_x = sl.E[0]  # x of vertical center line EF
    text_annotations = [
        ("Back of Sleeve",  np.array([sl.H[0] - 1.5, sl.H[1] - 2.0]), ann_color, ann_size),
        ("Front of Sleeve", np.array([sl.G[0] + 1.5, sl.G[1] - 2.0]), ann_color, ann_size),
        ("Elbow Line",      np.array([ef_x - 2.5, sl.I[1] + 0.45]), ann_color, ann_size),
        ("Bicep Line",      np.array([ef_x - 2.5, sl.G[1] + 0.45]), ann_color, ann_size),
    ]

    svg, w, h = _write_svg(
        None,
        sl.sleeve_outline,
        construction_lines=sl.construction_lines,
        dart_lines=sl.dart_lines,
        fill="white" if white_fill else "#d5f5e3",
        stroke="#2d6a4f",
        outline_labels=outline_labels,
        interior_labels=interior_labels,
        seam_allowance=seam_allowance,
        label_offsets={**_SLEEVE_INTERIOR_OFFSETS, **_SLEEVE_LABEL_OFFSETS},
        curve_seam_segments=sl.cap_segments,
        curve_seam_allowance=seam_allowance,
        unclipped_construction_lines=sl.unclipped_construction_lines,
        text_annotations=text_annotations,
    )
    return {'sleeve': svg, 'sleeve_w': w, 'sleeve_h': h}


def render(sigma, upsilon, omega, xi, psi,
           prefix="sleeve", seam_allowance=0.75):
    """Render sleeve block to SVG file."""
    from render import _write_svg, FONT_SIZE

    sl = build(sigma, upsilon, omega, xi, psi)

    outline_labels = {
        "K": sl.K, "G": sl.G, "M": sl.M, "N": sl.N,
        "E": sl.E, "O": sl.O, "P": sl.P, "H": sl.H,
        "L": sl.L, "Q": sl.Q, "R": sl.R,
    }
    interior_labels = {
        "A": sl.A, "B": sl.B, "C": sl.C, "D": sl.D,
        "F": sl.F, "I": sl.I, "J": sl.J,
    }

    # Text annotations: large labels in construction-line color
    ann_color = "#aaa"
    ann_size = FONT_SIZE * 2
    ef_x = sl.E[0]
    text_annotations = [
        ("Back of Sleeve",  np.array([sl.H[0] - 1.5, sl.H[1] - 2.0]), ann_color, ann_size),
        ("Front of Sleeve", np.array([sl.G[0] + 1.5, sl.G[1] - 2.0]), ann_color, ann_size),
        ("Elbow Line",      np.array([ef_x - 2.5, sl.I[1] + 0.45]), ann_color, ann_size),
        ("Bicep Line",      np.array([ef_x - 2.5, sl.G[1] + 0.45]), ann_color, ann_size),
    ]

    _write_svg(
        f"{prefix}.svg",
        sl.sleeve_outline,
        construction_lines=sl.construction_lines,
        dart_lines=sl.dart_lines,
        fill="#d5f5e3", stroke="#2d6a4f",
        outline_labels=outline_labels,
        interior_labels=interior_labels,
        seam_allowance=seam_allowance,
        label_offsets={**_SLEEVE_INTERIOR_OFFSETS, **_SLEEVE_LABEL_OFFSETS},
        curve_seam_segments=sl.cap_segments,
        curve_seam_allowance=seam_allowance,
        unclipped_construction_lines=sl.unclipped_construction_lines,
        text_annotations=text_annotations,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a sleeve block to SVG.")
    parser.add_argument("--sigma",   type=float, required=True,
                        help="shoulder to wrist")
    parser.add_argument("--upsilon", type=float, required=True,
                        help="underarm to elbow")
    parser.add_argument("--omega",   type=float, required=True,
                        help="underarm to wrist")
    parser.add_argument("--xi",      type=float, required=True,
                        help="armscye")
    parser.add_argument("--psi",     type=float, required=True,
                        help="wrist")
    parser.add_argument("--prefix",  type=str, default="sleeve",
                        help="output filename prefix")
    parser.add_argument("--seam-allowance", type=float, default=0.75,
                        help="seam allowance in inches")
    args = parser.parse_args()

    render(
        sigma=args.sigma, upsilon=args.upsilon, omega=args.omega,
        xi=args.xi, psi=args.psi, prefix=args.prefix,
        seam_allowance=args.seam_allowance,
    )
