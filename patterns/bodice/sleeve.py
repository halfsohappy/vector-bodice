"""Sleeve piece.

Measurements:
    sigma:   shoulder to wrist
    upsilon: underarm to elbow
    omega:   underarm to wrist
    xi:      armscye
    psi:     wrist
"""

import numpy as np
from types import SimpleNamespace

from geometry import xon_line, catmull_rom_chain


# ── Builder ───────────────────────────────────────────────────────────────────

def build(sigma, upsilon, omega, xi, psi):
    """Compute all derived measurements, points, and outlines for a sleeve.
    Returns a SimpleNamespace."""

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
    cap_segments = catmull_rom_chain(cap_pts)

    # ── Outline ───────────────────────────────────────────────────────────────
    # K → G → [cap: G→M→N→E→O→P→H] → H → L → Q → R → Q → K
    outline = (
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
        # outline and construction
        outline=outline,
        construction_lines=construction_lines,
        unclipped_construction_lines=unclipped_construction_lines,
        dart_lines=dart_lines,
    )
