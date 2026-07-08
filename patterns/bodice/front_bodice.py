"""Front bodice piece.

Drafts the shared construction grid (base rectangle, centerlines) and every
front point, then joins them into the front outline.  The back piece
(back_bodice.py) is drafted on top of this namespace.
"""

import numpy as np
from types import SimpleNamespace

from geometry import cubic_bezier


# ── Lookup tables ─────────────────────────────────────────────────────────────

def k1(bust):
    """Depth of bust dart (Chart 1), keyed on bust measurement."""
    if   bust <= 32:  return 0.5
    elif bust <= 34:  return 0.75
    elif bust <= 36:  return 1.0
    elif bust <= 38:  return 1.5
    elif bust <= 40:  return 1.75
    elif bust <= 42:  return 2.0
    elif bust <= 44:  return 2.5
    raise ValueError(f"bust {bust} out of chart range")

def k2(diff):
    """Width of waist dart (Chart 2), keyed on bust − waist difference."""
    if   diff <= 5.5:  return 0.5
    elif diff <= 7.5:  return 0.75
    elif diff <= 9.5:  return 1.0
    elif diff <= 11.5: return 1.25
    elif diff <= 13.5: return 1.5
    elif diff <= 15.5: return 1.75
    raise ValueError(f"bust−waist difference {diff} out of chart range")


# ── Nontrivial point solvers ───────────────────────────────────────────────────

def find_N(K, delta, gamma, a):
    N_y = gamma + a - 1.375
    N_x = K[0] - np.sqrt(delta**2 - (K[1] - N_y)**2)
    return np.array([N_x, N_y])

def find_S(R, epsilon, c, g):
    S_x = c + 0.5 - g
    S_y = R[1] - np.sqrt(epsilon**2 - (S_x - R[0])**2)
    return np.array([S_x, S_y])


# ── Curves ────────────────────────────────────────────────────────────────────

def curve_neck(K, M, t):
    r   = abs(K[1] - M[1])
    lam = 0.5523 * r
    P1  = np.array([K[0],        K[1] - lam])
    P2  = np.array([M[0] - lam,  M[1]      ])
    return cubic_bezier(K, P1, P2, M, t)

def curve_armhole_upper(K, N, P, t):
    """Front upper armhole: N → P with corner at N → vertical tangent at P."""
    start_dir = (P - N) / np.linalg.norm(P - N)
    tangent_P = np.array([0.0, -1.0])
    chord_len = np.linalg.norm(P - N)
    P1 = N + (1.0/3.0) * chord_len * start_dir
    P2 = P - (1.0/3.0) * chord_len * tangent_P
    return cubic_bezier(N, P1, P2, P, t)

def curve_armhole_lower(P, O, t):
    """Front lower armhole: P → O with vertical tangent at P → horizontal at O."""
    tangent_P = np.array([0.0, -1.0])
    tangent_O = np.array([-1.0, 0.0])
    width = abs(P[0] - O[0])
    height = abs(P[1] - O[1])
    # 0.55 is a perfect circle/ellipse. We use 0.70 to 0.75 for a "boxier" more scooped curve with more area.
    P1 = P + 0.75 * height * tangent_P
    P2 = O - 0.75 * width * tangent_O
    return cubic_bezier(P, P1, P2, O, t)


# ── Builder ───────────────────────────────────────────────────────────────────

def build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
          deepen_bust_dart=False):
    """Compute derived measurements, the shared construction grid, and all
    front points and outlines.  Returns a SimpleNamespace."""

    # Derived measurements
    a = k1(beta)
    if deepen_bust_dart:
        a += 0.5
    b = k2(beta - alpha)
    c = beta / 2
    d = gamma / 4
    e = eta / 2
    f = alpha / 4
    g = zeta / 2
    h = theta / 2

    # Base rectangle
    B = np.array([0,        0        ])
    A = np.array([0,        gamma + a])
    C = np.array([c + 0.5,  gamma + a])
    D = np.array([c + 0.5,  0        ])

    # Front: neck and shoulder
    K = np.array([c - 2.5,  gamma + a + 0.5])
    L = np.array([c - 2.5,  gamma + a - 2.5])
    M = np.array([c + 0.5,  gamma + a - 2.5])
    N = find_N(K, delta, gamma, a)

    # Front: armhole and waist
    O = np.array([(c + 0.5) / 2,  0.5*gamma + a ])
    P = np.array([c + 0.5 - e,    0.75*gamma + a])
    Q = np.array([c + 0.25 - f - b, 0            ])

    # Front: bust point
    R = (K + N) / 2
    S = find_S(R, epsilon, c, g)

    # Front: bust dart
    T = np.array([S[0] - 0.5,      S[1]        ])
    U = np.array([(c + 0.5) / 2,   S[1] - a    ])
    V = np.array([(c + 0.5) / 2,   S[1] - 1.5*a])

    # Front: waist dart
    W = np.array([S[0],  S[1] - 1])

    # Construction line endpoints
    E = np.array([(c + 0.5) / 2,  gamma + a     ])  # top of vertical centerline
    F = np.array([(c + 0.5) / 2,  0             ])  # bottom of vertical centerline
    G = np.array([0,               0.75*gamma + a])  # left of upper horizontal
    H = np.array([c + 0.5,         0.75*gamma + a])  # right of upper horizontal
    I = np.array([0,               0.5*gamma + a ])  # left of middle horizontal (I–J line)
    J = np.array([c + 0.5,         0.5*gamma + a ])  # right of middle horizontal (I–J line)

    # Dart points
    UU = np.array([(c + 0.5) / 2,  S[1] - a / 2  ])  # bust dart upper base

    VV = np.array([S[0] - b / 2,   0             ])   # front waist dart left base
    WW = np.array([S[0] + b / 2,   0             ])   # front waist dart right base

    # ── Quadratic Bézier control points ───────────────────────────────────────
    # Each CP is the intersection of the tangent lines at the two endpoints.
    # A quadratic Bézier has no inflection points.

    # front neck: K → M
    # tangent at K: vertical   → line x = K[0]
    # tangent at M: horizontal → line y = M[1]
    # intersection = L = (K[0], M[1])  (already defined)
    _neck_cp  = L

    # ── Outline ───────────────────────────────────────────────────────────────
    # Each segment: ("line", P0, P1), ("quadratic", P0, CP, P3), or ("cubic_curve", func, P0, P1)

    outline = [
        ("line",      M,   D  ),                         # center front
        ("line",      D,   WW ),                         # waist, right of dart
        ("dart",      WW,  W  ),                         # front waist dart leg
        ("dart",      W,   VV ),                         # front waist dart leg
        ("line",      VV,  Q  ),                         # waist, left of dart
        ("line",      Q,   V  ),                         # side seam, lower
        ("dart",      V,   T  ),                         # bust dart leg
        ("dart",      T,   UU ),                         # bust dart leg
        ("line",      UU,  O  ),                         # side seam, upper
        ("cubic_curve", lambda t: curve_armhole_lower(P, O, 1-t), O, P  ),   # front armhole, lower (O→P)
        ("cubic_curve", lambda t: curve_armhole_upper(K, N, P, 1-t), P, N  ),   # front armhole, upper (P→N)
        ("line",      N,   K  ),                         # shoulder seam
        ("quadratic", K,   _neck_cp,  M ),               # front neck (K→M)
    ]

    # ── Construction lines (shared grid) ──────────────────────────────────────
    construction_lines = [
        (E,  F ),   # vertical centerline
        (G,  H ),   # upper horizontal  (G–H line)
        (I,  J ),   # middle horizontal (I–J line)
    ]

    _mid_vvww = (VV + WW) / 2   # midpoint of front waist dart base

    dart_lines = [
        (UU, U ),           # bust dart: upper base to U
        (U,  V ),           # bust dart: U to lower base
        (U,  T ),           # bust dart: U to tip
        (VV, WW),           # front waist dart base line
        (W,  _mid_vvww),    # front waist dart: tip to base midpoint
    ]

    return SimpleNamespace(
        # objective measurements
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        epsilon=epsilon, zeta=zeta, eta=eta, theta=theta,
        # derived measurements
        a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h,
        # points
        A=A, B=B, C=C, D=D,
        E=E, F=F, G=G, H=H, I=I, J=J,
        K=K, L=L, M=M, N=N,
        O=O, P=P, Q=Q,
        R=R, S=S,
        T=T, U=U, V=V, W=W,
        UU=UU, VV=VV, WW=WW,
        # outline and construction
        outline=outline,
        construction_lines=construction_lines,
        dart_lines=dart_lines,
    )
