import numpy as np
from types import SimpleNamespace


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

def find_DD(AA, CC, delta):
    direction = (CC - AA) / np.linalg.norm(CC - AA)
    return AA + (delta + 0.5) * direction

def find_FF(S, a, c, V, Q, EE):
    upper_dart = np.array([(c + 0.5) / 2,  S[1] - a / 2])
    VQ_len     = np.linalg.norm(V - Q)
    FF_x       = EE[0]
    FF_y       = upper_dart[1] - np.sqrt(VQ_len**2 - (FF_x - upper_dart[0])**2)
    return np.array([FF_x, FF_y])


# ── Curves ────────────────────────────────────────────────────────────────────

def cubic_bezier(P0, P1, P2, P3, t):
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)[:, None]
    pts = ((1-t)**3 * P0 + 3*(1-t)**2*t * P1 +
           3*(1-t)*t**2 * P2 + t**3 * P3)
    return pts[0] if scalar else pts

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

def curve_back_neck(A, AA, DD, t):
    shoulder_dir = (DD - AA) / np.linalg.norm(DD - AA)
    neck_at_AA   = np.array([-shoulder_dir[1], shoulder_dir[0]])
    lam = np.linalg.norm(AA - A) * 0.5523
    P1  = A  + lam * np.array([1.0, 0.0])
    P2  = AA - lam * neck_at_AA
    return cubic_bezier(A, P1, P2, AA, t)

def curve_back_armhole_upper(AA, DD, BB, t):
    """Back upper armhole: DD → BB with corner at DD → vertical tangent at BB."""
    start_dir = (BB - DD) / np.linalg.norm(BB - DD)
    tangent_BB = np.array([0.0, -1.0])
    chord_len = np.linalg.norm(BB - DD)
    P1 = DD + (1.0/3.0) * chord_len * start_dir
    P2 = BB - (1.0/3.0) * chord_len * tangent_BB
    return cubic_bezier(DD, P1, P2, BB, t)

def curve_back_armhole_lower(BB, O, t):
    """Back lower armhole: BB → O with vertical tangent at BB → horizontal at O."""
    tangent_BB = np.array([0.0, -1.0])
    tangent_O = np.array([1.0, 0.0])
    width = abs(O[0] - BB[0])
    height = abs(BB[1] - O[1])
    P1 = BB + 0.75 * height * tangent_BB
    P2 = O - 0.75 * width * tangent_O
    return cubic_bezier(BB, P1, P2, O, t)


# ── Builder ───────────────────────────────────────────────────────────────────

def build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta, deepen_bust_dart=False):
    """Compute all derived measurements, points, and outlines for a given set
    of the 8 objective body measurements.  Returns a SimpleNamespace."""

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

    # Back: neck and shoulder
    AA = np.array([2.5,  gamma + a + 0.5    ])
    BB = np.array([h,    0.75*gamma + a     ])
    CC = np.array([h,    0.75*gamma + a + 3 ])
    DD = find_DD(AA, CC, delta)

    # Back: side seam and waist
    EE = np.array([f + b - 0.25,  0])
    FF = find_FF(S, a, c, V, Q, EE)
    GG = np.array([0,  FF[1]])

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

    _cx = FF[0] / 2 - 0.75
    XX = np.array([_cx - b / 2,    FF[1]         ])   # back dart left base  (on GG–FF line)
    YY = np.array([_cx + b / 2,    FF[1]         ])   # back dart right base (on GG–FF line)
    ZZ = np.array([_cx,            0.5*gamma + a ])   # back dart tip        (at I–J line)

    # ── Quadratic Bézier control points ───────────────────────────────────────
    # Each CP is the intersection of the tangent lines at the two endpoints.
    # A quadratic Bézier has no inflection points.

    # front neck: K → M
    # tangent at K: vertical   → line x = K[0]
    # tangent at M: horizontal → line y = M[1]
    # intersection = L = (K[0], M[1])  (already defined)
    _neck_cp  = L

    # front armhole: P → O  (stored reversed as O → P)
    # tangent at P: direction N→P
    # tangent at O: vertical   → line x = O[0]
    _NP_dir   = (P - N) / np.linalg.norm(P - N)
    _arm_cp   = P + ((O[0] - P[0]) / _NP_dir[0]) * _NP_dir

    # front neck: K → M
    # tangent at K: vertical   → line x = K[0]
    # tangent at M: horizontal → line y = M[1]
    # intersection = L = (K[0], M[1])  (already defined)
    _neck_cp  = L

    # back neck: A → AA
    # tangent at A:  horizontal → line y = A[1]
    # tangent at AA: perpendicular to shoulder (neck_at_AA direction)
    _sdir     = (DD - AA) / np.linalg.norm(DD - AA)
    _perp     = np.array([-_sdir[1], _sdir[0]])       # 90° CCW of shoulder
    _bneck_cp = AA + ((A[1] - AA[1]) / _perp[1]) * _perp

    # front armhole: now split into upper (N→P, vertical tangent at P) and lower (P→O, horizontal tangent at O)
    # back armhole: now split into upper (DD→BB, horizontal tangent at BB) and lower (BB→O, vertical tangent at O)
    
    # ── Outlines ──────────────────────────────────────────────────────────────
    # Each segment: ("line", P0, P1), ("quadratic", P0, CP, P3), or ("cubic_curve", func, P0, P1)

    back_bodice = [
        ("line",      GG,  A  ),                         # center back (GG→A)
        ("quadratic", A,   _bneck_cp, AA),               # back neck
        ("line",      AA,  DD ),                         # shoulder seam
        ("cubic_curve", lambda t: curve_back_armhole_upper(AA, DD, BB, t), DD, BB ),  # back armhole, upper (DD→BB)
        ("cubic_curve", lambda t: curve_back_armhole_lower(BB, O, t), BB, O  ),   # back armhole, lower (BB→O)
        ("line",      O,   FF ),                         # side seam
        ("line",      FF,  YY ),                         # bottom, right of dart
        ("dart",      YY,  ZZ ),                         # back dart leg
        ("dart",      ZZ,  XX ),                         # back dart leg
        ("line",      XX,  GG ),                         # bottom to center back
    ]

    front_bodice = [
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

    # ── Construction lines ────────────────────────────────────────────────────
    construction_lines = [
        (E,  F ),   # vertical centerline
        (G,  H ),   # upper horizontal  (G–H line)
        (I,  J ),   # middle horizontal (I–J line)
        (GG, FF),   # back waist horizontal
    ]

    _mid_vvww = (VV + WW) / 2   # midpoint of front waist dart base
    _mid_xxyy = (XX + YY) / 2   # midpoint of back dart base

    front_dart_lines = [
        (UU, U ),           # bust dart: upper base to U
        (U,  V ),           # bust dart: U to lower base
        (U,  T ),           # bust dart: U to tip
        (VV, WW),           # front waist dart base line
        (W,  _mid_vvww),    # front waist dart: tip to base midpoint
    ]

    back_dart_lines = [
        (XX, YY),           # back dart base line
        (ZZ, _mid_xxyy),    # back dart: tip to base midpoint
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
        AA=AA, BB=BB, CC=CC, DD=DD,
        EE=EE, FF=FF, GG=GG,
        UU=UU, VV=VV, WW=WW, XX=XX, YY=YY, ZZ=ZZ,
        # outlines and construction
        back_bodice=back_bodice,
        front_bodice=front_bodice,
        construction_lines=construction_lines,
        front_dart_lines=front_dart_lines,
        back_dart_lines=back_dart_lines,
    )
