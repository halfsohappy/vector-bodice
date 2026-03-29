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

def curve_armhole(P, O, N, t):
    NP_dir = (P - N) / np.linalg.norm(P - N)
    lam    = np.linalg.norm(O - P) * 0.5523
    P1     = P + lam * NP_dir
    P2     = np.array([O[0],  O[1] + lam])
    return cubic_bezier(P, P1, P2, O, t)

def curve_back_neck(A, AA, DD, t):
    shoulder_dir = (DD - AA) / np.linalg.norm(DD - AA)
    neck_at_AA   = np.array([-shoulder_dir[1], shoulder_dir[0]])
    lam = np.linalg.norm(AA - A) * 0.5523
    P1  = A  + lam * np.array([1.0, 0.0])
    P2  = AA - lam * neck_at_AA
    return cubic_bezier(A, P1, P2, AA, t)

def curve_back_armhole(BB, O, DD, t):
    DD_BB_dir = (BB - DD) / np.linalg.norm(BB - DD)
    lam = np.linalg.norm(O - BB) * 0.5523
    P1  = BB + lam * DD_BB_dir
    P2  = np.array([O[0],  O[1] + lam])
    return cubic_bezier(BB, P1, P2, O, t)


# ── Builder ───────────────────────────────────────────────────────────────────

def build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta):
    """Compute all derived measurements, points, and outlines for a given set
    of the 8 objective body measurements.  Returns a SimpleNamespace."""

    # Derived measurements
    a = k1(beta)
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

    # Dart points
    E = np.array([(c + 0.5) / 2,        S[1] - a / 2  ])  # bust dart upper base

    F = np.array([S[0] - b / 2,         0             ])   # front waist dart left base
    G = np.array([S[0] + b / 2,         0             ])   # front waist dart right base

    _cx = FF[0] / 2 - 0.75
    H = np.array([_cx - b / 2,          0             ])   # back waist dart left base
    I = np.array([_cx + b / 2,          0             ])   # back waist dart right base
    J = np.array([_cx,           0.5*gamma + a        ])   # back waist dart tip

    # ── Outlines ──────────────────────────────────────────────────────────────
    # Each segment: ("line", P0, P1)  or  ("curve", fn)  where fn(t) → point

    back_bodice = [
        ("line",  B,   A  ),
        ("curve", lambda t: curve_back_neck(A, AA, DD, t)    ),
        ("line",  AA,  DD ),
        ("line",  DD,  BB ),
        ("curve", lambda t: curve_back_armhole(BB, O, DD, t) ),
        ("line",  O,   FF ),
        ("line",  FF,  EE ),
        ("line",  EE,  I  ),
        ("line",  I,   J  ),
        ("line",  J,   H  ),
        ("line",  H,   B  ),
    ]

    front_bodice = [
        ("line",  M,   D  ),
        ("line",  D,   G  ),
        ("line",  G,   W  ),
        ("line",  W,   F  ),
        ("line",  F,   Q  ),
        ("line",  Q,   V  ),
        ("line",  V,   T  ),
        ("line",  T,   E  ),
        ("line",  E,   O  ),
        ("curve", lambda t: curve_armhole(P, O, N, 1 - t)    ),
        ("line",  P,   N  ),
        ("line",  N,   K  ),
        ("curve", lambda t: curve_neck(K, M, t)               ),
    ]

    return SimpleNamespace(
        # objective measurements
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        epsilon=epsilon, zeta=zeta, eta=eta, theta=theta,
        # derived measurements
        a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h,
        # points
        A=A, B=B, C=C, D=D,
        K=K, L=L, M=M, N=N,
        O=O, P=P, Q=Q,
        R=R, S=S,
        T=T, U=U, V=V, W=W,
        AA=AA, BB=BB, CC=CC, DD=DD,
        EE=EE, FF=FF, GG=GG,
        E=E, F=F, G=G, H=H, I=I, J=J,
        # outlines
        back_bodice=back_bodice,
        front_bodice=front_bodice,
    )
