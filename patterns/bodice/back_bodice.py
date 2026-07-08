"""Back bodice piece.

Drafted on top of the front piece: build() takes the front namespace because
several back points are located relative to front points (the side-seam
bottom FF balances the front side-seam length via S, V and Q, and both
pieces share the underarm point O and the base rectangle).
"""

import numpy as np
from types import SimpleNamespace

from geometry import cubic_bezier


# ── Nontrivial point solvers ───────────────────────────────────────────────────

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

def build(front):
    """Compute all back points and outlines from the front-piece namespace.
    Returns a SimpleNamespace."""

    a, b, c, f, h = front.a, front.b, front.c, front.f, front.h
    gamma, delta  = front.gamma, front.delta
    A, O, S, V, Q = front.A, front.O, front.S, front.V, front.Q

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
    _cx = FF[0] / 2 - 0.75
    XX = np.array([_cx - b / 2,    FF[1]         ])   # back dart left base  (on GG–FF line)
    YY = np.array([_cx + b / 2,    FF[1]         ])   # back dart right base (on GG–FF line)
    ZZ = np.array([_cx,            0.5*gamma + a ])   # back dart tip        (at I–J line)

    # ── Quadratic Bézier control points ───────────────────────────────────────
    # Each CP is the intersection of the tangent lines at the two endpoints.
    # A quadratic Bézier has no inflection points.

    # back neck: A → AA
    # tangent at A:  horizontal → line y = A[1]
    # tangent at AA: perpendicular to shoulder (neck_at_AA direction)
    _sdir     = (DD - AA) / np.linalg.norm(DD - AA)
    _perp     = np.array([-_sdir[1], _sdir[0]])       # 90° CCW of shoulder
    _bneck_cp = AA + ((A[1] - AA[1]) / _perp[1]) * _perp

    # ── Outline ───────────────────────────────────────────────────────────────
    # Each segment: ("line", P0, P1), ("quadratic", P0, CP, P3), or ("cubic_curve", func, P0, P1)

    outline = [
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

    # ── Construction lines ────────────────────────────────────────────────────
    construction_lines = [
        (GG, FF),   # back waist horizontal
    ]

    _mid_xxyy = (XX + YY) / 2   # midpoint of back dart base

    dart_lines = [
        (XX, YY),           # back dart base line
        (ZZ, _mid_xxyy),    # back dart: tip to base midpoint
    ]

    return SimpleNamespace(
        # points
        AA=AA, BB=BB, CC=CC, DD=DD,
        EE=EE, FF=FF, GG=GG,
        XX=XX, YY=YY, ZZ=ZZ,
        # outline and construction
        outline=outline,
        construction_lines=construction_lines,
        dart_lines=dart_lines,
    )
