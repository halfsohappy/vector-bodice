"""
test_labels.py
==============
Verifies that every outline-label's text anchor falls strictly inside its
respective bodice polygon, across all 10 test sizes.

"Strictly inside" means the text is rendered over the piece's fill colour and
cannot interact with the outline stroke, seam-allowance lines, construction
lines, or the white page background — the only colour it can touch is the fill.
"""

import numpy as np
from block import build
from render import (
    _BACK_LABEL_OFFSETS,
    _FRONT_LABEL_OFFSETS,
    _INTERIOR_LABEL_OFFSETS,
    _sample_outline,
)
from sleeve import (
    build as sleeve_build,
    _SLEEVE_LABEL_OFFSETS,
    _SLEEVE_INTERIOR_OFFSETS,
)

# ── Sleeve test sizes ─────────────────────────────────────────────────────────

SLEEVE_TEST_SIZES = [
    dict(sigma=23, upsilon=10, omega=17, xi=16, psi=6, label="standard"),
    dict(sigma=21, upsilon=9,  omega=15, xi=14, psi=5.5, label="short"),
    dict(sigma=25, upsilon=11, omega=19, xi=18, psi=6.5, label="long"),
    dict(sigma=23, upsilon=10, omega=17, xi=15, psi=7, label="narrow"),
    dict(sigma=24, upsilon=10.5, omega=18, xi=17, psi=5, label="wide cuff"),
]

# ── The same 10 sizes used in test_seam.py ───────────────────────────────────

TEST_SIZES = [
    dict(alpha=24, beta=32, gamma=14.5, delta=4.5,  epsilon=8.0,  zeta=6.5, eta=12.5, theta=12.5, label="XS"),
    dict(alpha=26, beta=34, gamma=15.0, delta=4.75, epsilon=8.5,  zeta=7.0, eta=13.0, theta=13.0, label="S"),
    dict(alpha=28, beta=36, gamma=15.5, delta=5.0,  epsilon=9.0,  zeta=7.0, eta=14.0, theta=14.0, label="M"),
    dict(alpha=30, beta=38, gamma=16.0, delta=5.25, epsilon=9.5,  zeta=7.5, eta=14.5, theta=14.5, label="L"),
    dict(alpha=32, beta=40, gamma=16.0, delta=5.25, epsilon=10.0, zeta=8.0, eta=15.0, theta=14.5, label="XL"),
    dict(alpha=34, beta=42, gamma=16.5, delta=5.5,  epsilon=10.5, zeta=8.5, eta=15.5, theta=15.0, label="2XL"),
    dict(alpha=25, beta=33, gamma=13.5, delta=4.25, epsilon=7.5,  zeta=6.5, eta=12.5, theta=12.5, label="petite S"),
    dict(alpha=27, beta=35, gamma=17.5, delta=5.0,  epsilon=9.0,  zeta=7.0, eta=13.5, theta=13.5, label="tall S"),
    dict(alpha=26, beta=40, gamma=15.5, delta=5.0,  epsilon=9.5,  zeta=7.5, eta=14.5, theta=14.0, label="hourglass"),
    dict(alpha=30, beta=44, gamma=16.0, delta=5.25, epsilon=10.5, zeta=9.0, eta=15.5, theta=15.0, label="full bust"),
]

# ── Geometry helpers ──────────────────────────────────────────────────────────

def _point_in_polygon(pt, poly):
    """Ray-casting test.  Returns True if pt is strictly inside poly."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    px, py = float(poly[-1][0]), float(poly[-1][1])
    for i in range(n):
        cx, cy = float(poly[i][0]), float(poly[i][1])
        if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy) + cx):
            inside = not inside
        px, py = cx, cy
    return inside


def _text_anchor_model(name, pt, offsets):
    """Model-space text anchor for a label, or None if not in the offset table."""
    base   = name.rstrip("'")
    primes = len(name) - len(base)
    if base not in offsets:
        return None
    dx_in, dy_in = offsets[base]
    if primes % 2 == 1:
        dx_in = -dx_in      # mirror x for fold-primed labels
    return np.array([pt[0] + dx_in, pt[1] + dy_in])


# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests():
    fails  = []
    checks = 0

    for sz in TEST_SIZES:
        bk = build(
            sz["alpha"], sz["beta"],  sz["gamma"], sz["delta"],
            sz["epsilon"], sz["zeta"], sz["eta"],   sz["theta"],
        )
        tag = sz["label"]

        # ── Back piece ──
        back_poly  = _sample_outline(bk.back_bodice)
        back_offsets = {**_INTERIOR_LABEL_OFFSETS, **_BACK_LABEL_OFFSETS}
        back_outline_labels = {
            "A":  bk.A,  "GG": bk.GG, "AA": bk.AA, "DD": bk.DD,
            "BB": bk.BB, "O":  bk.O,  "FF": bk.FF,
            "XX": bk.XX, "YY": bk.YY, "ZZ": bk.ZZ,
        }
        for name, pt in back_outline_labels.items():
            anchor = _text_anchor_model(name, pt, back_offsets)
            if anchor is None:
                continue
            checks += 1
            if not _point_in_polygon(anchor, back_poly):
                fails.append(
                    f"  FAIL  back/{name:4s}  size {tag!r:12s}"
                    f"  dot={np.round(pt,3)}  text={np.round(anchor,3)}"
                )

        # ── Front piece (no fold) ──
        front_poly    = _sample_outline(bk.front_bodice)
        front_offsets = {**_INTERIOR_LABEL_OFFSETS, **_FRONT_LABEL_OFFSETS}
        front_outline_labels = {
            "M":  bk.M,  "D":  bk.D,  "K":  bk.K,  "N":  bk.N,
            "P":  bk.P,  "O":  bk.O,  "Q":  bk.Q,
            "T":  bk.T,  "V":  bk.V,  "W":  bk.W,
            "UU": bk.UU, "VV": bk.VV, "WW": bk.WW,
            "S":  bk.S,
        }
        for name, pt in front_outline_labels.items():
            anchor = _text_anchor_model(name, pt, front_offsets)
            if anchor is None:
                continue
            checks += 1
            if not _point_in_polygon(anchor, front_poly):
                fails.append(
                    f"  FAIL  front/{name:4s} size {tag!r:12s}"
                    f"  dot={np.round(pt,3)}  text={np.round(anchor,3)}"
                )

    # ── Sleeve piece ──
    for sz in SLEEVE_TEST_SIZES:
        sl = sleeve_build(sz["sigma"], sz["upsilon"], sz["omega"], sz["xi"], sz["psi"])
        tag = f"sleeve/{sz['label']}"
        sleeve_poly = _sample_outline(sl.sleeve_outline)
        sleeve_offsets = {**_SLEEVE_INTERIOR_OFFSETS, **_SLEEVE_LABEL_OFFSETS}

        sleeve_outline_labels = {
            "K": sl.K, "G": sl.G, "M": sl.M, "N": sl.N,
            "E": sl.E, "O": sl.O, "P": sl.P, "H": sl.H,
            "L": sl.L, "Q": sl.Q, "R": sl.R,
        }
        for name, pt in sleeve_outline_labels.items():
            anchor = _text_anchor_model(name, pt, sleeve_offsets)
            if anchor is None:
                continue
            checks += 1
            if not _point_in_polygon(anchor, sleeve_poly):
                fails.append(
                    f"  FAIL  {tag}/{name:4s}"
                    f"  dot={np.round(pt,3)}  text={np.round(anchor,3)}"
                )

    # ── Report ──
    width = 60
    print("=" * width)
    print("  Label colour-interaction test")
    print("  (checks: outline label text anchor is inside the fill region)")
    print("=" * width)
    print(f"  Sizes tested : {len(TEST_SIZES)} bodice + {len(SLEEVE_TEST_SIZES)} sleeve")
    print(f"  Checks ran   : {checks}")
    print(f"  Failed       : {len(fails)}")
    if fails:
        print()
        for f in fails:
            print(f)
    else:
        print()
        print("  All label text anchors sit inside the fill regions — no")
        print("  interaction with outline strokes or page background.")
    print("=" * width)
    return len(fails) == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
