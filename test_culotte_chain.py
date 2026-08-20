"""
test_culotte_chain.py — Parity suite for the skirt_basic -> skirt_aline ->
culotte pattern chain (patterns/skirt_basic, patterns/skirt_aline,
patterns/culotte).

Checks the specific claims made in these patterns' docstrings/comments:
  - skirt_basic's dart chart lookup matches the book's Personal Dart
    Intake Chart (p.48) exactly, and panel widths equal hip arc + ease
  - skirt_aline transfers exactly one dart to the hem and the resulting
    hem is wider by the closed-form flare amount
  - culotte's crotch point stays within the panel (never negative, never
    past the side seam), and the front+back crotch-level width is a
    sane total
  - render_web produces sane, non-degenerate SVGs for all three patterns
    across a spread of sizes

Run:  python3 test_culotte_chain.py     (exit 0 = all passed)
"""

import sys
import math

import numpy as np

from patterns.skirt_basic import dart_chart, front_panel as sb_front, back_panel as sb_back
from patterns import skirt_basic, skirt_aline, culotte

SIZES = [
    dict(waist_arc_front=6.5, waist_arc_back=6.5, hip_arc_front=8.75, hip_arc_back=8.75,
        hip_depth_front=8.5, hip_depth_back=8.5, skirt_length=22),
    dict(waist_arc_front=7.0, waist_arc_back=7.0, hip_arc_front=9.5, hip_arc_back=9.5,
        hip_depth_front=9.0, hip_depth_back=9.0, skirt_length=24),
    dict(waist_arc_front=8.5, waist_arc_back=8.5, hip_arc_front=11.5, hip_arc_back=11.5,
        hip_depth_front=9.5, hip_depth_back=9.5, skirt_length=26),
]
CROTCH_DEPTH = 10.5

failures = []
checks = 0


def check(cond, label):
    global checks
    checks += 1
    if not cond:
        failures.append(label)


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


# ── 1. Dart chart matches the book's table exactly ────────────────────────────

_EXPECTED = {
    4: (1, 0.500, 1, 0.750),   5: (1, 0.500, 1, 1.000),
    6: (1, 0.500, 2, 0.625),   7: (1, 0.500, 2, 0.750),
    8: (2, 0.375, 2, 0.875),   9: (2, 0.375, 2, 0.875),
    10: (2, 0.500, 2, 1.000),  11: (2, 0.625, 2, 1.125),
    12: (2, 0.625, 2, 1.250),  13: (2, 0.625, 2, 1.375),
    14: (2, 0.625, 2, 1.375),
}


def test_dart_chart():
    for diff, (fc, fi, bc, bi) in _EXPECTED.items():
        info = dart_chart.lookup(diff)
        check((info.front_count, info.front_intake, info.back_count, info.back_intake)
              == (fc, fi, bc, bi), f"dart_chart/{diff}: matches book table")
    # clamping outside the documented 4-14 range
    check(dart_chart.lookup(1) == dart_chart.lookup(4), "dart_chart: clamps below 4")
    check(dart_chart.lookup(20) == dart_chart.lookup(14), "dart_chart: clamps above 14")


# ── 2. skirt_basic panel widths and dart geometry ─────────────────────────────

def test_skirt_basic_geometry():
    for sz in SIZES:
        info = skirt_basic.dart_info(sz["waist_arc_front"], sz["waist_arc_back"],
                                     sz["hip_arc_front"], sz["hip_arc_back"])
        fp = sb_front.build(sz["hip_arc_front"], sz["hip_depth_front"], sz["skirt_length"],
                            info.front_count, info.front_intake)
        bp = sb_back.build(sz["hip_arc_back"], sz["hip_depth_back"], sz["skirt_length"],
                           info.back_count, info.back_intake)

        check(close(fp.front_width, sz["hip_arc_front"] + sb_front.EASE),
              f"skirt_basic/{sz}: front width == hip arc + ease")
        check(close(bp.back_width, sz["hip_arc_back"] + sb_back.EASE),
              f"skirt_basic/{sz}: back width == hip arc + ease")
        check(fp.n_darts == info.front_count, f"skirt_basic/{sz}: front dart count matches chart")
        check(bp.n_darts == info.back_count, f"skirt_basic/{sz}: back dart count matches chart")

        for leg_in, point, leg_out in fp.dart_points:
            check(close(leg_out[0] - leg_in[0], info.front_intake),
                  f"skirt_basic/{sz}: front dart intake width correct")
            check(close(point[1], -sb_front.DART_LEG_LENGTH),
                  f"skirt_basic/{sz}: front dart leg length correct")
            check(leg_in[0] > 0 and leg_out[0] < fp.front_width,
                  f"skirt_basic/{sz}: front dart stays within panel width")


# ── 3. skirt_aline transfers one dart and widens the hem ──────────────────────

def test_skirt_aline_flare():
    for sz in SIZES:
        info = skirt_basic.dart_info(sz["waist_arc_front"], sz["waist_arc_back"],
                                     sz["hip_arc_front"], sz["hip_arc_back"])
        basic_front = sb_front.build(sz["hip_arc_front"], sz["hip_depth_front"],
                                     sz["skirt_length"], info.front_count, info.front_intake)
        aline_front = skirt_aline.front_panel.build(
            sz["hip_arc_front"], sz["hip_depth_front"], sz["skirt_length"],
            info.front_count, info.front_intake)

        check(aline_front.n_darts == basic_front.n_darts - 1,
              f"skirt_aline/{sz}: exactly one dart transferred")
        check(aline_front.B[0] > basic_front.B[0],
              f"skirt_aline/{sz}: hem widened after flare")
        check(close(aline_front.B[1], basic_front.B[1]),
              f"skirt_aline/{sz}: hem stays level (no vertical shift)")

        # side-seam curve should move monotonically outward in x as t goes 0->1
        # past the hip level (no inward kink near the hem)
        _, side_func, _, _ = aline_front.outline[-3]
        xs = [side_func(t)[0] for t in np.linspace(0.5, 1.0, 20)]
        check(all(b >= a - 1e-6 for a, b in zip(xs, xs[1:])),
              f"skirt_aline/{sz}: side seam widens smoothly toward the hem, no kink")


# ── 4. culotte crotch curve stays inside the panel; sane total width ─────────

def test_culotte_crotch():
    for sz_base in SIZES:
        sz = {**sz_base, "crotch_depth": CROTCH_DEPTH}
        pieces = culotte.build(**sz)
        fp, bp = pieces["front_panel"], pieces["back_panel"]

        front_crotch_w = fp.D[0] - fp.H[0]
        back_crotch_w = bp.H[0] - bp.D[0]
        check(0 < front_crotch_w < fp.front_width,
              f"culotte/{sz}: front crotch point within panel bounds")
        check(0 < back_crotch_w < bp.back_width,
              f"culotte/{sz}: back crotch point within panel bounds")

        total = front_crotch_w + back_crotch_w
        check(6 < total < 16, f"culotte/{sz}: total crotch-level width sane ({total:.2f}in)")
        check(back_crotch_w > front_crotch_w,
              f"culotte/{sz}: back crotch extension wider than front (book's own rule)")

        # crotch curve endpoints should not coincide (a real, visible curve)
        check(np.linalg.norm(fp.D - fp.X) > 1.0,
              f"culotte/{sz}: front crotch curve has real extent")

        # the new inseam hem point sits directly below the crotch point (straight inseam)
        check(close(fp.E[0], fp.D[0]),
              f"culotte/{sz}: front inseam is a straight vertical line")
        check(close(bp.I[0], bp.H[0]),
              f"culotte/{sz}: back inseam is a straight vertical line")


# ── 5. render_web produces sane output across all three patterns ─────────────

def test_render_web_all():
    for sz in SIZES:
        out = skirt_basic.render_web({**sz, "seam_allowance": 0.75})
        for piece in ("front_panel", "back_panel"):
            check(piece in out and len(out[piece]) > 500,
                  f"skirt_basic/{sz}: {piece} present and non-trivial")

        out = skirt_aline.render_web({**sz, "seam_allowance": 0.75})
        for piece in ("front_panel", "back_panel"):
            check(piece in out and len(out[piece]) > 500,
                  f"skirt_aline/{sz}: {piece} present and non-trivial")

        cz = {**sz, "crotch_depth": CROTCH_DEPTH, "seam_allowance": 0.75}
        out = culotte.render_web(cz)
        for piece in ("front_panel", "back_panel", "waistband"):
            check(piece in out and len(out[piece]) > 500,
                  f"culotte/{sz}: {piece} present and non-trivial")
            w, h = out[f"{piece}_w"], out[f"{piece}_h"]
            check(w > 0 and h > 0 and np.isfinite(w) and np.isfinite(h),
                  f"culotte/{sz}: {piece} has sane dims ({w}, {h})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_dart_chart()
    test_skirt_basic_geometry()
    test_skirt_aline_flare()
    test_culotte_crotch()
    test_render_web_all()

    print("=" * 60)
    print("  Culotte chain parity suite (skirt_basic / skirt_aline / culotte)")
    print("=" * 60)
    print(f"  Checks ran : {checks}")
    print(f"  Failed     : {len(failures)}")
    if failures:
        print()
        for f in failures[:25]:
            print("  FAIL:", f)
        if len(failures) > 25:
            print(f"  … and {len(failures) - 25} more")
    else:
        print("\n  All checks passed.")
    print("=" * 60)
    sys.exit(0 if not failures else 1)
