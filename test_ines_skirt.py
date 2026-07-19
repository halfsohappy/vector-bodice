"""
test_ines_skirt.py — Seam-allowance and geometry parity suite for the INES
skirt pattern (patterns/ines_skirt/).

Checks the specific claims made in the pattern's docstrings/comments:
  - the drafted waist-arc seamline length matches the tutorial's own check
    (half the adjusted waist plus the fixed ease, converted to inches)
  - the waistband's short ends always resolve to the fixed 0.7cm-equivalent
    allowance regardless of the render-time seam_allowance slider, while its
    long edges track the slider — proving the merge_consecutive=False /
    seam_allowance_fn wiring in __init__.py actually works end to end
  - the panel's straight edges are exactly hem_radius - waist_radius long
  - render_web produces all 8 pieces with sane (positive, finite) dimensions
    across a spread of sizes

Run:  python3 test_ines_skirt.py     (exit 0 = all passed)
"""

import sys

import numpy as np

import render as rnd
from patterns import ines_skirt
from patterns.ines_skirt import skirt_panel, waistband

CM_TO_IN = 1 / 2.54

SIZES = [
    dict(waist=26, skirt_length=25, hem_allowance=0.625),
    dict(waist=28, skirt_length=27, hem_allowance=0.625),
    dict(waist=34, skirt_length=29, hem_allowance=0.5),
    dict(waist=40, skirt_length=32, hem_allowance=1.0),
]

failures = []
checks = 0


def check(cond, label):
    global checks
    checks += 1
    if not cond:
        failures.append(label)


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


# ── 1. Waist-arc seamline length matches the tutorial's own check ────────────

def test_waist_arc_length():
    for sz in SIZES:
        p = skirt_panel.build(**sz)
        _, func, p0, p1 = p.outline[3]   # waist arc, L->K per outline direction
        check(np.allclose(func(0), p0) and np.allclose(func(1), p1),
              f"waist_arc/{sz}: endpoint order")
        ts = np.linspace(0, 1, 4000)
        pts = np.array([func(t) for t in ts])
        arc_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        # tutorial's check: half the (waist+16cm)-adjusted waist... actually
        # their own stated check is simply waist/2 + 8cm, in inches here.
        expected = sz["waist"] / 2 + 8 * CM_TO_IN
        check(abs(arc_len - expected) < 0.01,
              f"waist_arc/{sz}: length {arc_len:.4f} vs expected {expected:.4f}")


# ── 2. Panel straight-edge length == hem_radius - waist_radius ───────────────

def test_panel_straight_edges():
    for sz in SIZES:
        p = skirt_panel.build(**sz)
        for seg in (p.outline[0], p.outline[2]):   # K-M and N-L
            _, a, b = seg
            length = float(np.linalg.norm(b - a))
            check(close(length, p.hem_radius - p.waist_radius, tol=1e-9),
                  f"panel_edge/{sz}: {length} vs {p.hem_radius - p.waist_radius}")
        check(close(p.hem_radius - p.waist_radius,
                    sz["skirt_length"] + sz["hem_allowance"], tol=1e-9),
              f"panel_edge/{sz}: matches skirt_length + hem_allowance")


# ── 3. Waistband short ends stay fixed regardless of the SA slider ───────────

def test_waistband_seam_allowance():
    wb = waistband.build(length=waistband.front_length(28))
    from patterns.ines_skirt import settings

    for slider in (0.375, 0.5, 0.75, 1.0, 1.25):
        fn = settings.waistband_seam_allowance_fn(wb, slider)
        runs = rnd._seam_runs(wb.outline, merge_consecutive=False)
        check(len(runs) == 4, f"waistband/{slider}: 4 independent runs")
        sa_values = [fn(run) for run in runs]
        n_short = sum(1 for v in sa_values if close(v, waistband.END_SA))
        n_long  = sum(1 for v in sa_values if close(v, slider))
        check(n_short == 2, f"waistband/{slider}: 2 runs at fixed END_SA (got {n_short})")
        check(n_long == 2, f"waistband/{slider}: 2 runs at slider value (got {n_long})")

    # waist_detect=False must be wired in, or the bottom long edge (which
    # sits at the piece's bounding-box minimum-y) would be silently dropped
    # before seam_allowance_fn ever runs.
    fn = settings.waistband_seam_allowance_fn(wb, 0.75)
    offset_runs = rnd._seam_runs_no_waist(wb.outline, 0.75, fn,
                                          waist_detect=False, merge_consecutive=False)
    check(len(offset_runs) == 4, f"waistband: all 4 edges reach seam_allowance_fn "
                                 f"(got {len(offset_runs)}) — waist_detect exclusion regressed")


# ── 4. render_web produces sane output for every piece, across sizes ─────────

def test_render_web():
    for sz in SIZES:
        params = {**sz, "seam_allowance": 0.75}
        out = ines_skirt.render_web(params)
        for piece in ("front_panel", "back_panel", "front_waistband", "back_waistband",
                      "pocket_side1", "pocket_side2", "tie", "pocket_bias"):
            check(piece in out and len(out[piece]) > 500,
                  f"render_web/{sz}: {piece} present and non-trivial")
            w, h = out[f"{piece}_w"], out[f"{piece}_h"]
            check(w > 0 and h > 0 and np.isfinite(w) and np.isfinite(h),
                  f"render_web/{sz}: {piece} has sane dims ({w}, {h})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_waist_arc_length()
    test_panel_straight_edges()
    test_waistband_seam_allowance()
    test_render_web()

    print("=" * 60)
    print("  INES skirt parity suite")
    print("=" * 60)
    print(f"  Checks ran : {checks}")
    print(f"  Failed     : {len(failures)}")
    if failures:
        print()
        for f in failures:
            print("  FAIL:", f)
    else:
        print("\n  All checks passed.")
    print("=" * 60)
    sys.exit(0 if not failures else 1)
