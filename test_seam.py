"""
test_seam.py — Seam-allowance regression suite for block.py / render.py.

Tests a representative range of bodice measurements (XS through 2XL, petite,
tall, high-contrast hourglass) across a range of seam-allowance values.

Run:
    python3 test_seam.py

Exit code 0 = all passed.  Any failures are printed with details.
"""

import os
import sys
import math
import tempfile
import traceback
import numpy as np

# ── 1. Import project modules ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import block as blk
import render as rnd
import sleeve as slv

# ── 2. Test fixture definitions ───────────────────────────────────────────────
# Each tuple: (alpha, beta, gamma, delta, epsilon, zeta, eta, theta, label)
#   alpha   = waist (in)
#   beta    = bust (in)
#   gamma   = back nape to waist (in)
#   delta   = neck to shoulder (in)
#   epsilon = shoulder center to bust point (in)
#   zeta    = bust point to bust point (in)
#   eta     = front width (in)
#   theta   = back width (in)
TEST_CASES = [
    (24, 32, 14.5, 4.5,  8.0,  6.5, 12.5, 12.5, "XS"),
    (26, 34, 15.0, 4.75, 8.5,  7.0, 13.0, 13.0, "S"),
    (28, 36, 15.5, 5.0,  9.0,  7.0, 14.0, 14.0, "M"),
    (30, 38, 16.0, 5.25, 9.5,  7.5, 14.5, 14.5, "L"),
    (32, 40, 16.0, 5.25, 10.0, 8.0, 15.0, 14.5, "XL"),
    (34, 42, 16.5, 5.5,  10.5, 8.5, 15.5, 15.0, "2XL"),
    # Edge-case proportions
    (25, 33, 13.5, 4.25, 7.5,  6.5, 12.5, 12.5, "petite_S"),
    (27, 35, 17.5, 5.0,  9.0,  7.0, 13.5, 13.5, "tall_S"),
    (26, 40, 15.5, 5.0,  9.5,  7.5, 14.5, 14.0, "hourglass"),
    (30, 44, 16.0, 5.25, 10.5, 9.0, 15.5, 15.0, "full_bust"),  # near max beta
]

SEAM_ALLOWANCES = [0.0, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25]

# Sleeve test cases: (sigma, upsilon, omega, xi, psi, label)
SLEEVE_CASES = [
    (23, 10, 17, 16, 6,   "standard"),
    (21, 9,  15, 14, 5.5, "short"),
    (25, 11, 19, 18, 6.5, "long"),
    (23, 10, 17, 15, 7,   "narrow"),
    (24, 10.5, 18, 17, 5, "wide_cuff"),
]

# ── 3. Helpers ────────────────────────────────────────────────────────────────

def _build_bodice(alpha, beta, gamma, delta, epsilon, zeta, eta, theta):
    """Build and return the block namespace; propagate any exception."""
    return blk.build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta)


def _check_seam_runs(segments, distance, centroid, label):
    """
    Verify that:
      • every "line" (non-dart) edge belongs to exactly one seam run
      • each run's offset polyline has the same number of points as the run
      • no offset point is NaN or Inf
      • each offset point is *farther* from the centroid than the original
        (i.e. the offset is genuinely outward)
    Returns list of error strings (empty = pass).
    """
    errors = []
    centroid = np.asarray(centroid, float)

    # Collect seam line edges
    seam_edges = []
    for seg in segments:
        if seg[0] == "line":
            seam_edges.append((np.asarray(seg[1], float), np.asarray(seg[2], float)))

    runs = rnd._seam_runs(segments)

    # Every line edge must appear in exactly one run
    covered = set()
    for run in runs:
        for i in range(len(run) - 1):
            edge_key = (tuple(np.round(run[i],   4).tolist()),
                        tuple(np.round(run[i+1], 4).tolist()))
            if edge_key in covered:
                errors.append(f"{label}: edge {edge_key} appears in multiple runs")
            covered.add(edge_key)

    expected_keys = set()
    for p0, p1 in seam_edges:
        expected_keys.add((tuple(np.round(p0, 4).tolist()), tuple(np.round(p1, 4).tolist())))

    missing = expected_keys - covered
    if missing:
        errors.append(f"{label}: {len(missing)} seam edge(s) not in any run: "
                      f"{list(missing)[:3]}...")

    # Check each offset polyline
    for ri, run in enumerate(runs):
        if len(run) < 2:
            errors.append(f"{label}: run {ri} has only {len(run)} pt(s)")
            continue
        off = rnd._offset_open_polyline(run, distance, centroid)
        if off.shape != run.shape:
            errors.append(f"{label}: run {ri} shape mismatch {run.shape} vs {off.shape}")
            continue
        if not np.all(np.isfinite(off)):
            errors.append(f"{label}: run {ri} has NaN/Inf in offset")
            continue
        # Each offset pt should be farther from centroid (or same if distance==0)
        d_orig = np.linalg.norm(run - centroid, axis=1)
        d_off  = np.linalg.norm(off  - centroid, axis=1)
        bad = np.where((d_off < d_orig - 1e-3) & (d_orig > 1e-3))[0]
        if len(bad):
            errors.append(
                f"{label}: run {ri}: {len(bad)} offset pt(s) moved inward "
                f"(indices {bad[:3].tolist()})"
            )
    return errors


def _render_to_tempdir(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
                       seam_allowance, fold=False):
    """Render to a temp directory; return (front_path, back_path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "t")
        rnd.render(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
                   prefix=prefix, fold=fold, seam_allowance=seam_allowance)
        front = prefix + "_front.svg"
        back  = prefix + "_back.svg"
        front_ok = os.path.exists(front) and os.path.getsize(front) > 100
        back_ok  = os.path.exists(back)  and os.path.getsize(back)  > 100
        return front_ok, back_ok


# ── 4. Test runner ────────────────────────────────────────────────────────────

def run_tests():
    total, passed, failed = 0, 0, 0
    failure_log = []

    for alpha, beta, gamma, delta, epsilon, zeta, eta, theta, name in TEST_CASES:
        # 4a. Build block
        try:
            bk = _build_bodice(alpha, beta, gamma, delta, epsilon, zeta, eta, theta)
        except Exception as e:
            failed += 1; total += 1
            failure_log.append(f"[BUILD FAIL] {name}: {e}")
            continue

        centroid_front = rnd._sample_outline(bk.front_bodice).mean(axis=0)
        centroid_back  = rnd._sample_outline(bk.back_bodice).mean(axis=0)

        # 4b. Seam-run geometry checks across allowances
        for sa in SEAM_ALLOWANCES:
            total += 1
            errs = []
            if sa > 0:
                errs += _check_seam_runs(bk.front_bodice, sa, centroid_front,
                                         f"{name}/front/sa={sa}")
                errs += _check_seam_runs(bk.back_bodice,  sa, centroid_back,
                                         f"{name}/back/sa={sa}")
            if errs:
                failed += 1
                failure_log.extend(errs)
            else:
                passed += 1

        # 4c. Render smoke-test (no-fold, default + 0.5 + 0)
        for sa in [0.0, 0.5, 0.75]:
            total += 1
            try:
                front_ok, back_ok = _render_to_tempdir(
                    alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
                    seam_allowance=sa)
                if not front_ok or not back_ok:
                    raise RuntimeError("SVG file missing or empty")
                passed += 1
            except Exception as e:
                failed += 1
                failure_log.append(
                    f"[RENDER FAIL] {name} sa={sa}: {e}\n"
                    + traceback.format_exc()
                )

        # 4d. Fold mode smoke-test
        total += 1
        try:
            front_ok, back_ok = _render_to_tempdir(
                alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
                seam_allowance=0.625, fold=True)
            if not front_ok or not back_ok:
                raise RuntimeError("Fold SVG file missing or empty")
            passed += 1
        except Exception as e:
            failed += 1
            failure_log.append(f"[FOLD FAIL] {name}: {e}\n" + traceback.format_exc())

    # ── 5. Sleeve tests ────────────────────────────────────────────────────────
    for sigma, upsilon, omega, xi, psi, name in SLEEVE_CASES:
        # 5a. Build sleeve
        total += 1
        try:
            sl = slv.build(sigma, upsilon, omega, xi, psi)
        except Exception as e:
            failed += 1
            failure_log.append(f"[SLEEVE BUILD FAIL] {name}: {e}")
            continue
        passed += 1

        # 5b. Curve seam allowance offset check
        for sa in [0.375, 0.5, 0.75, 1.0]:
            total += 1
            try:
                centroid = rnd._sample_outline(sl.sleeve_outline).mean(axis=0)
                off = rnd._offset_curve_samples(sl.cap_segments, sa, centroid)
                if len(off) < 2:
                    raise RuntimeError("curve offset produced < 2 points")
                if not np.all(np.isfinite(off)):
                    raise RuntimeError("curve offset has NaN/Inf")
                # Offset should be farther from centroid
                cap_pts = []
                for seg in sl.cap_segments:
                    _, func, _, _ = seg
                    for t in np.linspace(0, 1, 20):
                        cap_pts.append(func(t))
                cap_pts = np.array(cap_pts)
                d_orig = np.linalg.norm(cap_pts - centroid, axis=1).mean()
                d_off  = np.linalg.norm(off - centroid, axis=1).mean()
                if d_off < d_orig - 0.01:
                    raise RuntimeError(
                        f"curve offset moved inward: orig={d_orig:.3f} off={d_off:.3f}")
                passed += 1
            except Exception as e:
                failed += 1
                failure_log.append(f"[SLEEVE CURVE SA FAIL] {name} sa={sa}: {e}")

        # 5c. Render smoke-test
        for sa in [0.0, 0.5, 0.75]:
            total += 1
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    prefix = os.path.join(tmpdir, "sleeve_test")
                    slv.render(sigma, upsilon, omega, xi, psi,
                               prefix=prefix, seam_allowance=sa)
                    svg_path = prefix + ".svg"
                    if not os.path.exists(svg_path) or os.path.getsize(svg_path) < 100:
                        raise RuntimeError("SVG file missing or empty")
                passed += 1
            except Exception as e:
                failed += 1
                failure_log.append(
                    f"[SLEEVE RENDER FAIL] {name} sa={sa}: {e}\n"
                    + traceback.format_exc()
                )

    # ── 6. Report ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Seam allowance test suite")
    print(f"  Cases:  {len(TEST_CASES)} bodice + {len(SLEEVE_CASES)} sleeve")
    print(f"  SA values tested per case: {SEAM_ALLOWANCES}")
    print("=" * 60)
    print(f"  Total checks : {total}")
    print(f"  Passed       : {passed}")
    print(f"  Failed       : {failed}")
    print("=" * 60)
    if failure_log:
        print("\nFAILURES:")
        for msg in failure_log:
            print("  •", msg)
        print()
    else:
        print("\n  All checks passed.\n")

    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
