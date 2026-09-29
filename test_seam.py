"""
test_seam.py — Seam-allowance regression suite for render.py and the patterns.

Tests a representative range of bodice measurements (XS through 2XL, petite,
tall, high-contrast hourglass) across a range of seam-allowance values, then
sweeps every pattern in patterns/index.json at its manifest test sizes.

The core property checked everywhere: every seam-allowance offset point
(straight runs and curve groups alike) lies OUTSIDE the piece outline.  The
outward side is decided by the outline's winding direction, so this holds on
concave edges (neck and armhole scoops, the neckline against a centre-front
fold) where a "move away from the centroid" test would flip the allowance
into the garment.

Run:
    python3 test_seam.py

Exit code 0 = all passed.  Any failures are printed with details.
"""

import os
import sys
import json
import importlib
import tempfile
import traceback
import numpy as np

# ── 1. Import project modules ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import render as rnd
import patterns.bodice as blk
import patterns.bodice.sleeve as slv
from patterns.bodice import settings as bodice_settings

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

# Offset points may graze the outline at a corner; anything deeper than this
# (inches) inside the piece is a genuinely inverted allowance.
INSIDE_TOL = 0.02

# ── 3. Helpers ────────────────────────────────────────────────────────────────

def _build_bodice(alpha, beta, gamma, delta, epsilon, zeta, eta, theta):
    """Build and return the block namespace; propagate any exception."""
    return blk.build(alpha, beta, gamma, delta, epsilon, zeta, eta, theta)


def _inside_points(off_pts, poly):
    """Indices of offset points lying inside the outline polygon by more
    than INSIDE_TOL.  Vectorised equivalents of render._pip / _poly_dist so
    the sweep over every pattern stays fast."""
    P  = np.atleast_2d(np.asarray(off_pts, float))
    A  = np.asarray(poly, float)
    B  = np.roll(A, -1, axis=0)
    # ray-casting: for each point, count crossings of edges A→B
    x, y   = P[:, 0][:, None], P[:, 1][:, None]
    ax, ay = A[:, 0][None, :], A[:, 1][None, :]
    bx, by = B[:, 0][None, :], B[:, 1][None, :]
    straddle = (ay > y) != (by > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (bx - ax) * (y - ay) / (by - ay) + ax
    inside = (np.sum(straddle & (x < xint), axis=1) % 2) == 1
    if not inside.any():
        return []
    # distance from each inside point to the nearest polygon edge
    idx = np.where(inside)[0]
    Q   = P[idx]
    V   = B - A                                         # edge vectors
    L2  = np.maximum(np.sum(V * V, axis=1), 1e-12)
    W   = Q[:, None, :] - A[None, :, :]                 # point - edge start
    t   = np.clip(np.sum(W * V[None, :, :], axis=2) / L2[None, :], 0.0, 1.0)
    proj = A[None, :, :] + t[:, :, None] * V[None, :, :]
    dist = np.min(np.linalg.norm(Q[:, None, :] - proj, axis=2), axis=1)
    return [int(i) for i, d in zip(idx, dist) if d > INSIDE_TOL]


def _check_seam_runs(segments, distance, label):
    """
    Verify that:
      • every "line" (non-dart) edge belongs to exactly one seam run
      • each run's offset polyline has the same number of points as the run
      • no offset point is NaN or Inf
      • the run's endpoints are offset by exactly *distance* (pure perpendicular)
      • no offset point lies inside the piece (the offset is genuinely outward)
    Returns list of error strings (empty = pass).
    """
    errors = []
    poly        = rnd._sample_outline(segments)
    orientation = rnd._outline_orientation(poly)

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
        off = rnd._offset_open_polyline(run, distance, orientation)
        if off.shape != run.shape:
            errors.append(f"{label}: run {ri} shape mismatch {run.shape} vs {off.shape}")
            continue
        if not np.all(np.isfinite(off)):
            errors.append(f"{label}: run {ri} has NaN/Inf in offset")
            continue
        for end in (0, -1):
            d = float(np.linalg.norm(off[end] - run[end]))
            if abs(d - distance) > 1e-6:
                errors.append(f"{label}: run {ri} endpoint {end} offset by {d:.4f}, "
                              f"expected {distance}")
        bad = _inside_points(off, poly)
        if bad:
            errors.append(
                f"{label}: run {ri} ({np.round(run[0], 2).tolist()}→"
                f"{np.round(run[-1], 2).tolist()}): {len(bad)} offset pt(s) "
                f"inside the piece (indices {bad[:3]})"
            )
    return errors


def _check_curve_groups(segments, groups, distance, label):
    """Every curve-group offset must be finite and lie outside the piece."""
    errors = []
    poly        = rnd._sample_outline(segments)
    orientation = rnd._outline_orientation(poly)
    for gi, group in enumerate(groups):
        off = rnd._offset_curve_samples(group, distance, orientation)
        if len(off) < 2:
            errors.append(f"{label}: curve group {gi} offset produced < 2 points")
            continue
        if not np.all(np.isfinite(off)):
            errors.append(f"{label}: curve group {gi} offset has NaN/Inf")
            continue
        bad = _inside_points(off, poly)
        if bad:
            errors.append(
                f"{label}: curve group {gi}: {len(bad)}/{len(off)} offset pt(s) "
                f"inside the piece (indices {bad[:3]}…{bad[-3:]})"
            )
    return errors


def _check_rendered_piece(label, outline, kw):
    """Recompute exactly the offsets _write_svg draws for one piece (from its
    keyword arguments) and check none of them land inside the outline."""
    errors = []
    poly        = rnd._sample_outline(outline)
    orientation = rnd._outline_orientation(poly)
    sa    = kw.get("seam_allowance", 0)
    sa_fn = kw.get("seam_allowance_fn")
    for run, run_sa in rnd._seam_runs_no_waist(
            outline, sa, sa_fn,
            waist_detect=kw.get("waist_detect", True),
            merge_consecutive=kw.get("merge_consecutive", True)):
        off = rnd._offset_open_polyline(run, run_sa, orientation)
        bad = _inside_points(off, poly)
        if bad:
            errors.append(f"{label}: line run {np.round(run[0], 2).tolist()}→"
                          f"{np.round(run[-1], 2).tolist()}: {len(bad)} offset "
                          f"pt(s) inside the piece")
    groups, csa = kw.get("curve_seam_segments"), kw.get("curve_seam_allowance")
    if groups and csa and csa > 1e-6:
        errors += _check_curve_groups(outline, groups, csa, label)
    for pt in (kw.get("notches") or []):
        d = rnd._notch_dir(pt, poly, orientation)
        probe = np.asarray(pt, float) + d * 0.1
        if not rnd._pip(probe, poly):
            errors.append(f"{label}: notch at {np.round(pt, 2).tolist()} points "
                          f"out of the piece")
    return errors


def _render_to_tempdir(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
                       seam_allowance, fold=False):
    """Render to a temp directory; return (front_path, back_path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "t")
        blk.render(alpha, beta, gamma, delta, epsilon, zeta, eta, theta,
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

        folded_front, _ = blk._front_piece_args(bk, fold=True)
        pieces = [("front",      bk.front_bodice),
                  ("back",       bk.back_bodice),
                  ("front_fold", folded_front)]

        # 4b. Seam-run + curve-group geometry checks across allowances
        for sa in SEAM_ALLOWANCES:
            total += 1
            errs = []
            if sa > 0:
                for pname, outline in pieces:
                    errs += _check_seam_runs(outline, sa, f"{name}/{pname}/sa={sa}")
                    errs += _check_curve_groups(outline, rnd._curve_groups(outline),
                                                sa, f"{name}/{pname}/sa={sa}")
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

        # 4e. Exactly what the renderer draws (per-run SA rules, centre-back
        #     override, fold) must stay outside the piece too.
        for sa in [0.5, 0.75, 1.0]:
            total += 1
            front_args, back_args = blk._bodice_svg_args(bk, True, sa)
            errs  = _check_rendered_piece(f"{name}/render-front-fold/sa={sa}",
                                          front_args.pop("outline"), front_args)
            errs += _check_rendered_piece(f"{name}/render-back/sa={sa}",
                                          back_args.pop("outline"), back_args)
            if errs:
                failed += 1
                failure_log.extend(errs)
            else:
                passed += 1

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

        # 5b. Curve (cap) and straight seam allowance offset check
        for sa in [0.375, 0.5, 0.75, 1.0]:
            total += 1
            errs  = _check_curve_groups(sl.outline, [sl.cap_segments], sa,
                                        f"sleeve/{name}/sa={sa}")
            errs += _check_seam_runs(sl.outline, sa, f"sleeve/{name}/sa={sa}")
            if errs:
                failed += 1
                failure_log.extend(errs)
            else:
                passed += 1

        # 5c. Render smoke-test
        for sa in [0.0, 0.5, 0.75]:
            total += 1
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    prefix = os.path.join(tmpdir, "sleeve_test")
                    blk.render_sleeve(sigma, upsilon, omega, xi, psi,
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

    # ── 6. Every pattern, every manifest test size ────────────────────────────
    # Intercept _write_svg so the exact per-piece arguments each pattern
    # passes (SA rules, waist_detect, merge_consecutive, curve groups,
    # notches) are checked, then let the real renderer run.
    real_write_svg = rnd._write_svg
    captured = []

    def capturing_write_svg(path, outline, *args, **kw):
        captured.append((outline, dict(kw)))
        return real_write_svg(path, outline, *args, **kw)

    index = json.load(open(os.path.join(SCRIPT_DIR, "patterns", "index.json")))
    for pid in index["patterns"]:
        mod = importlib.import_module(f"patterns.{pid}")
        manifest = json.load(open(os.path.join(SCRIPT_DIR, "patterns", pid,
                                               "manifest.json")))
        # Patterns import _write_svg by name; patch the module global.
        had_own = hasattr(mod, "_write_svg")
        if had_own:
            mod._write_svg = capturing_write_svg
        rnd._write_svg = capturing_write_svg
        try:
            for size in manifest.get("testSizes", []):
                for sa in [0.5, 0.75, 1.0]:
                    total += 1
                    params = dict(size["values"])
                    for opt in manifest.get("options", []):
                        if opt.get("default") is not None:
                            params[opt["key"]] = opt["default"]
                    params["seam_allowance"] = sa
                    captured.clear()
                    try:
                        mod.render_web(params)
                        errs = []
                        for pi, (outline, kw) in enumerate(captured):
                            errs += _check_rendered_piece(
                                f"{pid}/{size['label']}/piece{pi}/sa={sa}",
                                outline, kw)
                        if not captured:
                            errs.append(f"{pid}/{size['label']}: rendered no pieces")
                        if errs:
                            failed += 1
                            failure_log.extend(errs)
                        else:
                            passed += 1
                    except Exception as e:
                        failed += 1
                        failure_log.append(
                            f"[PATTERN RENDER FAIL] {pid}/{size['label']} sa={sa}: {e}\n"
                            + traceback.format_exc())
        finally:
            rnd._write_svg = real_write_svg
            if had_own:
                mod._write_svg = real_write_svg

    # ── 7. Report ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Seam allowance test suite")
    print(f"  Cases:  {len(TEST_CASES)} bodice + {len(SLEEVE_CASES)} sleeve"
          f" + {len(index['patterns'])} pattern sweep")
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
