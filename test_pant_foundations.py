"""
test_pant_foundations.py — Parity suite for the Trouser / Slack / Jean
foundations (patterns/trouser, patterns/slack, patterns/jean).

Checks the specific claims made in these patterns' docstrings/comments:
  - trouser's rectangle/hip-width/crotch-extension/dart formulas match
    the book's literal equations exactly (these are closed-form, not
    chart lookups, so exact-match tolerance is appropriate)
  - slack drops exactly one dart per panel and applies the book's stated
    waist/hip/crotch/hem tightening offsets on top of trouser
  - jean's relaxed-fit option changes the back crotch extension by
    exactly 1in, and back-pitching lifts the waist row by a positive
    amount that shrinks under the relaxed (longer) extension
  - render_web produces sane, non-degenerate SVGs for all three patterns
    across a spread of sizes

Run:  python3 test_pant_foundations.py     (exit 0 = all passed)
"""

import sys

import numpy as np

from patterns.trouser import front_panel as t_front, back_panel as t_back
from patterns.slack import front_panel as s_front, back_panel as s_back
from patterns.jean import front_panel as j_front, back_panel as j_back
from patterns import trouser, slack, jean

SIZES = [
    dict(waist_arc_front=6.5, waist_arc_back=6.5, hip_arc_front=8.75, hip_arc_back=8.75,
        crotch_depth=10.0, pant_length=38),
    dict(waist_arc_front=7.0, waist_arc_back=7.0, hip_arc_front=9.5, hip_arc_back=9.5,
        crotch_depth=10.5, pant_length=40),
    dict(waist_arc_front=8.5, waist_arc_back=8.5, hip_arc_front=11.5, hip_arc_back=11.5,
        crotch_depth=11.0, pant_length=42),
]

failures = []
checks = 0


def check(cond, label):
    global checks
    checks += 1
    if not cond:
        failures.append(label)


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


# ── 1. Trouser rectangle / width / crotch / dart formulas match the book ──────

def test_trouser_formulas():
    for sz in SIZES:
        bp = t_back.build(sz["hip_arc_back"], sz["waist_arc_back"],
                          sz["crotch_depth"], sz["pant_length"])
        fp = t_front.build(sz["hip_arc_front"], sz["waist_arc_front"],
                           sz["crotch_depth"], sz["pant_length"])

        full_crotch_depth = sz["crotch_depth"] + t_back.CROTCH_EASE
        check(close(bp.hip_depth, full_crotch_depth / 3.0),
              f"trouser/{sz}: hip depth == one-third of A-D")
        check(close(bp.back_width, sz["hip_arc_back"] + t_back.HIP_EASE),
              f"trouser/{sz}: back width == back hip arc + ease")
        check(close(fp.front_width, sz["hip_arc_front"] + t_front.HIP_EASE),
              f"trouser/{sz}: front width == front hip arc + ease")

        check(close(-bp.I[0], 0.5 * bp.back_width),
              f"trouser/{sz}: back crotch extension == one-half of G-D")
        check(close(-fp.M[0], 0.25 * fp.front_width),
              f"trouser/{sz}: front crotch extension == one-fourth of K-D")

        check(bp.n_darts == 2 and close(bp.intake_each, 1.0),
              f"trouser/{sz}: back has 2 darts, 1in intake each")
        check(fp.n_darts == 2 and close(fp.intake_each, 0.5),
              f"trouser/{sz}: front has 2 darts, 0.5in intake each")
        for leg_in, point, leg_out in bp.dart_points:
            check(close(point[1], -t_back.DART_DEPTH),
                  f"trouser/{sz}: back dart depth == 4.5in")
            check(leg_in[0] > bp.N[0] and leg_out[0] < bp.O[0],
                  f"trouser/{sz}: back darts stay within the N-O ease zone")

        check(close(bp.Y[0] - bp.Z[0], 2 * t_back.HEM_HALF),
              f"trouser/{sz}: back hem span == 9in")
        check(close(fp.U[0] - fp.V[0], 2 * t_front.HEM_HALF),
              f"trouser/{sz}: front hem span == 8in")


# ── 2. Slack drops a dart and applies trouser's own tightening offsets ────────

def test_slack_tightening():
    for sz in SIZES:
        tbp = t_back.build(sz["hip_arc_back"], sz["waist_arc_back"],
                           sz["crotch_depth"], sz["pant_length"])
        sbp = s_back.build(sz["hip_arc_back"], sz["waist_arc_back"],
                           sz["crotch_depth"], sz["pant_length"])
        tfp = t_front.build(sz["hip_arc_front"], sz["waist_arc_front"],
                            sz["crotch_depth"], sz["pant_length"])
        sfp = s_front.build(sz["hip_arc_front"], sz["waist_arc_front"],
                            sz["crotch_depth"], sz["pant_length"])

        check(sbp.n_darts == tbp.n_darts - 1, f"slack/{sz}: back drops the side-most dart")
        check(sfp.n_darts == tfp.n_darts - 1, f"slack/{sz}: front drops the side-most dart")

        check(close(tbp.O[0] - sbp.O[0], s_back.WAIST_TRIM),
              f"slack/{sz}: back waist trimmed by the book's stated amount")
        check(close(tbp.C[0] - sbp.C[0], s_back.HIP_TRIM),
              f"slack/{sz}: back hip drawn in by the book's stated amount")
        check(close(sbp.I[0] - tbp.I[0], s_back.CROTCH_TRIM),
              f"slack/{sz}: back crotch extension shortened by the book's stated amount")
        check(close(sbp.Y[0] - sbp.Z[0], 2 * s_back.HEM_HALF),
              f"slack/{sz}: back hem narrowed to 8in")
        check(close(sfp.U[0] - sfp.V[0], 2 * s_front.HEM_HALF),
              f"slack/{sz}: front hem narrowed to 7in")


# ── 3. Jean's relaxed-fit option and back-pitching ─────────────────────────────

def test_jean_fit_and_pitch():
    for sz in SIZES:
        contour = j_back.build(sz["hip_arc_back"], sz["waist_arc_back"],
                               sz["crotch_depth"], sz["pant_length"], relaxed_fit=False)
        relaxed = j_back.build(sz["hip_arc_back"], sz["waist_arc_back"],
                               sz["crotch_depth"], sz["pant_length"], relaxed_fit=True)
        fp = j_front.build(sz["hip_arc_front"], sz["waist_arc_front"],
                           sz["crotch_depth"], sz["pant_length"])

        check(close(-contour.I[0], j_back.CROTCH_EXT_FRAC * contour.back_width),
              f"jean/{sz}: contour-fit back crotch extension == one-fourth of G-D")
        check(close((-relaxed.I[0]) - (-contour.I[0]), 1.0),
              f"jean/{sz}: relaxed fit adds exactly 1in to the back crotch extension")

        check(contour.pitch > 0, f"jean/{sz}: contour fit needs a positive back pitch")
        check(relaxed.pitch >= 0, f"jean/{sz}: relaxed-fit pitch is non-negative")
        check(relaxed.pitch < contour.pitch,
              f"jean/{sz}: relaxed fit (longer extension) needs less pitch than contour fit")
        check(close(contour.H[1], contour.pitch),
              f"jean/{sz}: back waist row lifted by exactly the pitch amount")

        check(close(-fp.M[0], j_front.CROTCH_EXT_FIXED),
              f"jean/{sz}: front crotch extension is the fixed 2in")
        check(contour.n_darts == 1 and fp.n_darts == 1,
              f"jean/{sz}: exactly one dart per panel")


# ── 4. render_web produces sane output across all three patterns ─────────────

def test_render_web_all():
    for sz in SIZES:
        for mod, name in ((trouser, "trouser"), (slack, "slack")):
            out = mod.render_web({**sz, "seam_allowance": 0.75})
            for piece in ("front_panel", "back_panel", "waistband"):
                check(piece in out and len(out[piece]) > 500,
                      f"{name}/{sz}: {piece} present and non-trivial")
                w, h = out[f"{piece}_w"], out[f"{piece}_h"]
                check(w > 0 and h > 0 and np.isfinite(w) and np.isfinite(h),
                      f"{name}/{sz}: {piece} has sane dims ({w}, {h})")

        for relaxed in (False, True):
            out = jean.render_web({**sz, "seam_allowance": 0.75, "relaxed_fit": relaxed})
            for piece in ("front_panel", "back_panel", "waistband"):
                check(piece in out and len(out[piece]) > 500,
                      f"jean/{sz}/relaxed={relaxed}: {piece} present and non-trivial")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_trouser_formulas()
    test_slack_tightening()
    test_jean_fit_and_pitch()
    test_render_web_all()

    print("=" * 60)
    print("  Pant foundations parity suite (trouser / slack / jean)")
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
