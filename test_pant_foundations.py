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

from geometry import curve_length
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
            # measured down from the waistline (the legs' midpoint sits on
            # it), not from y=0 — the waistline is slightly pitched
            waist_at_dart = 0.5 * (float(leg_in[1]) + float(leg_out[1]))
            check(close(waist_at_dart - float(point[1]), t_back.DART_DEPTH),
                  f"trouser/{sz}: back dart depth == 4.5in below the waistline")
            check(leg_in[0] > bp.N[0] and leg_out[0] < bp.O[0],
                  f"trouser/{sz}: back darts stay within the N-O ease zone")

        check(close(bp.Y[0] - bp.Z[0], 2 * t_back.HEM_HALF),
              f"trouser/{sz}: back hem span == 9in")
        check(close(fp.U[0] - fp.V[0], 2 * t_front.HEM_HALF),
              f"trouser/{sz}: front hem span == 8in")

        # Centre back is pitched: the waistline starts at S (up from N), and
        # the H-N corner is construction only — including it would make the
        # back waist a full CENTER_BACK_OFFSET too wide.
        check(close(bp.S[0], bp.N[0]) and bp.S[1] > bp.N[1],
              f"trouser/{sz}: back centre-back point S is pitched up from N")
        check(bp.outline[-1][2] is bp.S,
              f"trouser/{sz}: centre back seam ends at S, not the H corner")


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

        # The centre back is pitched IN (H-N) and UP (N-T) by the book's own
        # fixed amounts; the side waist O is the hinge and must stay on the
        # waist row, or the back outseam ends up longer than the front.
        check(close(contour.pitch, j_back.CB_PITCH_UP),
              f"jean/{sz}: back pitch is the book's fixed 1in N-T lift")
        check(close(contour.T[0], contour.N[0])
              and close(contour.T[1] - contour.N[1], j_back.CB_PITCH_UP),
              f"jean/{sz}: T is squared up from N by the pitch amount")
        check(close(contour.O[1], 0.0),
              f"jean/{sz}: side waist O stays on the waist row (pitch hinges there)")
        check(close(contour.H[1], 0.0),
              f"jean/{sz}: H is a plain rectangle corner, not lifted")
        check(close(relaxed.pitch, contour.pitch),
              f"jean/{sz}: relaxed fit changes the crotch extension, not the pitch")
        check(contour.outline[-1][2] is contour.T,
              f"jean/{sz}: centre back seam ends at T, not the H corner")

        check(close(-fp.M[0], j_front.CROTCH_EXT_FIXED),
              f"jean/{sz}: front crotch extension is the fixed 2in")
        check(contour.n_darts == 1 and fp.n_darts == 1,
              f"jean/{sz}: exactly one dart per panel")


# ── 4. Waist totals, waistband match, and seam walking ───────────────────────

def _seg_len(seg):
    if seg[0] in ("line", "dart"):
        return float(np.linalg.norm(np.asarray(seg[2], float) - np.asarray(seg[1], float)))
    if seg[0] == "cubic_curve":
        return curve_length(seg[1])
    return 0.0


def _waist_span(ns):
    """Horizontal width the panel's waistline covers with the darts folded
    out — the measurement the book's formulas are stated in.

    The waistline is everything before the hip curve (the first cubic), and
    its "line" segments are exactly the gaps between the darts, so summing
    their x-extent already excludes the dart intakes.  Pleats are NOT cut
    out of the edge (the waistline runs straight across them), so their
    fullness has to be subtracted separately."""
    span = 0.0
    for seg in ns.outline:
        if seg[0] == "cubic_curve":
            break
        if seg[0] == "line":
            span += abs(float(seg[2][0]) - float(seg[1][0]))
    for a, b in getattr(ns, "pleats", []):
        span -= abs(float(b[0]) - float(a[0]))
    return span


def test_waist_band_and_seam_walk():
    """The drafted waist must equal the book's (body + 1in ease, i.e. waist
    arc + 1/4in per quarter), the waistband must match the pant exactly, and
    front/back seams must walk within a normal truing tolerance."""
    for sz in SIZES:
        body = 2 * (sz["waist_arc_front"] + sz["waist_arc_back"])
        expected = body + 1.0          # book: 1/4in ease on each of 4 quarters
        for name, mod in (("trouser", trouser), ("slack", slack), ("jean", jean)):
            pieces = mod.build(**sz)
            fp, bp, band = pieces["front_panel"], pieces["back_panel"], pieces["waistband"]
            total = 2 * (fp.finished_waist + bp.finished_waist)

            # The *horizontal* span is the patternmaking measurement and must
            # hit the book's number exactly. (The sewn length along the seam
            # is a hair longer because the pitched waistline is tilted.)
            span = 2 * (_waist_span(fp) + _waist_span(bp))
            check(abs(span - expected) < 1e-9,
                  f"{name}/{sz}: waist span {span:.4f} == body + 1in ease ({expected})")
            # The band is cut to the sewn length, so that match must be exact.
            check(abs(total - band.span) < 1e-9,
                  f"{name}/{sz}: waistband span {band.span:.4f} matches pant waist {total:.4f}")

            f_out = _seg_len(fp.outline[-6]) + _seg_len(fp.outline[-5])
            b_out = _seg_len(bp.outline[-6]) + _seg_len(bp.outline[-5])
            f_in, b_in = _seg_len(fp.outline[-3]), _seg_len(bp.outline[-3])
            check(abs(f_out - b_out) < 0.5,
                  f"{name}/{sz}: outseams walk within 1/2in (got {abs(f_out-b_out):.3f})")
            check(abs(f_in - b_in) < 0.5,
                  f"{name}/{sz}: inseams walk within 1/2in (got {abs(f_in-b_in):.3f})")


# ── 5. Alignment notches ─────────────────────────────────────────────────────

def test_notches():
    for sz in SIZES:
        for name, mod in (("trouser", trouser), ("slack", slack), ("jean", jean)):
            pieces = mod.build(**sz)
            fp, bp, band = pieces["front_panel"], pieces["back_panel"], pieces["waistband"]
            for lbl, ns in (("front", fp), ("back", bp)):
                check(len(ns.notches) == 3,
                      f"{name}/{sz}: {lbl} has hip + two knee notches")
            check(len(band.notches) == 4,
                  f"{name}/{sz}: waistband notched at 2 side seams, centre back, centre front")
            # front and back knee notches must sit at the same height so they
            # actually line up when the leg seams are sewn
            check(abs(fp.notches[1][1] - bp.notches[1][1]) < 1e-6,
                  f"{name}/{sz}: front/back outseam knee notches at matching height")
            check(abs(fp.notches[2][1] - bp.notches[2][1]) < 1e-6,
                  f"{name}/{sz}: front/back inseam knee notches at matching height")
            # band notches must be strictly increasing and land inside the span
            xs = [float(n[0]) for n in band.notches]
            check(all(b > a for a, b in zip(xs, xs[1:])) and xs[-1] <= band.span + 1e-9,
                  f"{name}/{sz}: waistband notches ordered and within the waist span")


# ── 6. render_web produces sane output across all three patterns ─────────────

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


# ── 6. Design options: fly, belt loops, pockets, flared leg, pleats ──────────

def test_addon_rectangles():
    """Fly, shield and belt-loop strip are the book's exact stated sizes,
    and each is detected as a ruler-draftable rectangle."""
    from patterns.trouser import fly, belt_loops
    from render import rectangle_dims
    f, sh, bl = fly.build_fly(), fly.build_shield(), belt_loops.build()
    for ns, w, h, label in ((f, 1.5, 7.5, "fly"),
                            (sh, 2.5, 8.5, "shield"),
                            (bl, 13.5, 0.75, "belt-loop strip")):
        check(close(ns.width if label != "belt-loop strip" else ns.length, w),
              f"{label}: width {w}in per the book")
        check(close(ns.height if label != "belt-loop strip" else ns.width, h),
              f"{label}: height {h}in per the book")
        rect = rectangle_dims(ns.outline, 0.5, None, waist_detect=False, merge_consecutive=False)
        check(rect is not None, f"{label}: reported as a plain rectangle")
    check(bl.loop_count == 6, "belt-loop strip: cut into six loops")


def test_optional_pieces_appear_only_when_asked():
    sz = SIZES[1]
    for name, mod in (("trouser", trouser), ("slack", slack), ("jean", jean)):
        plain = set(mod.build(**sz))
        for opt, expected in (("fly_front", {"fly", "shield"}),
                              ("belt_loops_opt", {"belt_loops"}),
                              ("pockets", {"pocket_facing", "pocket_pouch",
                                           "pocket_backing", "pocket_lining"})):
            got = set(mod.build(**sz, **{opt: True}))
            check(got - plain == expected,
                  f"{name}/{opt}: adds exactly {sorted(expected)}")
            check(not (plain - got), f"{name}/{opt}: keeps the base pieces")


def test_flared_leg():
    from patterns.trouser import legline
    for sz in SIZES:
        for name, mod in (("trouser", trouser), ("slack", slack), ("jean", jean)):
            plain = mod.build(**sz)
            for pos in legline.POSITIONS:
                fl = mod.build(**sz, flared_leg=True, flare_position=pos)
                pf, pb = plain["front_panel"], plain["back_panel"]
                ff, fb = fl["front_panel"], fl["back_panel"]

                back_hem = float(fb.Y[0] - fb.Z[0])
                front_hem = float(_hem_span(ff))
                check(abs(back_hem - fb.crotch_width) < 1e-9,
                      f"{name}/{pos}: back hem == back crotch-level width")
                check(abs(front_hem - (fb.crotch_width - 2 * legline.FRONT_REDUCTION)) < 1e-9,
                      f"{name}/{pos}: front hem is 1in narrower than the back")
                check(back_hem > float(pb.Y[0] - pb.Z[0]),
                      f"{name}/{pos}: flared hem is wider than the plain draft")
                check(abs(float(fb.Y[1]) - (float(pb.Y[1]) - legline.EXTRA_LENGTH)) < 1e-9,
                      f"{name}/{pos}: the book's 1in extra length is added")
                check(abs(float(ff.notches[1][1]) - float(fb.notches[1][1])) < 1e-9,
                      f"{name}/{pos}: knee notches still line up front to back")


def _hem_span(ns):
    """Ankle width of a front panel, whichever letters it uses."""
    if hasattr(ns, "U") and hasattr(ns, "V") and abs(float(ns.U[1]) - float(ns.V[1])) < 1e-9:
        return float(ns.U[0] - ns.V[0])
    return float(ns.V[0] - ns.W[0])


def test_pleats_preserve_the_waist():
    """Pleat fullness folds out before the band goes on, so the waist span
    and the waistband must be untouched — the regression that would bite
    hardest if the shear ever drifted."""
    for sz in SIZES:
        plain = trouser.build(**sz)
        for depth in (1.0, 2.0, 3.0):
            pl = trouser.build(**sz, pleated_front=True, pleat_depth=depth)
            pf, ff = plain["front_panel"], pl["front_panel"]
            check(abs(_waist_span(ff) - _waist_span(pf)) < 1e-9,
                  f"trouser/{sz}/pleat {depth}: waist span unchanged")
            check(ff.n_pleats == 2 and ff.n_darts == 0,
                  f"trouser/{sz}/pleat {depth}: two pleats replace both darts")
            check(abs(float(ff.Q[0] - ff.LL[0])
                      - float(pf.Q[0] - pf.LL[0]) - depth) < 1e-9,
                  f"trouser/{sz}/pleat {depth}: panel widened by exactly the pleat depth")
            total = 2 * (ff.finished_waist + pl["back_panel"].finished_waist)
            check(abs(total - pl["waistband"].span) < 1e-9,
                  f"trouser/{sz}/pleat {depth}: waistband still matches the pant")


def test_pockets():
    from patterns.trouser import pocket
    sz = SIZES[1]
    for name, mod in (("trouser", trouser), ("slack", slack), ("jean", jean)):
        pieces = mod.build(**sz, pockets=True)
        panel = pieces["front_panel"]
        for pid in ("pocket_facing", "pocket_pouch", "pocket_backing", "pocket_lining"):
            ns = pieces[pid]
            pts = [np.asarray(s[1], float) for s in ns.outline]
            area = 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
                                 - pts[(i + 1) % len(pts)][0] * pts[i][1]
                                 for i in range(len(pts))))
            check(area > 1.0, f"{name}/{pid}: non-degenerate ({area:.2f}in2)")
        f = pieces["pocket_facing"]
        check(abs(float(np.linalg.norm(f.C - panel.side_waist)) - pocket.ENTRY_IN) < 1e-6,
              f"{name}: entry starts 1 3/4in from the side waist")
        # entry edge takes the tighter 1/4in allowance
        fn = pocket.entry_seam_allowance_fn(f, 0.5)
        check(fn is not None and close(fn(np.array([f.C, f.D])), pocket.ENTRY_SA),
              f"{name}: pocket entry gets the 1/4in seam allowance")
        check(close(fn(np.array([f.E, f.F])), 0.5),
              f"{name}: other pocket edges get the standard allowance")


def test_box_pleated_culotte():
    from patterns import culotte
    from patterns.culotte import front_panel as cfp
    sz = dict(waist_arc_front=7.0, waist_arc_back=7.0, hip_arc_front=9.5, hip_arc_back=9.5,
              hip_depth_front=9.0, hip_depth_back=9.0, crotch_depth=10.5, skirt_length=24)
    plain = culotte.build(**sz)["front_panel"]
    boxed = culotte.build(**sz, box_pleat=True)["front_panel"]
    flared = culotte.build(**sz, box_pleat=True, box_pleat_flare=True)["front_panel"]
    check(abs(boxed.finished_waist_span - plain.finished_waist_span) < 1e-9,
          "culotte box pleat: finished waist span unchanged")
    check(abs(float(boxed.A[0] - boxed.H[0]) - float(plain.A[0] - plain.H[0])
              - cfp.BOX_PLEAT_SHIFT) < 1e-9,
          "culotte box pleat: panel widened by the book's 5in")
    check(boxed.n_pleats == 1 and close(boxed.pleat_intake, cfp.BOX_PLEAT_SHIFT),
          "culotte box pleat: one pleat taking the full 5in")
    check(float(flared.E[0]) < float(boxed.E[0]),
          "culotte box pleat: the optional flare swings the CF hem further out")


def test_creaseline_flare():
    """The back splits in two at the creaseline; the two new seam edges
    must be the same length or the halves will not sew together."""
    from patterns.slack import creaseline
    for sz in SIZES:
        plain = slack.build(**sz)
        cf = slack.build(**sz, creaseline_flare=True)
        check("back_panel" in plain and "back_panel" not in cf,
              f"slack/{sz}: the plain back is replaced when contoured")
        check({"back_side", "back_inner"} <= set(cf),
              f"slack/{sz}: back is cut as two pieces")

        side, inner = cf["back_side"], cf["back_inner"]
        side_seam = [g for g in side.outline if g[0] == "cubic_curve"][-4:]
        inner_seam = [g for g in inner.outline if g[0] == "cubic_curve"][:4]
        ls = sum(curve_length(g[1]) for g in side_seam)
        li = sum(curve_length(g[1]) for g in inner_seam)
        check(abs(ls - li) < 1e-6,
              f"slack/{sz}: the two creaseline edges match ({ls:.4f} vs {li:.4f})")

        # the contour scoops in by the book's 1/2in on each half, mirrored
        crease_x = float(plain["back_panel"].V[0])
        check(close(float(side.A[0]) - crease_x, creaseline.CONTOUR_IN),
              f"slack/{sz}: side half scoops 1/2in in at crotch level")
        check(close(crease_x - float(inner.A[0]), creaseline.CONTOUR_IN),
              f"slack/{sz}: inner half scoops the same amount, mirrored")

        # hem drops more at the new seam than at the side seam
        check(float(side.hem_seam[1]) < float(side.hem_out[1]),
              f"slack/{sz}: hem drops lower at the creaseline seam")


def test_manifest_lists_every_module():
    """Every .py a pattern folder contains must appear in its manifest's
    "files" list — that list is exactly what the web frontend fetches into
    the Pyodide filesystem, so anything missing imports fine locally and
    then dies with an ImportError in the browser (which is how both
    slack/settings.py and slack/creaseline.py first broke the site)."""
    import json, pathlib
    for mdir in sorted(pathlib.Path("patterns").iterdir()):
        mf = mdir / "manifest.json"
        if not mf.is_file():
            continue
        listed = set(json.loads(mf.read_text())["files"])
        on_disk = {f.name for f in mdir.glob("*.py") if f.name != "__main__.py"}
        missing = sorted(on_disk - listed)
        check(not missing,
              f"{mdir.name}/manifest.json lists every module (missing: {missing})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_trouser_formulas()
    test_slack_tightening()
    test_jean_fit_and_pitch()
    test_waist_band_and_seam_walk()
    test_notches()
    test_render_web_all()
    test_addon_rectangles()
    test_optional_pieces_appear_only_when_asked()
    test_flared_leg()
    test_pleats_preserve_the_waist()
    test_pockets()
    test_box_pleated_culotte()
    test_creaseline_flare()
    test_manifest_lists_every_module()

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
