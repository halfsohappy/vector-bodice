"""
test_studio.py — Parity and round-trip suite for the pattern studio.

1. Evaluator parity: the studio project model must reproduce the
   hand-written sleeve piece (patterns/bodice/sleeve.py) exactly, across
   the 5 sleeve test sizes — every point and every cap-curve sample.
2. Op coverage: a bodice-front/back subset exercises circle_h, circle_v,
   along, midpoint, intersect_lines and dist() against the hand-written
   front_bodice.py / back_bodice.py solvers.
3. Codegen round-trip: exporting the sleeve example and importing the
   generated module must give identical coordinates to the evaluator.

Run:  python3 test_studio.py     (exit 0 = all passed)
"""

import json
import sys
import tempfile
import importlib
from pathlib import Path

import numpy as np

import patterns.bodice.sleeve as slv
import patterns.bodice.front_bodice as fbd
import patterns.bodice.back_bodice as bbd
from studio import model, evaluator, codegen

SCRIPT_DIR = Path(__file__).resolve().parent
SLEEVE_EXAMPLE = SCRIPT_DIR / "studio" / "examples" / "sleeve.studio.json"

TOL = 1e-9
failures = []
checks = 0


def check(cond, label):
    global checks
    checks += 1
    if not cond:
        failures.append(label)


def close(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=TOL)


# ── 1. Sleeve parity ──────────────────────────────────────────────────────────

def test_sleeve_parity():
    project = json.loads(SLEEVE_EXAMPLE.read_text())
    check(model.validate(project) == [], "sleeve example: validate clean")

    for size in project["testSizes"]:
        label = size["label"]
        values = size["values"]
        ref = slv.build(**values)
        res = evaluator.evaluate(project, values)
        pc = res["pieces"]["sleeve"]
        check(not pc["errors"], f"sleeve/{label}: no eval errors ({pc['errors'][:1]})")

        for name in "ABCDEFGHIJKLMNOPQR":
            check(close(pc["points"][name], getattr(ref, name)),
                  f"sleeve/{label}: point {name}")

        # cap curve: same number of cubic segments, same samples
        studio_cubics = [e for e in pc["outline"] if e[0] == "cubic_curve"]
        ref_cubics = [e for e in ref.cap_segments]
        check(len(studio_cubics) == len(ref_cubics),
              f"sleeve/{label}: cap segment count")
        ts = np.linspace(0, 1, 17)
        for i, (se, re_) in enumerate(zip(studio_cubics, ref_cubics)):
            for t in ts:
                check(close(se[1](t), re_[1](t)),
                      f"sleeve/{label}: cap seg {i} @t={t:.2f}")

        # outline structure parity (types in order)
        studio_types = [e[0] for e in pc["outline"]]
        ref_types = [e[0] for e in ref.outline]
        check(studio_types == ref_types, f"sleeve/{label}: outline seg types")


# ── 2. Bodice op-coverage subset ──────────────────────────────────────────────

def _bodice_subset_project():
    """Front-bodice neck/shoulder/bust + back shoulder, using the
    construction-op point kinds.  a and b (chart values) are entered as
    measurements since chart lookups live outside the expression model."""
    return {
        "id": "bodice_subset", "name": "bodice subset", "source": "",
        "measurementGroups": [{
            "id": "main", "label": "Main", "required": True,
            "fields": [{"key": k, "label": k} for k in
                       ("alpha", "beta", "gamma", "delta", "epsilon",
                        "zeta", "eta", "theta", "a", "b")],
        }],
        "derived": [
            {"name": "c", "expr": "beta / 2"},
            {"name": "e", "expr": "eta / 2"},
            {"name": "f", "expr": "alpha / 4"},
            {"name": "g", "expr": "zeta / 2"},
            {"name": "h", "expr": "theta / 2"},
        ],
        "options": [], "testSizes": [],
        "pieces": [{
            "id": "front", "label": "Front", "group": None, "fold": None,
            "points": [
                {"name": "K", "kind": "formula", "x": "c - 2.5", "y": "gamma + a + 0.5"},
                {"name": "L", "kind": "formula", "x": "c - 2.5", "y": "gamma + a - 2.5"},
                {"name": "M", "kind": "formula", "x": "c + 0.5", "y": "gamma + a - 2.5"},
                {"name": "N", "kind": "circle_h", "center": "K", "radius": "delta",
                 "y": "gamma + a - 1.375", "branch": "left"},
                {"name": "R", "kind": "midpoint", "p1": "K", "p2": "N"},
                {"name": "S", "kind": "circle_v", "center": "R", "radius": "epsilon",
                 "x": "c + 0.5 - g", "branch": "down"},
                {"name": "O", "kind": "formula", "x": "(c + 0.5) / 2", "y": "0.5 * gamma + a"},
                {"name": "P", "kind": "formula", "x": "c + 0.5 - e", "y": "0.75 * gamma + a"},
                {"name": "Q", "kind": "formula", "x": "c + 0.25 - f - b", "y": "0"},
                {"name": "V", "kind": "formula", "x": "(c + 0.5) / 2", "y": "S.y - 1.5 * a"},
                # tangent-intersection control point for the neck (K vertical, M horizontal)
                {"name": "Kv", "kind": "offset", "from": "K", "dx": "0", "dy": "-1"},
                {"name": "Mh", "kind": "offset", "from": "M", "dx": "-1", "dy": "0"},
                {"name": "CPn", "kind": "intersect_lines", "a1": "K", "a2": "Kv",
                 "b1": "M", "b2": "Mh"},
            ],
            "segments": [], "construction_lines": [],
            "unclipped_construction_lines": [], "dart_lines": [],
            "labels": {"outline": [], "interior": []}, "label_offsets": {},
            "style": {}, "text_annotations": [],
        }, {
            "id": "back", "label": "Back", "group": None, "fold": None,
            "points": [
                {"name": "AA", "kind": "formula", "x": "2.5", "y": "gamma + a + 0.5"},
                {"name": "CC", "kind": "formula", "x": "h", "y": "0.75 * gamma + a + 3"},
                {"name": "DD", "kind": "along", "from": "AA", "toward": "CC",
                 "dist": "delta + 0.5"},
                {"name": "EE", "kind": "formula", "x": "f + b - 0.25", "y": "0"},
                # FF via dist() + sqrt formula (mirrors find_FF)
                {"name": "UD", "kind": "formula", "x": "(c + 0.5) / 2",
                 "y": "front.S.y - a / 2"},
                {"name": "FF", "kind": "formula", "x": "EE.x",
                 "y": "UD.y - sqrt(dist(front.V, front.Q)**2 - (EE.x - UD.x)**2)"},
            ],
            "segments": [], "construction_lines": [],
            "unclipped_construction_lines": [], "dart_lines": [],
            "labels": {"outline": [], "interior": []}, "label_offsets": {},
            "style": {}, "text_annotations": [],
        }],
    }


BODICE_SIZES = [
    dict(alpha=28, beta=36, gamma=15.5, delta=5.0, epsilon=9.0, zeta=7.0, eta=14.0, theta=14.0),
    dict(alpha=24, beta=32, gamma=14.5, delta=4.5, epsilon=8.0, zeta=6.5, eta=12.5, theta=12.5),
    dict(alpha=30, beta=44, gamma=16.0, delta=5.25, epsilon=10.5, zeta=9.0, eta=15.5, theta=15.0),
]


def test_bodice_subset():
    project = _bodice_subset_project()
    check(model.validate(project) == [], "bodice subset: validate clean")

    for size in BODICE_SIZES:
        label = f"beta={size['beta']}"
        front_ref = fbd.build(**size)
        back_ref = bbd.build(front_ref)
        values = {**size, "a": front_ref.a, "b": front_ref.b}
        res = evaluator.evaluate(project, values)
        fr, bk = res["pieces"]["front"], res["pieces"]["back"]
        check(not fr["errors"] and not bk["errors"],
              f"bodice/{label}: no eval errors ({(fr['errors'] + bk['errors'])[:1]})")

        for name in ("K", "L", "M", "N", "R", "S", "O", "P", "Q", "V"):
            check(close(fr["points"][name], getattr(front_ref, name)),
                  f"bodice/{label}: front {name}")
        check(close(fr["points"]["CPn"], front_ref.L),
              f"bodice/{label}: neck control point == L")
        for name in ("AA", "CC", "DD", "EE", "FF"):
            check(close(bk["points"][name], getattr(back_ref, name)),
                  f"bodice/{label}: back {name}")


# ── 3. Codegen round-trip ─────────────────────────────────────────────────────

def test_codegen_roundtrip():
    project = json.loads(SLEEVE_EXAMPLE.read_text())
    files = codegen.generate(project)

    tmp = Path(tempfile.mkdtemp())
    pkg = tmp / project["id"]
    pkg.mkdir()
    for fn, content in files.items():
        (pkg / fn).write_text(content, encoding="utf-8")
    sys.path.insert(0, str(tmp))
    try:
        mod = importlib.import_module(project["id"])
        for size in project["testSizes"]:
            label = size["label"]
            values = size["values"]
            built = mod.build_pieces(values)
            res = evaluator.evaluate(project, values)
            for piece_id, ns in built.items():
                for name, pt in res["pieces"][piece_id]["points"].items():
                    check(close(getattr(ns, name), pt),
                          f"roundtrip/{label}: {piece_id}.{name}")
            r = mod.render_web({**values, "seam_allowance": 0.75})
            check("sleeve" in r and len(r["sleeve"]) > 1000,
                  f"roundtrip/{label}: render_web svg")

        # generated svg should be identical to the hand-written sleeve render
        values = project["testSizes"][0]["values"]
        gen_svg = mod.render_web({**values, "seam_allowance": 0.75})["sleeve"]
        from patterns.bodice import render_sleeve_svg
        ref_svg = render_sleeve_svg(**values, seam_allowance=0.75)["sleeve"]
        check(gen_svg == ref_svg, "roundtrip: generated SVG == hand-written SVG")
    finally:
        sys.path.remove(str(tmp))


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_sleeve_parity()
    test_bodice_subset()
    test_codegen_roundtrip()

    print("=" * 60)
    print("  Pattern studio parity & round-trip suite")
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
    raise SystemExit(0 if not failures else 1)
