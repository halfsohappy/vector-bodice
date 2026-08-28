"""CLI for the bodice-with-sleeve pattern.

Measurement arguments come from manifest.json (the same source the web
frontend uses).  Render bodice pieces, the sleeve piece, or both:

    python -m patterns.bodice --alpha 28 --beta 36 --gamma 15.5 --delta 5 \
        --epsilon 9 --zeta 7 --eta 14 --theta 14
    python -m patterns.bodice --sigma 23 --upsilon 10 --omega 17 --xi 16 --psi 6
"""

import argparse
import json
from pathlib import Path

from . import render, render_sleeve

_manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
_groups = {g["id"]: g for g in _manifest["measurementGroups"]}
BODICE_ARGS = [(f["key"], f["label"].lower()) for f in _groups["bodice"]["fields"]]
SLEEVE_ARGS = [(f["key"], f["label"].lower()) for f in _groups["sleeve"]["fields"]]

parser = argparse.ArgumentParser(
    prog="python -m patterns.bodice",
    description=f"Render {_manifest['name']} pieces to SVG.")
for name, help_text in BODICE_ARGS + SLEEVE_ARGS:
    parser.add_argument(f"--{name}", type=float, help=help_text)
parser.add_argument("--prefix", type=str, default="bodice",
                    help="output filename prefix")
parser.add_argument("--fold", action="store_true",
                    help="render front bodice on fold (mirrored, full width)")
parser.add_argument("--deepen-bust-dart", action="store_true",
                    help='add 0.5" to the bust dart depth from Chart 1')
parser.add_argument("--seam-allowance", type=float, default=0.75,
                    help="seam allowance in inches (default 0.75)")
args = parser.parse_args()

bodice_vals = [getattr(args, name) for name, _ in BODICE_ARGS]
sleeve_vals = [getattr(args, name) for name, _ in SLEEVE_ARGS]
has_bodice = all(v is not None for v in bodice_vals)
has_sleeve = all(v is not None for v in sleeve_vals)

if not has_bodice and not has_sleeve:
    parser.error("provide all 8 bodice measurements (--alpha … --theta), "
                 "all 5 sleeve measurements (--sigma … --psi), or both")

if has_bodice:
    render(*bodice_vals, prefix=args.prefix, fold=args.fold,
           seam_allowance=args.seam_allowance,
           deepen_bust_dart=args.deepen_bust_dart)
if has_sleeve:
    render_sleeve(*sleeve_vals, prefix=f"{args.prefix}_sleeve" if has_bodice
                  else "sleeve", seam_allowance=args.seam_allowance)
