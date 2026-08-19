"""CLI for the jean pattern.

Measurement arguments come from manifest.json (the same source the web
frontend uses).

    python -m patterns.jean --waist_arc_front 7 --waist_arc_back 7 \
        --hip_arc_front 9.5 --hip_arc_back 9.5 \
        --crotch_depth 10.5 --pant_length 40 --relaxed_fit
"""

import argparse
import json
from pathlib import Path

from . import render

_manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
FIELDS = _manifest["measurementGroups"][0]["fields"]
OPTIONS = _manifest.get("options", [])

parser = argparse.ArgumentParser(
    prog="python -m patterns.jean",
    description=f"Render {_manifest['name']} pieces to SVG.")
for f in FIELDS:
    parser.add_argument(f"--{f['key']}", type=float, required=True, help=f["title"].lower())
for o in OPTIONS:
    parser.add_argument(f"--{o['key']}", action="store_true", help=o["label"].lower())
parser.add_argument("--prefix", type=str, default="jean",
                    help="output filename prefix")
parser.add_argument("--seam-allowance", type=float, default=0.75,
                    help="seam allowance in inches (default 0.75)")
args = parser.parse_args()

vals = {f["key"]: getattr(args, f["key"]) for f in FIELDS}
opt_vals = {o["key"]: getattr(args, o["key"]) for o in OPTIONS}
render(vals["waist_arc_front"], vals["waist_arc_back"],
       vals["hip_arc_front"], vals["hip_arc_back"],
       vals["crotch_depth"], vals["pant_length"],
       prefix=args.prefix, seam_allowance=args.seam_allowance,
       **opt_vals)
