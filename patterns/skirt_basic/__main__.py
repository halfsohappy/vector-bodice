"""CLI for the basic skirt sloper pattern.

Measurement arguments come from manifest.json (the same source the web
frontend uses).

    python -m patterns.skirt_basic --waist_arc_front 7 --waist_arc_back 7 \
        --hip_arc_front 9.5 --hip_arc_back 9.5 \
        --hip_depth_front 9 --hip_depth_back 9 --skirt_length 24
"""

import argparse
import json
from pathlib import Path

from . import render

_manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
FIELDS = _manifest["measurementGroups"][0]["fields"]

parser = argparse.ArgumentParser(
    prog="python -m patterns.skirt_basic",
    description=f"Render {_manifest['name']} pieces to SVG.")
for f in FIELDS:
    parser.add_argument(f"--{f['key']}", type=float, required=True, help=f["title"].lower())
parser.add_argument("--prefix", type=str, default="skirt_basic",
                    help="output filename prefix")
parser.add_argument("--seam-allowance", type=float, default=0.75,
                    help="seam allowance in inches (default 0.75)")
args = parser.parse_args()

vals = {f["key"]: getattr(args, f["key"]) for f in FIELDS}
render(vals["waist_arc_front"], vals["waist_arc_back"],
       vals["hip_arc_front"], vals["hip_arc_back"],
       vals["hip_depth_front"], vals["hip_depth_back"], vals["skirt_length"],
       prefix=args.prefix, seam_allowance=args.seam_allowance)
