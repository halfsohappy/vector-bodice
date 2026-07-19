"""CLI for the INES skirt pattern.

Measurement arguments come from manifest.json (the same source the web
frontend uses). Measurements are in inches, matching every other pattern
in this repo (the source tutorial is metric — see skirt_panel.py).

    python -m patterns.ines_skirt --waist 28 --skirt_length 27
"""

import argparse
import json
from pathlib import Path

from . import render

_manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
FIELDS = _manifest["measurementGroups"][0]["fields"]

parser = argparse.ArgumentParser(
    prog="python -m patterns.ines_skirt",
    description=f"Render {_manifest['name']} pieces to SVG.")
for f in FIELDS:
    parser.add_argument(f"--{f['key']}", type=float, required=(f["key"] != "hem_allowance"),
                        default=f.get("default") if f["key"] == "hem_allowance" else None,
                        help=f["title"].lower())
parser.add_argument("--prefix", type=str, default="ines_skirt",
                    help="output filename prefix")
parser.add_argument("--seam-allowance", type=float, default=0.75,
                    help="seam allowance in inches (default 0.75)")
args = parser.parse_args()

render(args.waist, args.skirt_length, args.hem_allowance,
       prefix=args.prefix, seam_allowance=args.seam_allowance)
