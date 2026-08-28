"""Pattern library.

Each subfolder is one pattern.  A pattern folder contains:

  * one file per pattern piece (e.g. front_bodice.py, back_bodice.py,
    sleeve.py) holding the drafting steps for that piece: every point
    expressed as a function of, or offset from, the body measurements, and
    an outline saying how the points are joined — ("line", P0, P1),
    ("dart", P0, P1), ("quadratic", P0, CP, P1), or
    ("cubic_curve", func, P0, P1);
  * settings.py — render settings for the pattern as a whole and for each
    piece (fills/strokes, label sets, label offset tables, seam-allowance
    rules, text annotations);
  * __init__.py — assembles the pieces and exposes build/render entry points
    that feed the master renderer (render.py at the repo root).

To add a new pattern, copy this folder shape and give the new pieces
build() functions that return outlines in the segment format above.
"""


# ── manifest option helpers (shared by every pattern's __main__.py) ───────────
# Options come in three types, matching what the web frontend renders:
#   "checkbox" (the default) → store_true flag
#   "choice"                 → --key VALUE, restricted to the listed choices
#   "number"                 → --key FLOAT
# Keeping this here means a pattern's CLI picks up new option types for free.

def add_option_args(parser, options):
    """Add one argparse flag per manifest option."""
    for o in options:
        key, otype = o["key"], o.get("type", "checkbox")
        helptext = o["label"].lower()
        if otype == "checkbox":
            parser.add_argument(f"--{key}", action="store_true", help=helptext)
        elif otype == "choice":
            parser.add_argument(f"--{key}", type=str, default=o.get("default"),
                                choices=[c["value"] for c in o.get("choices", [])],
                                help=helptext)
        else:
            parser.add_argument(f"--{key}", type=float, default=o.get("default"),
                                help=helptext)


def collect_options(args, options):
    """Read the parsed option values back out as a {key: value} dict."""
    return {o["key"]: getattr(args, o["key"]) for o in options}
