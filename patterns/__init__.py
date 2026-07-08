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
