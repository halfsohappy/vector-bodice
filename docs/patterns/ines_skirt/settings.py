"""Render settings for the INES skirt pattern.

Everything here is presentation, not drafting math: per-piece colours,
which points get labelled, the pattern's seam-allowance rules, and text
annotations. See skirt_panel.py / waistband.py / pocket.py for why each
piece's seam-allowance treatment differs.
"""

import numpy as np

from . import waistband


# ── Per-piece styles ──────────────────────────────────────────────────────────

FRONT_PANEL_STYLE = dict(fill="#f6dfa0", stroke="#8a6d1a")
BACK_PANEL_STYLE  = dict(fill="#eccb72", stroke="#8a6d1a")
WAISTBAND_STYLE   = dict(fill="#e8e2d5", stroke="#5c5346")
POCKET_STYLE      = dict(fill="#f0f0f0", stroke="#888888")
TIE_STYLE         = dict(fill="#f6dfa0", stroke="#8a6d1a")
BIAS_STYLE        = dict(fill="#f0f0f0", stroke="#888888")


# ── Label sets ────────────────────────────────────────────────────────────────
# Which points of the built namespaces get labelled on each piece.
# Outline labels are solid dots with dark text; interior labels are hollow.

PANEL_OUTLINE_LABELS  = ["K", "L", "M", "N"]
PANEL_INTERIOR_LABELS = ["O"]

WAISTBAND_OUTLINE_LABELS = ["A", "B", "C", "D"]
POCKET_OUTLINE_LABELS    = ["A", "B", "C", "D", "E"]
TIE_OUTLINE_LABELS       = ["A", "B", "C", "D"]
BIAS_OUTLINE_LABELS      = ["A", "B", "C", "D"]


# ── Seam-allowance rules ──────────────────────────────────────────────────────

def waistband_seam_allowance_fn(wb, seam_allowance):
    """Per-run seam allowance for the waistband: the two short ends (boxed/
    tunnel corners) always get a fixed technique allowance (0.7cm-equiv,
    waistband.END_SA), independent of the render-time slider; the two long
    edges get the standard value."""
    B, C, D, A = wb.end_points
    def _fn(run):
        p0, p1 = run[0], run[-1]
        if ((np.allclose(p0, B, atol=1e-4) and np.allclose(p1, C, atol=1e-4)) or
                (np.allclose(p0, D, atol=1e-4) and np.allclose(p1, A, atol=1e-4))):
            return waistband.END_SA
        return seam_allowance
    return _fn


# ── Text annotations ──────────────────────────────────────────────────────────

def format_annotations(pairs, color="#aaa", size=None):
    """(text, pos) pairs → the 4-tuples render._write_svg expects."""
    from render import FONT_SIZE
    fsize = size or FONT_SIZE * 1.2
    return [(text, pos, color, fsize) for text, pos in pairs]
