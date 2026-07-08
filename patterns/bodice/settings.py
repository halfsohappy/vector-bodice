"""Render settings for the bodice-with-sleeve pattern.

Everything here is presentation, not drafting math: per-piece colours,
which points get labelled on each piece, hand-tuned label offsets, the
pattern's seam-allowance rules, and large text annotations.
"""

import numpy as np

from render import FONT_SIZE


# ── Per-piece styles ──────────────────────────────────────────────────────────

FRONT_STYLE  = dict(fill="#dac7ff", stroke="#7a6f8a")
BACK_STYLE   = dict(fill="#dce8f5", stroke="#2255aa")
SLEEVE_STYLE = dict(fill="#d5f5e3", stroke="#2d6a4f")


# ── Label sets ────────────────────────────────────────────────────────────────
# Which points of the built namespaces get labelled on each piece.
# Outline labels are solid dots with dark text; interior labels are hollow.

BACK_OUTLINE_LABELS  = ["A", "GG", "AA", "DD", "BB", "O", "FF", "XX", "YY", "ZZ"]
FRONT_OUTLINE_LABELS = ["D", "M", "K", "N", "P", "O", "Q",
                        "T", "V", "W", "UU", "VV", "WW", "S"]
# Construction rectangle corners + reference points, shared by both views
SHARED_INTERIOR_LABELS = ["B", "C", "E", "F", "G", "H", "I", "J",
                          "R", "U", "EE", "CC"]

SLEEVE_OUTLINE_LABELS  = ["K", "G", "M", "N", "E", "O", "P", "H", "L", "Q", "R"]
SLEEVE_INTERIOR_LABELS = ["A", "B", "C", "D", "F", "I", "J"]


# ── Per-label offset tables ───────────────────────────────────────────────────
# All values in model-space inches (dx rightward, dy upward).
# SVG y is flipped in render._label_elements, so dy upward → negative SVG dy.
# Back and front use separate tables because shared points (e.g. O) sit on
# opposite sides of their respective pieces.

BACK_LABEL_OFFSETS = {
    "A":   ( 0.71, -0.71),  # nape (315°)
    "GG":  ( 0.71,  0.71),  # bottom-left corner (45°)
    "AA":  ( 0.42, -0.91),  # above nape, neck RHS (295°)
    "DD":  (-0.87, -0.50),  # shoulder tip (210°)
    "BB":  (-1.00,  0.00),  # upper armhole (180°)
    "O":   (-0.77, -0.64),  # side seam top (220°)
    "FF":  (-0.64,  0.77),  # side seam bottom (130°)
    "XX":  (-0.71,  0.71),  # back dart left base (135°)
    "YY":  ( 0.71,  0.71),  # back dart right base (45°)
    "ZZ":  ( 0.45,  0.00),  # dart tip → right (already optimal)
}

FRONT_LABEL_OFFSETS = {
    "M":   (-0.71, -0.71),  # CF neck corner (225°)
    "D":   (-0.45,  0.38),  # CF waist corner (already optimal)
    "K":   (-0.57, -0.82),  # front neck curve (235°)
    "N":   ( 0.91, -0.42),  # shoulder tip (335°)
    "P":   ( 1.00,  0.00),  # lower armhole transition (0°)
    "O":   ( 0.71, -0.71),  # side seam top (315°)
    "Q":   ( 0.64,  0.77),  # side seam base (50°)
    "V":   ( 0.87, -0.50),  # bust dart lower (330°)
    "T":   ( 0.87,  0.50),  # bust dart upper (30°)
    "UU":  ( 0.64,  0.77),  # bust dart upper base (50°)
    "VV":  (-0.64,  0.77),  # waist dart left base (130°)
    "WW":  ( 0.64,  0.77),  # waist dart right base (50°)
    "W":   (-0.45,  0.00),  # waist dart tip (already optimal)
    "S":   ( 0.40,  0.00),  # bust point (already optimal)
}

INTERIOR_LABEL_OFFSETS = {
    # Construction-rectangle reference points. Offsets chosen so text lands
    # inside whichever piece the dot appears in.
    "B":   ( 0.40,  0.35),  # bottom-left corner → right and up
    "C":   (-0.40, -0.35),  # top-right corner → left and down
    "E":   (-0.35, -0.40),  # top of vertical centerline → left and down
    "F":   ( 0.35,  0.40),  # bottom of vertical centerline → right and up
    "G":   ( 0.40,  0.00),  # left of upper horizontal → right
    "H":   (-0.40,  0.00),  # right of upper horizontal → left
    "I":   ( 0.40,  0.00),  # left of middle horizontal → right
    "J":   (-0.40,  0.00),  # right of middle horizontal → left
    "R":   ( 0.40, -0.25),  # shoulder midpoint → right and slightly down
    "U":   ( 0.40,  0.00),  # bust dart apex → right
    "EE":  ( 0.40,  0.35),  # waist curve reference → right and up
    "CC":  (-0.40, -0.35),  # armhole reference (above BB) → left and down
}

SLEEVE_LABEL_OFFSETS = {
    "K":  ( 0.50,  0.71),   # bottom left → inward
    "G":  ( 0.71,  0.00),   # top left → right (into sleeve)
    "M":  ( 0.71,  0.00),   # cap, left side → right
    "N":  ( 0.00, -0.80),   # cap, left-center → down (into sleeve)
    "E":  ( 0.00, -0.80),   # cap peak → down (into sleeve)
    "O":  ( 0.00, -0.80),   # cap, right-center → down (into sleeve)
    "P":  (-0.71,  0.00),   # cap, right side → left
    "H":  (-0.71,  0.00),   # top right → left (into sleeve)
    "L":  (-0.50,  0.71),   # bottom right → inward
    "Q":  (-0.71,  0.71),   # slit base → inward
    "R":  (-0.71,  0.00),   # slit top → left (into sleeve)
}

SLEEVE_INTERIOR_OFFSETS = {
    "A":  ( 0.40, -0.35),   # top-left rectangle corner
    "B":  ( 0.40,  0.35),   # bottom-left rectangle corner
    "C":  (-0.40, -0.35),   # top-right rectangle corner
    "D":  (-0.40,  0.35),   # bottom-right rectangle corner
    "F":  ( 0.35,  0.40),   # bottom center
    "I":  ( 0.40,  0.00),   # elbow line left
    "J":  (-0.40,  0.00),   # elbow line right
}


# ── Seam-allowance rules ──────────────────────────────────────────────────────

def back_seam_allowance_fn(bk, seam_allowance):
    """Per-run seam allowance for the back piece.
    A→GG (center back seam) must never have SA between 0 and 1 exclusive."""
    center_back_sa = 0.0 if seam_allowance == 0 else max(seam_allowance, 1.0)
    a_pt = bk.A
    def _fn(run):
        # The center-back run [XX, GG, A] ends at A
        if np.allclose(run[-1], a_pt, atol=1e-4):
            return center_back_sa
        return seam_allowance
    return _fn


# ── Text annotations ──────────────────────────────────────────────────────────

def sleeve_text_annotations(sl):
    """Large labels in construction-line colour for the sleeve piece."""
    ann_color = "#aaa"
    ann_size = FONT_SIZE * 2
    ef_x = sl.E[0]  # x of vertical center line EF
    return [
        ("Back of Sleeve",  np.array([sl.H[0] - 1.5, sl.H[1] - 2.0]), ann_color, ann_size),
        ("Front of Sleeve", np.array([sl.G[0] + 1.5, sl.G[1] - 2.0]), ann_color, ann_size),
        ("Elbow Line",      np.array([ef_x - 2.5, sl.I[1] + 0.45]), ann_color, ann_size),
        ("Bicep Line",      np.array([ef_x - 2.5, sl.G[1] + 0.45]), ann_color, ann_size),
    ]
