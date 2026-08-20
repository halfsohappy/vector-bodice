"""Render settings for the basic 2-dart skirt sloper pattern.

Corner-point labels are fixed per panel; dart labels are built dynamically
since dart count varies (1 or 2) with the dart-intake chart lookup.
"""

BACK_STYLE  = dict(fill="#d8e0ef", stroke="#2c4a7c")
FRONT_STYLE = dict(fill="#cfe8e0", stroke="#2d6a58")

BACK_CORNER_LABELS  = ["D", "A", "F", "B"]
FRONT_CORNER_LABELS = ["H", "A", "J", "B"]
BACK_INTERIOR_LABELS  = ["G", "C"]
FRONT_INTERIOR_LABELS = ["I", "C"]


# The book (p.48-50) describes placing darts without assigning each leg/
# point its own letter (unlike the bodice's named T/U/V bust dart) — these
# continue the alphabet from where the panel's own lettering (…D/A/G/C/F/B
# or H/A/I/C/J/B) leaves off, first dart (center-most) then second
# (side-most), matching the book's own drafting order.
_DART_LETTERS = [("R", "S", "T"), ("U", "V", "W")]


def dart_outline_labels(ns):
    """Build outline-label entries for a panel's dart points."""
    labels = {}
    for i, (leg_in, point, leg_out) in enumerate(ns.dart_points):
        leg_in_l, point_l, leg_out_l = _DART_LETTERS[i]
        labels[leg_in_l] = leg_in
        labels[point_l] = point
        labels[leg_out_l] = leg_out
    return labels
