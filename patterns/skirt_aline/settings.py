"""Render settings for the A-line skirt pattern."""

BACK_STYLE  = dict(fill="#e6d8ef", stroke="#5c2c7c")
FRONT_STYLE = dict(fill="#efe0cf", stroke="#7c5a2d")

BACK_CORNER_LABELS  = ["D", "A", "F", "B"]
FRONT_CORNER_LABELS = ["H", "A", "J", "B"]
BACK_INTERIOR_LABELS  = ["G", "C"]
FRONT_INTERIOR_LABELS = ["I", "C"]


# Same letters as skirt_basic's own dart labels — the remaining dart here
# is the same physical point, just possibly missing its side-most sibling
# (transferred to the hem flare).
_DART_LETTERS = [("R", "S", "T"), ("U", "V", "W")]


def dart_outline_labels(ns):
    """Build outline-label entries for a panel's remaining dart points."""
    labels = {}
    for i, (leg_in, point, leg_out) in enumerate(ns.dart_points):
        leg_in_l, point_l, leg_out_l = _DART_LETTERS[i]
        labels[leg_in_l] = leg_in
        labels[point_l] = point
        labels[leg_out_l] = leg_out
    return labels
