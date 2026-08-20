"""Render settings for the culotte pattern."""

BACK_STYLE     = dict(fill="#f0d8c8", stroke="#8c4a1a")
FRONT_STYLE    = dict(fill="#f5e3cf", stroke="#a06a2a")
WAISTBAND_STYLE = dict(fill="#e8e2d5", stroke="#5c5346")

BACK_CORNER_LABELS  = ["D", "A", "X", "H", "I", "B"]
FRONT_CORNER_LABELS = ["H", "A", "X", "D", "E", "B"]
BACK_INTERIOR_LABELS  = ["G", "C"]
FRONT_INTERIOR_LABELS = ["I", "C"]

WAISTBAND_OUTLINE_LABELS = ["A", "B", "C", "D"]

# Same letters as skirt_basic's own dart labels — inherited unchanged.
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
