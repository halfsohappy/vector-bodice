"""Render settings for the trouser pattern."""

BACK_STYLE      = dict(fill="#e0d4e8", stroke="#5a3a78")
FRONT_STYLE     = dict(fill="#ece0f0", stroke="#7a4f96")
WAISTBAND_STYLE = dict(fill="#e8e2d5", stroke="#5c5346")

BACK_CORNER_LABELS  = ["H", "N", "O", "C", "Y", "Z", "I", "X"]
FRONT_CORNER_LABELS = ["L", "Q", "C", "U", "V", "M", "X"]
BACK_INTERIOR_LABELS  = ["G", "D", "V", "R", "S"]
FRONT_INTERIOR_LABELS = ["K", "D", "W", "Y", "Z"]

WAISTBAND_OUTLINE_LABELS = ["A", "B", "C", "D"]


def dart_outline_labels(ns):
    """Build outline-label entries for a panel's dart points, using the
    letters the panel itself assigned (see back_panel.py/front_panel.py)."""
    labels = {}
    for i, (leg_in, point, leg_out) in enumerate(ns.dart_points):
        leg_in_l, point_l, leg_out_l = ns.dart_letters[i]
        labels[leg_in_l] = leg_in
        labels[point_l] = point
        labels[leg_out_l] = leg_out
    return labels
