"""Render settings for the culotte pattern."""

BACK_STYLE     = dict(fill="#f0d8c8", stroke="#8c4a1a")
FRONT_STYLE    = dict(fill="#f5e3cf", stroke="#a06a2a")
WAISTBAND_STYLE = dict(fill="#e8e2d5", stroke="#5c5346")

BACK_CORNER_LABELS  = ["D", "A", "crotch_top", "crotch_out", "new_hem_center", "B"]
FRONT_CORNER_LABELS = ["H", "A", "crotch_top", "crotch_out", "new_hem_center", "B"]
BACK_INTERIOR_LABELS  = ["G", "C"]
FRONT_INTERIOR_LABELS = ["I", "C"]

WAISTBAND_OUTLINE_LABELS = ["A", "B", "C", "D"]


def dart_outline_labels(ns):
    """Build outline-label entries for a panel's remaining dart points."""
    labels = {}
    for i, (leg_in, point, leg_out) in enumerate(ns.dart_points):
        labels[f"d{i}i"] = leg_in
        labels[f"d{i}p"] = point
        labels[f"d{i}o"] = leg_out
    return labels
