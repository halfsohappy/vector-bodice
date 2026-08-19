"""Render settings for the A-line skirt pattern."""

BACK_STYLE  = dict(fill="#e6d8ef", stroke="#5c2c7c")
FRONT_STYLE = dict(fill="#efe0cf", stroke="#7c5a2d")

BACK_CORNER_LABELS  = ["D", "A", "F", "B"]
FRONT_CORNER_LABELS = ["H", "A", "J", "B"]
BACK_INTERIOR_LABELS  = ["G", "C"]
FRONT_INTERIOR_LABELS = ["I", "C"]


def dart_outline_labels(ns):
    """Build outline-label entries for a panel's remaining dart points."""
    labels = {}
    for i, (leg_in, point, leg_out) in enumerate(ns.dart_points):
        labels[f"d{i}i"] = leg_in
        labels[f"d{i}p"] = point
        labels[f"d{i}o"] = leg_out
    return labels
