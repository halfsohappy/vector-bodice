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


def dart_outline_labels(ns):
    """Build outline-label entries for a panel's dart points: dNi (leg
    in), dNp (dart point), dNo (leg out), for dart index N."""
    labels = {}
    for i, (leg_in, point, leg_out) in enumerate(ns.dart_points):
        labels[f"d{i}i"] = leg_in
        labels[f"d{i}p"] = point
        labels[f"d{i}o"] = leg_out
    return labels
