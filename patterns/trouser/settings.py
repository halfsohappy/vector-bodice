"""Render settings for the trouser pattern."""

BACK_STYLE      = dict(fill="#e0d4e8", stroke="#5a3a78")
FRONT_STYLE     = dict(fill="#ece0f0", stroke="#7a4f96")
WAISTBAND_STYLE = dict(fill="#e8e2d5", stroke="#5c5346")

BACK_CORNER_LABELS  = ["H", "N", "O", "C", "ankle_outseam", "ankle_inseam",
                        "crotch_point", "crotch_top"]
FRONT_CORNER_LABELS = ["L", "Q", "C", "ankle_outseam", "ankle_inseam",
                        "crotch_point", "crotch_top"]
BACK_INTERIOR_LABELS  = ["G", "D", "knee_outseam", "knee_inseam"]
FRONT_INTERIOR_LABELS = ["K", "D", "knee_outseam", "knee_inseam"]

WAISTBAND_OUTLINE_LABELS = ["A", "B", "C", "D"]


def dart_outline_labels(ns):
    """Build outline-label entries for a panel's dart points: dNi (leg
    in), dNp (dart point), dNo (leg out), for dart index N."""
    labels = {}
    for i, (leg_in, point, leg_out) in enumerate(ns.dart_points):
        labels[f"d{i}i"] = leg_in
        labels[f"d{i}p"] = point
        labels[f"d{i}o"] = leg_out
    return labels
