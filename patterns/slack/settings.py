"""Render settings for the slack pattern."""

BACK_STYLE      = dict(fill="#d8e8e0", stroke="#2c6a4a")
FRONT_STYLE     = dict(fill="#e0f0e8", stroke="#3a8a60")
WAISTBAND_STYLE = dict(fill="#e8e2d5", stroke="#5c5346")

BACK_CORNER_LABELS  = ["S", "O", "C", "Y", "Z", "I", "X"]
FRONT_CORNER_LABELS = ["LL", "Q", "C", "U", "V", "M", "X"]
BACK_INTERIOR_LABELS  = ["H", "N", "G", "D", "V", "YY", "ZZ"]
FRONT_INTERIOR_LABELS = ["L", "K", "D", "W", "UU", "VV"]

WAISTBAND_OUTLINE_LABELS = ["A", "B", "C", "D"]


def dart_outline_labels(ns):
    """Build outline-label entries for a panel's dart points, using the
    letters trouser's panel itself assigned (carried through unchanged)."""
    labels = {}
    for i, (leg_in, point, leg_out) in enumerate(ns.dart_points):
        leg_in_l, point_l, leg_out_l = ns.dart_letters[i]
        labels[leg_in_l] = leg_in
        labels[point_l] = point
        labels[leg_out_l] = leg_out
    return labels

# Optional design pieces (shared with the trouser foundation)
FLY_STYLE       = dict(fill="#efe4d8", stroke="#8a6a44")
SHIELD_STYLE    = dict(fill="#e8dccd", stroke="#7a5c38")
BELT_LOOP_STYLE = dict(fill="#e4e0d6", stroke="#6a6252")
POCKET_STYLE    = dict(fill="#dfe8e4", stroke="#3d6b58")
