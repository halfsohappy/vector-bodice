"""Render settings for the trouser pattern."""

BACK_STYLE      = dict(fill="#e0d4e8", stroke="#5a3a78")
FRONT_STYLE     = dict(fill="#ece0f0", stroke="#7a4f96")
WAISTBAND_STYLE = dict(fill="#e8e2d5", stroke="#5c5346")

BACK_CORNER_LABELS  = ["S", "O", "C", "Y", "Z", "I", "X"]
FRONT_CORNER_LABELS = ["LL", "Q", "C", "U", "V", "M", "X"]
BACK_INTERIOR_LABELS  = ["H", "N", "G", "D", "V", "YY", "ZZ"]
FRONT_INTERIOR_LABELS = ["L", "K", "D", "W", "UU", "VV"]

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

# Optional design pieces (fly/shield closure, belt loops)
FLY_STYLE       = dict(fill="#efe4d8", stroke="#8a6a44")
SHIELD_STYLE    = dict(fill="#e8dccd", stroke="#7a5c38")
BELT_LOOP_STYLE = dict(fill="#e4e0d6", stroke="#6a6252")
POCKET_STYLE    = dict(fill="#dfe8e4", stroke="#3d6b58")
