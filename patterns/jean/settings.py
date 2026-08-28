"""Render settings for the jean pattern."""

BACK_STYLE      = dict(fill="#dce4f0", stroke="#2c4a78")
FRONT_STYLE     = dict(fill="#e6ecf7", stroke="#4a6a9c")
WAISTBAND_STYLE = dict(fill="#e8e2d5", stroke="#5c5346")

BACK_CORNER_LABELS  = ["T", "O", "C", "Y", "Z", "I", "X"]
FRONT_CORNER_LABELS = ["U", "R", "C", "V", "W", "M", "X"]
BACK_INTERIOR_LABELS  = ["H", "N", "G", "D", "V", "W", "YY", "ZZ"]
FRONT_INTERIOR_LABELS = ["L", "Q", "K", "D", "Y", "Z", "VV", "WW"]

WAISTBAND_OUTLINE_LABELS = ["A", "B", "C", "D"]

# Jean always has exactly one dart per panel — back's leg/tip letters
# (J, P, K) and front's (N, S, O) come from the panel's own build().
_BACK_DART_LETTERS = ("J", "P", "K")
_FRONT_DART_LETTERS = ("N", "S", "O")


def back_dart_outline_labels(ns):
    leg_in, point, leg_out = ns.dart_points[0]
    return dict(zip(_BACK_DART_LETTERS, (leg_in, point, leg_out)))


def front_dart_outline_labels(ns):
    leg_in, point, leg_out = ns.dart_points[0]
    return dict(zip(_FRONT_DART_LETTERS, (leg_in, point, leg_out)))

# Optional design pieces (shared with the trouser foundation)
FLY_STYLE       = dict(fill="#efe4d8", stroke="#8a6a44")
SHIELD_STYLE    = dict(fill="#e8dccd", stroke="#7a5c38")
BELT_LOOP_STYLE = dict(fill="#e4e0d6", stroke="#6a6252")
POCKET_STYLE    = dict(fill="#dfe8e4", stroke="#3d6b58")
