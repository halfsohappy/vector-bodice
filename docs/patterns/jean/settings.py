"""Render settings for the jean pattern."""

BACK_STYLE      = dict(fill="#dce4f0", stroke="#2c4a78")
FRONT_STYLE     = dict(fill="#e6ecf7", stroke="#4a6a9c")
WAISTBAND_STYLE = dict(fill="#e8e2d5", stroke="#5c5346")

BACK_CORNER_LABELS  = ["H", "N", "O", "C", "Y", "Z", "I", "X"]
FRONT_CORNER_LABELS = ["L", "Q", "R", "C", "U", "V", "M", "X"]
BACK_INTERIOR_LABELS  = ["G", "D", "V", "W", "R", "S"]
FRONT_INTERIOR_LABELS = ["K", "D", "Y", "Z", "T", "W"]

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
