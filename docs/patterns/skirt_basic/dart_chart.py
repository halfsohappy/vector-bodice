"""Personal Dart Intake Chart — Pattern Making for Fashion Design, 5th ed., p.48.

Keyed on the (whole-number-rounded) difference between total hip and total
waist circumference. Returns dart count and per-dart intake for both front
and back, shared by front_panel.py and back_panel.py (both need the same
chart row, since a single hip-waist difference determines both columns).
"""

from types import SimpleNamespace

# diff -> (front_count, front_intake, back_count, back_intake), all inches
_TABLE = {
    4:  (1, 0.500, 1, 0.750),
    5:  (1, 0.500, 1, 1.000),
    6:  (1, 0.500, 2, 0.625),
    7:  (1, 0.500, 2, 0.750),
    8:  (2, 0.375, 2, 0.875),
    9:  (2, 0.375, 2, 0.875),
    10: (2, 0.500, 2, 1.000),
    11: (2, 0.625, 2, 1.125),
    12: (2, 0.625, 2, 1.250),
    13: (2, 0.625, 2, 1.375),
    14: (2, 0.625, 2, 1.375),
}


def lookup(diff):
    """diff: total hip circumference minus total waist circumference (in).
    Returns SimpleNamespace(front_count, front_intake, back_count, back_intake).
    Clamped to the chart's documented 4-14 inch range."""
    key = min(max(round(diff), 4), 14)
    front_count, front_intake, back_count, back_intake = _TABLE[key]
    return SimpleNamespace(
        front_count=front_count, front_intake=front_intake,
        back_count=back_count, back_intake=back_intake,
    )
