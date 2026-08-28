"""Flared / bell legline development, shared by trouser, slack and jean.

Adapted from Pattern Making for Fashion Design, 5th ed., "Pant with Flared
Leg" (p.612):

    Add approximately 1-inch length to the pant.
    Measure the back crotch level and divide in half.  Measure out equally
    from each side of the creaseline at hem using this measurement.  Mark.
    Draw a line from marks at the hem to a location below, at, or above
    knee level.
    Square in from the flared legline, blending to the grainline at hem.
    Repeat instructions for the front, subtracting 1/2 inch.

Both panels take their flare from the BACK crotch-level width, which is why
patterns/trouser/__init__.py drafts the back first and hands that width to
the front (the front then subtracts 1/2in) — the same "back is drafted from
the front's namespace" arrangement patterns/bodice already uses, just in
the other direction.

Two deliberate simplifications, both flagged in the manifest notes:
  * the book's "location below, at, or above knee level" is unquantified;
    ABOVE/BELOW_OFFSET below fix it at 2in either side of the knee.
  * the hemline is drawn straight between the two ankle points rather than
    squared to the flared legline and blended, matching how every other
    hem in this tool is drawn.
"""

import numpy as np
from types import SimpleNamespace

from geometry import on_line

EXTRA_LENGTH = 1.0      # in, "add approximately 1-inch length to the pant"
FRONT_REDUCTION = 0.5   # in, front flare is 1/2in less than the back's
ABOVE_OFFSET = 2.0      # in, how far above the knee "above knee" starts
BELOW_OFFSET = 2.0      # in, how far below the knee "below knee" starts

POSITIONS = ("above_knee", "at_knee", "below_knee")


def flare_start_y(knee_y, flare_position):
    """y at which the flared legline leaves the original leg."""
    if flare_position == "above_knee":
        return knee_y + ABOVE_OFFSET
    if flare_position == "below_knee":
        return knee_y - BELOW_OFFSET
    return knee_y


def build(hip_side, crotch_point, grain_x, hem_half_plain, ankle_y, knee_y,
          flare_half, flare_position):
    """Develop the flared legline.

    hip_side      — the outseam's upper anchor (panel point C)
    crotch_point  — the inseam's upper anchor (back I / front M)
    grain_x       — x of the creaseline/grainline the flare is measured from
    hem_half_plain— half the unflared hem width, for locating the original legline
    ankle_y       — the unflared ankle height
    knee_y        — knee height
    flare_half    — half the flared hem width (from the book's formula)
    flare_position— one of POSITIONS

    Returns a SimpleNamespace with the new ankle/knee points and the
    intermediate points where the flare begins.
    """
    if flare_position not in POSITIONS:
        raise ValueError(f"flare_position must be one of {POSITIONS}, got {flare_position!r}")

    new_ankle_y = ankle_y - EXTRA_LENGTH
    fy = flare_start_y(knee_y, flare_position)

    # The original (unflared) legline, used only to locate where the flare
    # departs from it.
    plain_out = np.array([grain_x + hem_half_plain, ankle_y])
    plain_in = np.array([grain_x - hem_half_plain, ankle_y])
    flare_out = on_line(hip_side, plain_out, y=fy)
    flare_in = on_line(crotch_point, plain_in, y=fy)

    ankle_out = np.array([grain_x + flare_half, new_ankle_y])
    ankle_in = np.array([grain_x - flare_half, new_ankle_y])

    # Knee notches stay at knee level, on whichever segment spans it.
    def at_knee(upper_anchor, flare_pt, ankle_pt):
        if knee_y >= fy:
            return on_line(upper_anchor, flare_pt, y=knee_y)
        return on_line(flare_pt, ankle_pt, y=knee_y)

    return SimpleNamespace(
        ankle_y=new_ankle_y, flare_y=fy,
        ankle_out=ankle_out, ankle_in=ankle_in,
        flare_out=flare_out, flare_in=flare_in,
        knee_out=at_knee(hip_side, flare_out, ankle_out),
        knee_in=at_knee(crotch_point, flare_in, ankle_in),
    )
