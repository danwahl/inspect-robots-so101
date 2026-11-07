"""Read an action chunk back off a shape-preserving cubic through some of its steps.

A chunk carries the motion the policy intends and a step-to-step wobble on top of it.
Keeping ``CONTROL_POINTS`` of its steps and curving between them holds the motion and drops
the wobble. The slopes are Fritsch-Carlson (PCHIP), so no piece leaves the interval its two
control points span, and the curve never swings past a command the policy gave.

The slopes scale with the values, so the result is the same in degrees as in radians.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# How many of a chunk's steps the curve passes through.
CONTROL_POINTS = 13

FloatArray = npt.NDArray[np.float64]


def _end_slope(near: FloatArray, beyond: FloatArray, h_near: float, h_beyond: float) -> FloatArray:
    """The slope at an end knot, from the two secants next to it.

    A one-sided quadratic through the three end knots, held to the direction it starts out
    in: a slope disagreeing in sign with the secant beside it flattens, and one running away
    from a turning point is held to three times that secant.
    """
    slope = ((2 * h_near + h_beyond) * near - h_near * beyond) / (h_near + h_beyond)
    slope = np.where(np.sign(slope) != np.sign(near), 0.0, slope)
    runaway = (np.sign(near) != np.sign(beyond)) & (np.abs(slope) > 3 * np.abs(near))
    return np.where(runaway, 3 * near, slope)


def smooth(chunk: npt.NDArray[np.floating[Any]]) -> FloatArray:
    """``chunk`` (steps, joints) read back off a cubic through ``CONTROL_POINTS`` of its steps.

    The control points are spread evenly from the first step to the last. A chunk with no
    more steps than that passes through every one, so it comes back unchanged.
    """
    values = np.asarray(chunk, dtype=np.float64)
    steps = values.shape[0]
    at = np.unique(np.round(np.linspace(0, steps - 1, CONTROL_POINTS)).astype(int))
    if len(at) == steps:
        return values.copy()
    knots = values[at]
    widths = np.diff(at).astype(np.float64)[:, None]
    secants = np.diff(knots, axis=0) / widths

    before, after = secants[:-1], secants[1:]
    # A knot the path turns at gets a flat slope, which keeps the curve from rounding a
    # corner into an overshoot.
    onward = before * after > 0
    w1, w2 = 2 * widths[1:] + widths[:-1], widths[1:] + 2 * widths[:-1]
    harmonic = (w1 + w2) / (w1 / np.where(onward, before, 1.0) + w2 / np.where(onward, after, 1.0))
    slopes = np.zeros_like(knots)
    slopes[1:-1] = np.where(onward, harmonic, 0.0)
    slopes[0] = _end_slope(secants[0], secants[1], widths[0, 0], widths[1, 0])
    slopes[-1] = _end_slope(secants[-1], secants[-2], widths[-1, 0], widths[-2, 0])

    grid = np.arange(steps, dtype=np.float64)
    piece = np.clip(np.searchsorted(at, grid, side="right") - 1, 0, len(at) - 2)
    width = widths[piece]
    s = (grid - at[piece])[:, None] / width
    s2, s3 = s * s, s * s * s
    out: FloatArray = (
        (2 * s3 - 3 * s2 + 1) * knots[piece]
        + (s3 - 2 * s2 + s) * width * slopes[piece]
        + (-2 * s3 + 3 * s2) * knots[piece + 1]
        + (s3 - s2) * width * slopes[piece + 1]
    )
    return out
