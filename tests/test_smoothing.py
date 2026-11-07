import numpy as np

from inspect_robots_so101.smoothing import CONTROL_POINTS, smooth


def _wobbly(steps: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    motion = np.linspace(0.0, 30.0, steps)[:, None] * np.linspace(-1.0, 1.0, 6)
    return motion + rng.normal(scale=0.5, size=(steps, 6))


def test_passes_through_its_control_points() -> None:
    chunk = _wobbly(50)
    at = np.unique(np.round(np.linspace(0, 49, CONTROL_POINTS)).astype(int))
    assert np.allclose(smooth(chunk)[at], chunk[at])


def test_removes_the_wobble() -> None:
    chunk = _wobbly(50)
    jerk = lambda c: np.abs(np.diff(c, 2, axis=0)).mean()  # noqa: E731
    assert jerk(smooth(chunk)) < jerk(chunk) / 3


def test_never_leaves_the_interval_its_control_points_span() -> None:
    # A step up and back down: a C2 spline would swing past both plateaus.
    chunk = np.repeat(np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0])[:, None], 6, axis=1)
    chunk = np.repeat(chunk, 10, axis=0)
    out = smooth(chunk)
    assert out.min() >= -1e-9 and out.max() <= 10.0 + 1e-9


def test_holds_a_runaway_end_slope() -> None:
    # The end knot sits just before a turn, where the one-sided quadratic overshoots.
    chunk = np.zeros((25, 1))
    chunk[2:] = 1.0
    chunk[4:] = -5.0
    out = smooth(chunk)
    assert out.min() >= -5.0 - 1e-9 and out.max() <= 1.0 + 1e-9


def test_a_short_chunk_comes_back_unchanged() -> None:
    chunk = _wobbly(CONTROL_POINTS)
    assert np.array_equal(smooth(chunk), chunk)
