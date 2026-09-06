"""Tests for judging a displacement field by whether it hangs together."""

import numpy as np

from georeferencer.georeferencer import displacement_scatter


def a_grid(step=64, size=2048):
    """Return points spread over a swath."""
    return [(y, x) for y in range(step, size, step) for x in range(step, size, step)]


def test_a_field_a_rigid_geometry_could_produce_scatters_little():
    """A time or attitude error moves the swath smoothly, so little is left over.

    Every parameter the fit solves for shifts the swath in a way that varies
    smoothly across it: a time offset moves it bodily, a yaw error by an amount
    proportional to the distance from nadir. A field of that shape is evidence of
    a real displacement, whatever its size.
    """
    points = a_grid()
    y = np.array([p[0] for p in points], dtype=float)
    x = np.array([p[1] for p in points], dtype=float)
    across = (x - 1024.0) / 1024.0
    smooth = np.column_stack([3.0 + 7.0 * across, -2.0 + 0.5 * (y / 2048.0)])

    assert displacement_scatter(points, smooth) < 0.1


def test_matched_noise_scatters_widely():
    """Displacements that agree with nothing must read as incoherent.

    A match found in cloud or in featureless desert lands wherever the covariance
    surface happened to peak. No geometry connects one to the next, so a smooth
    field explains none of it.
    """
    points = a_grid()
    rng = np.random.default_rng(0)
    noise = rng.uniform(-24, 24, size=(len(points), 2))

    assert displacement_scatter(points, noise) > 5.0
