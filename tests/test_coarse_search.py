"""Tests for finding a displacement too large for the fine search to reach."""

import numpy as np

from georeferencer.georeferencer import estimate_gross_displacement

#: The fine search looks this far around each candidate, in pixels.
FINE_SEARCH_RADIUS = 24


def a_textured_scene(shape=(700, 700), seed=0):
    """Return an image with detail at many scales, as a landscape has."""
    rng = np.random.default_rng(seed)
    scene = np.zeros(shape)
    for scale in (2, 8, 32):
        coarse = rng.normal(size=(shape[0] // scale + 1, shape[1] // scale + 1))
        scene += np.kron(coarse, np.ones((scale, scale)))[:shape[0], :shape[1]] * scale
    return scene


def test_a_coarse_search_reaches_beyond_the_fine_one():
    """A displacement of 80 pixels is out of the fine search's reach, not the coarse one.

    A displacement is reported as the shift that carries the swath onto the
    reference, so a swath rolled 80 rows forward is reported as -80. Getting the
    swath within the fine search's reach is all the coarse pass has to do, so the
    answer is required only to that accuracy.
    """
    scene = a_textured_scene()
    displaced = np.roll(scene, 80, axis=0)
    points = [(y, x) for y in range(100, 600, 100) for x in range(100, 600, 100)]

    found = estimate_gross_displacement(displaced, scene, points, factor=8)

    assert abs(-80 - found[0]) <= FINE_SEARCH_RADIUS
    assert abs(found[1]) <= FINE_SEARCH_RADIUS
