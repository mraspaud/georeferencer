"""Measuring where a swath places the shoreline, against coastlines it was not fitted to."""

import numpy as np

from georeferencer.shoreline import offset_to_shoreline, profile_along


def test_the_shoreline_is_found_where_water_gives_way_to_land():
    """A profile sampled across the coast puts the shore at its steepest step."""
    water_then_land = np.array([0., 0., 0., 0., 0., 1., 1., 1., 1.])

    assert offset_to_shoreline(water_then_land) == 0.5


def test_a_profile_reads_the_image_along_the_given_direction():
    """Stepping east across a coast that runs down the image reads its columns in turn."""
    water_to_the_west = np.tile([0., 0., 0., 1., 1., 1.], (5, 1))

    profile = profile_along(water_to_the_west, at=(2, 2), direction=(0, 1), reach=2)

    np.testing.assert_allclose(profile, [0., 0., 0., 1., 1.])


def test_a_profile_may_be_read_between_pixels():
    """Half-pixel steps read values interpolated between neighbouring columns."""
    water_to_the_west = np.tile([0., 0., 0., 1., 1., 1.], (5, 1))

    profile = profile_along(water_to_the_west, at=(2, 2), direction=(0, 0.5), reach=2)

    np.testing.assert_allclose(profile, [0., 0., 0., 0.5, 1.])
