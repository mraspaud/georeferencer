"""Measuring where a swath places the shoreline, against coastlines it was not fitted to."""

import numpy as np

from georeferencer.shoreline import (
    coast_normal,
    offset_to_shoreline,
    profile_along,
    swath_pixel_of,
)


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


def test_a_coastline_point_is_placed_on_the_swath_pixel_it_falls_on():
    """The navigation under test says where on the swath a known position lands."""
    lons = np.array([[10., 11., 12.], [10., 11., 12.]])
    lats = np.array([[60., 60., 60.], [59., 59., 59.]])

    assert swath_pixel_of(lons, lats, (11.1, 58.9)) == (1, 1)


def test_a_point_is_placed_correctly_across_the_dateline():
    """Two degrees across the dateline is nearer than eighty-one degrees away from it."""
    lons = np.array([[179., 100.]])
    lats = np.array([[60., 60.]])

    assert swath_pixel_of(lons, lats, (-179., 60.)) == (0, 0)


def test_a_point_is_placed_correctly_near_the_pole():
    """Ten degrees of longitude at eighty north is a shorter way than five of latitude."""
    lons = np.array([[10., 0.]])
    lats = np.array([[80., 75.]])

    assert swath_pixel_of(lons, lats, (0., 80.)) == (0, 0)


def test_the_coast_is_crossed_at_right_angles_to_itself():
    """A coast running down the swath is crossed by stepping along a line."""
    normal = coast_normal(start=(0, 5), end=(4, 5))

    np.testing.assert_allclose(normal, (0., 1.))
