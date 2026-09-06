"""Measuring where a swath places the shoreline, against coastlines it was not fitted to."""

import numpy as np

from georeferencer.shoreline import (
    coast_normal,
    crosses_a_coast,
    ground_step,
    offset_to_shoreline,
    profile_along,
    shoreline_offset,
    shoreline_offsets,
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


def test_a_step_across_the_swath_measures_a_ground_distance():
    """One degree of longitude at the equator is a hundred and eleven kilometres."""
    lons = np.array([[0., 1.]])
    lats = np.array([[0., 0.]])

    step = ground_step(lons, lats, at=(0, 0), direction=(0, 1))

    np.testing.assert_allclose(step, 111319.49, rtol=1e-6)


def test_a_featureless_profile_holds_no_coast():
    """Where cloud covers the coast there is nothing to measure, however steep the noise."""
    all_cloud = np.full(9, 0.8)

    assert not crosses_a_coast(all_cloud, least_prominence=0.2)


def test_a_clear_step_from_water_to_land_holds_a_coast():
    """A step of one, against a demanded contrast of a fifth, is a shore."""
    water_then_land = np.array([0., 0., 0., 0., 0., 1., 1., 1., 1.])

    assert crosses_a_coast(water_then_land, least_prominence=0.2)


def test_a_profile_that_jitters_everywhere_holds_no_coast():
    """Sea ice and broken cloud step as hard as a shore, but they step everywhere."""
    jitter = np.array([0., 0.5, 0., 0.5, 0., 0.5, 0., 0.5, 0.])

    assert not crosses_a_coast(jitter, least_prominence=0.2)


def test_an_offset_is_reported_where_the_swath_and_the_coastline_disagree():
    """The image puts the shore half a degree east of where the coastline says it is."""
    image = np.tile([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.], (5, 1))
    lons = np.tile(np.arange(10.), (5, 1))
    lats = np.tile(np.array([[2.], [1.], [0.], [-1.], [-2.]]), (1, 10))
    coastline = [(4., 1.), (4., -1.)]

    offset = shoreline_offset(image, lons, lats, coastline, reach=2, least_prominence=0.2)

    np.testing.assert_allclose(offset, 55659.745, rtol=1e-5)


def test_every_segment_of_a_coastline_is_measured_on_its_own_geometry():
    """The shore lies four pixels further out along the second segment than the first."""
    near = [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]
    far = [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]
    image = np.array([near, near, near, far, far])
    lons = np.tile(np.arange(10.), (5, 1))
    lats = np.tile(np.array([[2.], [1.], [0.], [-1.], [-2.]]), (1, 10))
    coastline = [(4., 2.), (4., 0.), (4., -2.)]

    offsets = shoreline_offsets(image, lons, lats, coastline, reach=3, least_prominence=0.2)

    np.testing.assert_allclose(offsets[1] / offsets[0], 5.0, rtol=1e-9)


def test_a_crossing_that_shows_no_shore_is_left_out():
    """Cloud over the second crossing leaves only the first to be measured."""
    near = [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]
    cloud = [0.8] * 10
    image = np.array([near, near, near, cloud, cloud])
    lons = np.tile(np.arange(10.), (5, 1))
    lats = np.tile(np.array([[2.], [1.], [0.], [-1.], [-2.]]), (1, 10))
    coastline = [(4., 2.), (4., 0.), (4., -2.)]

    offsets = shoreline_offsets(image, lons, lats, coastline, reach=3, least_prominence=0.2)

    assert len(offsets) == 1
