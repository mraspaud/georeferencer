"""Tests for measuring how far a navigated swath sits from the reference.

See DOUBLE_DIFFERENCE.md for what the measurement is and why it is built this way.
"""

import numpy as np
from pytest import approx


def test_a_footprint_at_nadir_is_square():
    """Looking straight down, the ground seen by one detector is as wide as it is long.

    Both axes reduce to the field of view times the distance, so the only thing that
    can make them differ is the geometry away from nadir.
    """
    from georeferencer.georeferencer import footprint_sizes

    along, across = footprint_sizes(slant_range=833_000.0, local_zenith=0.0,
                                    field_of_view=1.3e-3)

    assert along == approx(across)
    assert along == approx(1083.0, abs=1.0)


def test_a_footprint_stretches_across_track_when_the_ground_tilts_away():
    """Along track only the range matters; across track the tilted ground stretches it.

    This is why the across-track footprint outgrows the along-track one towards the
    swath edge, and why the two axes need separate divisors.
    """
    from georeferencer.georeferencer import footprint_sizes

    along, across = footprint_sizes(slant_range=1.0, local_zenith=np.radians(60.0),
                                    field_of_view=1.0)

    assert along == approx(1.0)
    assert across == approx(2.0)


def test_misses_that_scatter_evenly_report_no_systematic_offset():
    """Accuracy and precision are different numbers and must not be collapsed.

    A set of misses that lands as often one way as the other has no systematic
    offset, however far it scatters. Summarising it by the size of a typical miss
    instead -- the median of the absolute values -- would report the scatter as
    though it were a bias, and would say a perfectly centred navigation was off by
    a whole pixel.
    """
    from georeferencer.georeferencer import summarise_misses

    offset, scatter = summarise_misses(np.array([-1.0, 1.0, -1.0, 1.0]))

    assert offset == approx(0.0)
    assert scatter == approx(1.0)


def test_a_miss_is_reported_against_both_the_spacing_and_the_footprint():
    """The same miss is a different fraction of a sample step than of a footprint.

    Away from nadir the footprint is the larger of the two, so the same miss is a
    smaller fraction of it. Both are reported because they are not interchangeable
    and because earlier figures for this record were quoted against the spacing:
    showing them side by side is what makes the two comparable instead of silently
    swapped.
    """
    from georeferencer.georeferencer import as_pixel_fractions

    fractions = as_pixel_fractions(miss=np.array([550.0]), spacing=np.array([1100.0]),
                                   footprint=np.array([2200.0]))

    assert fractions.of_spacing == approx([0.5])
    assert fractions.of_footprint == approx([0.25])


def test_a_crossing_only_one_image_resolves_is_dropped_from_both():
    """The two terms only cancel if they are measured on the very same crossings.

    Cloud sits over the second crossing in the swath but not in the reference. If
    the reference's measurement of it were kept, it would enter the difference with
    nothing to cancel against, and the coastline's own error -- the very thing this
    construction exists to remove -- would come back in on that crossing alone.
    """
    from georeferencer.shoreline import shoreline_double_difference

    shore = [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]
    further = [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]
    cloud = [0.8] * 10
    swath = np.array([shore, shore, shore, cloud, cloud])
    reference = np.array([shore, shore, shore, further, further])
    lons = np.tile(np.arange(10.), (5, 1))
    lats = np.tile(np.array([[2.], [1.], [0.], [-1.], [-2.]]), (1, 10))
    coastline = [(4., 2.), (4., 0.), (4., -2.)]

    differences = shoreline_double_difference(
        swath, lons, lats, reference, lons, lats, coastline,
        reach=3, least_prominence=0.2)

    assert len(differences) == 1


def test_where_the_coastline_puts_the_shore_does_not_reach_the_answer():
    """The coastline is a common reference, so its own error leaves no trace.

    The same two images are measured against two coastlines that disagree with each
    other about where the shore is. Each disagreement moves both terms by the same
    amount, so the difference between them is untouched. That is the whole reason
    for measuring twice: the coastline may be wrong, and it does not matter.
    """
    from georeferencer.shoreline import shoreline_double_difference

    at_five = [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]
    at_seven = [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]
    swath = np.tile(at_five, (5, 1))
    reference = np.tile(at_seven, (5, 1))
    lons = np.tile(np.arange(10.), (5, 1))
    lats = np.tile(np.array([[2.], [1.], [0.], [-1.], [-2.]]), (1, 10))

    one_coastline = shoreline_double_difference(
        swath, lons, lats, reference, lons, lats, [(4., 1.), (4., -1.)],
        reach=3, least_prominence=0.2)
    another_coastline = shoreline_double_difference(
        swath, lons, lats, reference, lons, lats, [(6., 1.), (6., -1.)],
        reach=3, least_prominence=0.2)

    np.testing.assert_allclose(one_coastline, another_coastline, rtol=1e-9)
