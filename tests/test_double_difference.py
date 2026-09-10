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


def test_only_the_coastline_the_swath_actually_covers_is_measured():
    """A coastline runs far beyond any one pass, and most of it is not in the image.

    Measuring a crossing the swath never saw would read whatever happens to sit at
    the edge of the array, so the coastline is cut down to what the pass covers
    before anything is measured.
    """
    from georeferencer.shoreline import coastline_within

    lons = np.tile(np.arange(10.), (5, 1))
    lats = np.tile(np.array([[2.], [1.], [0.], [-1.], [-2.]]), (1, 10))
    coastline = [(4., 1.), (5., 0.), (40., 60.)]

    covered = coastline_within(coastline, lons, lats)

    assert covered == [(4., 1.), (5., 0.)]


def test_a_point_in_the_gap_a_slanted_swath_leaves_is_not_covered():
    """A swath is a band across the globe, not a rectangle drawn around it.

    A pass climbing north-east leaves large empty corners inside the box that
    encloses it. A coastline point sitting in one of those corners was never
    imaged, and judging coverage by the box would hand it a nearest pixel four
    degrees away and read a shoreline off it.
    """
    from georeferencer.shoreline import coastline_within

    lons = np.array([[0., 1., 2.], [10., 11., 12.], [20., 21., 22.]])
    lats = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    on_the_band = (11., 1.)
    in_the_corner = (6., 0.)

    covered = coastline_within([on_the_band, in_the_corner], lons, lats)

    assert covered == [on_the_band]


def test_a_point_between_the_samples_of_a_coarse_swath_is_still_covered():
    """How near is near enough depends on how finely the swath is described.

    Geolocation is often computed on sample points rather than on every pixel, so
    the grid handed to this measurement can be coarse. A point falling midway
    between two samples of such a grid was imaged perfectly well, and a fixed reach
    tuned for a fine grid would throw it away -- discarding real coastline for no
    reason but the sampling of the array it was compared against.
    """
    from georeferencer.shoreline import coastline_within

    lons = np.array([[0., 10., 20.], [0., 10., 20.], [0., 10., 20.]])
    lats = np.array([[0., 0., 0.], [10., 10., 10.], [20., 20., 20.]])
    between_two_samples = (5., 0.)

    covered = coastline_within([between_two_samples], lons, lats)

    assert covered == [between_two_samples]


def test_the_footprint_a_miss_is_judged_against_follows_the_direction_it_was_measured():
    """A crossing is measured across the coast, which points wherever the coast runs.

    The footprint is not round: it is much wider across the track than along it. A
    miss measured across the track must be judged against the wide side and one
    measured along the track against the narrow side, or the same error reads as
    two different fractions of a pixel depending only on how the coastline happened
    to lie.
    """
    from georeferencer.georeferencer import footprint_towards

    along, across = 1100.0, 5000.0

    assert footprint_towards(0.0, along, across) == approx(along)
    assert footprint_towards(np.pi / 2, along, across) == approx(across)


def test_a_coast_running_obliquely_is_judged_against_a_width_between_the_two():
    """Most coasts run neither along the track nor across it.

    The footprint is an ellipse, so its width in an oblique direction is the
    quadrature combination of the two axis widths, not their average. With a
    three-by-four footprint measured at forty-five degrees that is five over root
    two, which a straight average would put at three and a half -- a one per cent
    error on every oblique crossing, and most crossings are oblique.
    """
    from georeferencer.georeferencer import footprint_towards

    assert footprint_towards(np.pi / 4, 3.0, 4.0) == approx(5.0 / np.sqrt(2.0))


def test_a_coastline_is_read_from_a_shapefile(tmp_path):
    """GSHHG ships as shapefiles, which pyshp already reads.

    The coastline arrives as one or more sequences of points. Nothing here parses
    the format: that is a solved problem and reimplementing it would be a second
    reader to keep correct.
    """
    import shapefile

    from georeferencer.shoreline import coastline_from

    path = tmp_path / "coast.shp"
    with shapefile.Writer(str(path)) as writing:
        writing.field("id", "N")
        writing.line([[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]])
        writing.record(1)

    read = coastline_from(str(path))

    assert read == [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]]


def test_a_point_nearest_the_last_sample_of_a_swath_is_still_covered():
    """The reach is asked of the grid, and at the far edge there is no next sample.

    Reading the spacing from the sample one column further east works everywhere
    except the last column, where there is nothing further east and the question
    answers itself with zero. Coverage would then be refused for every point whose
    nearest sample is on that edge -- the whole trailing side of every swath.

    Found on a dateline case, which is how it looks in the data: GSHHG's largest
    polygon begins at longitude exactly 180, so a Pacific pass meets the swath edge
    and the meridian in the same place. The wrap itself is handled; the edge is not.
    """
    from georeferencer.shoreline import coastline_within

    lons = np.tile(np.array([179.5, 179.8, 180.0]), (3, 1))
    lats = np.tile(np.array([[1.], [0.], [-1.]]), (1, 3))
    just_past_the_line = (-179.9, 0.0)

    covered = coastline_within([just_past_the_line], lons, lats)

    assert covered == [just_past_the_line]


def test_the_angle_a_crossing_was_measured_at_is_read_off_the_swath_axes():
    """Crossings are stepped in pixels, and a pixel step is already in track axes.

    The coast normal comes back as a step in lines and columns, and a line is along
    the track while a column is across it. So the angle the footprint has to be
    taken at is simply the angle of that step, with no frame to convert between.

    A crossing stepped purely down the lines was measured along the track, and the
    angle from that axis is zero.
    """
    from georeferencer.shoreline import direction_from_track

    assert direction_from_track(np.array([1.0, 0.0])) == approx(0.0)
