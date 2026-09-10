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
