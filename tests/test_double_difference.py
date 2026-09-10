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
