"""Measuring where a swath places the shoreline, against coastlines it was not fitted to."""

import numpy as np

from georeferencer.shoreline import offset_to_shoreline


def test_the_shoreline_is_found_where_water_gives_way_to_land():
    """A profile sampled across the coast puts the shore at its steepest step."""
    water_then_land = np.array([0., 0., 0., 0., 0., 1., 1., 1., 1.])

    assert offset_to_shoreline(water_then_land) == 0.5
