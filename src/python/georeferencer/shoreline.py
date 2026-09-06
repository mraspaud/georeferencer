"""Measuring where a swath places the shoreline, against coastlines it was not fitted to."""

import numpy as np


def offset_to_shoreline(profile):
    """Return how far the shore lies from the middle of *profile*, in samples.

    The profile is sampled across the coast, so the shore is where water gives
    way to land: the steepest step along it.
    """
    steps = np.abs(np.diff(profile))
    middle = (len(profile) - 1) / 2
    return float(np.argmax(steps)) + 0.5 - middle
