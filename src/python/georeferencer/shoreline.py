"""Measuring where a swath places the shoreline, against coastlines it was not fitted to."""

import numpy as np
from scipy.ndimage import map_coordinates


def offset_to_shoreline(profile):
    """Return how far the shore lies from the middle of *profile*, in samples.

    The profile is sampled across the coast, so the shore is where water gives
    way to land: the steepest step along it.
    """
    steps = np.abs(np.diff(profile))
    middle = (len(profile) - 1) / 2
    return float(np.argmax(steps)) + 0.5 - middle


def profile_along(image, at, direction, reach):
    """Return the image values along *direction* through *at*, out to *reach* either side."""
    steps = np.arange(-reach, reach + 1)
    rows = at[0] + steps * direction[0]
    columns = at[1] + steps * direction[1]
    return map_coordinates(image, [rows, columns], order=1, mode="nearest")


def swath_pixel_of(lons, lats, point):
    """Return the swath pixel on which *point* falls, as a line and a column."""
    lon, lat = point
    eastwards = (lons - lon + 180) % 360 - 180
    away = np.hypot(eastwards, lats - lat)
    return np.unravel_index(np.argmin(away), away.shape)
