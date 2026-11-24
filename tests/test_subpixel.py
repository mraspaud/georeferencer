"""Tests for locating the correlation maximum between pixels (Khlopenkov et al., 2010, section VI)."""

import georeferencer.displacement_calc as dc
import numpy as np
import pytest


def a_peak(cy, cx, size=49, width=1.6):
    """Return a correlation surface scaled to 0-255 like the matcher's, peaking at (cy, cx)."""
    yy, xx = np.mgrid[:size, :size]
    surface = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * width**2))
    return (255 * (surface - surface.min()) / (surface.max() - surface.min())).astype(np.float32)


def located(surface):
    """Return the sub-pixel maximum the matcher finds, as (row, column)."""
    py, px = np.unravel_index(np.argmax(surface), surface.shape)
    x, y = dc.subpixel_maximum(surface, int(py), int(px))
    return y, x


@pytest.mark.parametrize(("cy", "cx"), [(24.1, 24.15), (20.2, 30.0), (30.0, 19.85), (18.9, 27.05)])
def test_a_maximum_near_a_pixel_is_found_from_its_neighbourhood(cy, cx):
    """Close to a grid node, a quadratic through the peak and its eight neighbours places it."""
    y, x = located(a_peak(cy, cx))

    assert y == pytest.approx(cy, abs=0.05)
    assert x == pytest.approx(cx, abs=0.05)


@pytest.mark.parametrize(("cy", "cx"), [(24.0, 30.45), (24.0, 29.55), (18.45, 26.0), (17.55, 26.0)])
def test_a_maximum_between_two_pixels_is_found_from_four_along_that_axis(cy, cx):
    """Between two pixels along one axis, a cubic is fitted to four samples along it, two on each side.

    The two samples nearest the expected maximum sit in the middle, whichever side of
    the peak pixel it lies on.
    """
    y, x = located(a_peak(cy, cx))

    assert y == pytest.approx(cy, abs=0.1)
    assert x == pytest.approx(cx, abs=0.1)


@pytest.mark.parametrize(("cy", "cx"), [(20.4, 30.45), (20.4, 29.6), (19.6, 30.45), (19.6, 29.55)])
def test_a_maximum_between_four_pixels_is_found_from_the_four_by_four_box_around_it(cy, cx):
    """Between four pixels, a cubic surface is fitted to the 4 x 4 box with those four at its centre.

    Off the diagonal and in every quadrant, so neither the axes nor the side can be mixed up.
    """
    y, x = located(a_peak(cy, cx))

    assert y == pytest.approx(cy, abs=0.1)
    assert x == pytest.approx(cx, abs=0.1)
