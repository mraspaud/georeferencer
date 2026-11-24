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
