"""Tests for the orthocorrection as it runs in production, through its compiled scan-line routine."""

import numpy as np
import pytest
from georeferencer.georeferencer import _orthocorrect_line

FACTOR = 4


def reference_orthocorrection(elev, dists, tanz, swath_lats, swath_lons, lon_sup, lat_sup):
    """Return what the scan-line orthocorrection must give, by the most direct search there is.

    Each pixel is moved to the furthest supersampled point towards nadir whose terrain
    rises above the line of sight to it; nadir is where the view is most nearly vertical.
    """
    npix = swath_lats.shape[0]
    sup_nadir = int(np.argmin(tanz)) * FACTOR
    out_lat, out_lon = np.empty(npix), np.empty(npix)
    for j in range(npix):
        if tanz[j] == 0:
            out_lat[j], out_lon[j] = swath_lats[j], swath_lons[j]
            continue
        nominal = j * FACTOR
        towards_nadir = range(nominal, sup_nadir) if nominal <= sup_nadir else range(nominal, sup_nadir, -1)
        best = nominal
        for k in towards_nadir:
            if elev[k] > abs(dists[k] - dists[nominal]) / tanz[j]:
                best = k
        out_lat[j], out_lon[j] = lat_sup[best], lon_sup[best]
    return out_lat, out_lon


def a_scan_line(rng, npix=400, spacing=275.0, highest=6000.0, spikes=3):
    """Return a scan line shaped like an AVHRR one: a vertical view somewhere, rough terrain, a few peaks."""
    nadir = rng.integers(npix // 4, 3 * npix // 4)
    angle = np.deg2rad(np.abs(np.arange(npix) - nadir) * 55.37 / 1024 * 2048 / npix)
    tanz = np.tan(np.minimum(angle * 1.2, np.deg2rad(68)))
    elev = np.clip(np.cumsum(rng.normal(0, 60, npix * FACTOR)), 0, None)
    elev = elev / max(elev.max(), 1) * highest * rng.uniform(0.1, 1)
    elev[rng.integers(0, npix * FACTOR, spikes)] = highest
    dists = np.cumsum(np.full(npix * FACTOR, spacing / FACTOR) * (1 + tanz.repeat(FACTOR) ** 2)) - spacing / FACTOR
    lat_sup, lon_sup = np.linspace(-20, -10, npix * FACTOR), np.linspace(-70, -60, npix * FACTOR)
    return elev, dists, tanz, lat_sup[::FACTOR], lon_sup[::FACTOR], lon_sup, lat_sup


@pytest.mark.parametrize("seed", range(25))
def test_the_compiled_scan_line_matches_the_direct_search(seed):
    """Whatever makes the routine fast must leave every pixel exactly where the direct search puts it."""
    elev, dists, tanz, lats, lons, lon_sup, lat_sup = a_scan_line(np.random.default_rng(seed))
    out_lat, out_lon = np.empty(len(lats)), np.empty(len(lats))

    _orthocorrect_line(elev, dists, tanz, lats, lons, lon_sup, lat_sup, out_lat, out_lon)

    expected_lat, expected_lon = reference_orthocorrection(elev, dists, tanz, lats, lons, lon_sup, lat_sup)
    np.testing.assert_array_equal(out_lat, expected_lat)
    np.testing.assert_array_equal(out_lon, expected_lon)


def test_terrain_moves_many_pixels_in_these_lines():
    """The lines above must exercise the search, not leave every pixel where it was."""
    moved = []
    for seed in range(25):
        elev, dists, tanz, lats, lons, lon_sup, lat_sup = a_scan_line(np.random.default_rng(seed))
        out_lat, out_lon = np.empty(len(lats)), np.empty(len(lats))
        _orthocorrect_line(elev, dists, tanz, lats, lons, lon_sup, lat_sup, out_lat, out_lon)
        moved.append(np.mean(out_lon != lons))

    assert np.median(moved) > 0.2
