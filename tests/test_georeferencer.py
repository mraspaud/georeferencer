"""Tests for georeferencer."""

import numpy as np
from georeferencer.georeferencer import _orthocorrect_line_py, _process_scanline_py


def make_supersampled(swath, factor=4):
    """Repeat each element in swath 'factor' times to simulate supersampling."""
    return np.repeat(swath, factor)


def test_flat_dem_identity():
    """Test with flat elevation."""
    n = 6
    factor = 4
    swath_lats = np.linspace(0, 1, n)
    swath_lons = np.linspace(10, 11, n)
    dem_swath = np.zeros((1, n * factor))
    lons = make_supersampled(swath_lons)[None, :]
    lats = make_supersampled(swath_lats)[None, :]
    sat_zen = np.zeros((1, n))
    out_lat, out_lon = _process_scanline_py(0, dem_swath, lons, lats, swath_lats[None, :], swath_lons[None, :], sat_zen)

    assert np.allclose(out_lat, swath_lats)
    assert np.allclose(out_lon, swath_lons)


def test_dem_forward():
    """Test with constant elevation for testing forward search."""
    n = 6
    factor = 4
    swath_lats = np.linspace(0, 1, n)
    swath_lons = np.linspace(20, 21, n)

    lat_sup = make_supersampled(swath_lats)
    lon_sup = make_supersampled(swath_lons)

    out_lat = np.empty(n)
    out_lon = np.empty(n)
    elev = np.full(n * factor, 2000)
    dists = np.linspace(0, 10000, n * factor)
    tanz = np.array([np.tan(np.radians(30))] * n)
    _orthocorrect_line_py(elev, dists, tanz, swath_lats, swath_lons, lon_sup, lat_sup, out_lat, out_lon)
    for j in range(n):
        idx_nom = j * factor
        if j <= n // 2:
            expected_idx = idx_nom + 2
            assert out_lat[j] == lat_sup[expected_idx]
            assert out_lon[j] == lon_sup[expected_idx]


def test_dem_backward():
    """Test with constant elevation for testing backward search."""
    n = 6
    factor = 4
    swath_lats = np.linspace(0, 1, n)
    swath_lons = np.linspace(20, 21, n)

    lat_sup = make_supersampled(swath_lats)
    lon_sup = make_supersampled(swath_lons)

    out_lat = np.empty(n)
    out_lon = np.empty(n)
    elev = np.full(n * factor, 2000)
    dists = np.linspace(0, 10000, n * factor)
    tanz = np.array([np.tan(np.radians(30))] * n)
    _orthocorrect_line_py(elev, dists, tanz, swath_lats, swath_lons, lon_sup, lat_sup, out_lat, out_lon)
    for j in range(n):
        idx_nom = j * factor
        if j > n // 2:
            expected_idx = idx_nom - 2
            assert out_lat[j] == lat_sup[expected_idx]
            assert out_lon[j] == lon_sup[expected_idx]


def test_spike_interception():
    """One large spike in forward and backward scan. Corrected FOV locations should be equal to spike location."""
    n = 8
    factor = 4
    swath_lats = np.linspace(0, 2, n)
    swath_lons = np.linspace(30, 32, n)
    dists = np.arange(n * factor, dtype=float)
    tanz = np.ones_like(swath_lats) * 0.5

    elev = np.zeros_like(dists)
    spike_idx = 10
    spike_idx_backward = 20
    elev[spike_idx] = 1e6
    elev[spike_idx_backward] = 1e6

    lat_sup = make_supersampled(swath_lats)
    lon_sup = make_supersampled(swath_lons)

    out_lat = np.empty(n)
    out_lon = np.empty(n)

    _orthocorrect_line_py(elev, dists, tanz, swath_lats, swath_lons, lon_sup, lat_sup, out_lat, out_lon)

    for j in range(n):
        idx_nom = j * factor
        if j < n // 2:
            expected_idx = spike_idx if spike_idx >= idx_nom else idx_nom
        else:
            expected_idx = spike_idx_backward if spike_idx_backward <= idx_nom else idx_nom

        assert out_lat[j] == lat_sup[expected_idx]
        assert out_lon[j] == lon_sup[expected_idx]


def test_the_destination_is_resampled_in_blocks():
    """A whole swath is never held at once, so a polar pass cannot exhaust the memory.

    The reference is resampled onto the swath's own geolocation, and that destination
    is the largest thing in the calculation: a LAC pass is a few thousand lines of 2048
    samples, and the resampler holds the arrays it is given. Handing it the swath in
    blocks bounds what is live at any moment, at the cost of a little scheduling.
    """
    import xarray as xr
    from georeferencer.georeferencer import swath_in_blocks

    lons = xr.DataArray(np.zeros((3530, 2048)), dims=("y", "x"))
    lats = xr.DataArray(np.zeros((3530, 2048)), dims=("y", "x"))

    destination = swath_in_blocks(lons, lats)

    assert max(max(sizes) for sizes in destination.lons.chunks) <= 1024
