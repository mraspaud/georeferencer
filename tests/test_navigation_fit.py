"""Tests for fitting a navigation to control points that have already been found."""

import datetime as dt

import numpy as np
import xarray as xr

from georeferencer.georeferencer import fit_navigation

TLE = ("1 33591U 09005A   12345.45213434  .00000391  00000-0  24004-3 0  6113",
       "2 33591 098.8821 283.2036 0013384 242.4835 117.4960 14.11432063197875")
STARTED = dt.datetime(2012, 12, 12, 4, 16, 1, 575000)
LINES = 400


def a_pass_displaced_along_its_track(seconds):
    """Return control points, and where they truly are for a swath that late."""
    from pyorbital.geoloc_avhrr import compute_avhrr_gcps_lonlatalt

    gcps = np.array([[float(line), float(sample)]
                     for line in range(20, LINES, 40) for sample in (300, 1000, 1700)])
    lons, lats, _ = compute_avhrr_gcps_lonlatalt(
        gcps, 55.37, (0, 0, 0), STARTED + dt.timedelta(seconds=seconds), TLE)
    times = np.array([np.datetime64(STARTED) + np.timedelta64(int(line * 1e6 / 6), "us")
                      for line in range(LINES)])
    calibrated_ds = xr.Dataset(
        {"times": ("scan_line_index", times)},
        coords={"scan_line_index": np.arange(LINES)},
        attrs={"tle": TLE, "max_scan_angle": 55.37})
    return calibrated_ds, gcps, np.column_stack([lons, lats])


def test_a_fit_not_asked_for_time_holds_it_at_zero():
    """Where the clock is known, the swath must not be slid along its track to fit."""
    calibrated_ds, gcps, lonlats = a_pass_displaced_along_its_track(6.0)

    seconds, _, _ = fit_navigation(calibrated_ds, gcps, lonlats, solve_for_time=False)

    assert seconds == 0.0
