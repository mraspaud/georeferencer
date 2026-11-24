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


def test_the_dem_is_read_under_every_supersampled_point(tmp_path):
    """Each point gets the height of the DEM pixel it falls in, and NaN off the DEM's edge.

    The grid is long enough to be read in more than one piece, so the answer must not
    depend on where one piece ends and the next begins.
    """
    import rasterio
    from georeferencer.georeferencer import _get_dem_swath
    from rasterio.transform import from_origin

    heights = np.arange(200 * 300, dtype="float32").reshape(200, 300)
    dem = tmp_path / "dem.tif"
    with rasterio.open(dem, "w", driver="GTiff", height=200, width=300, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(10.0, 50.0, 0.1, 0.1)) as out:
        out.write(heights, 1)
    rng = np.random.default_rng(0)
    # off the DEM's last two rows and columns, which the subsetting can round away
    rows = rng.integers(0, 198, size=(1500, 40))
    cols = rng.integers(0, 298, size=(1500, 40))
    lons = 10.0 + (cols + 0.1) * 0.1        # just inside each pixel's corner
    lats = 50.0 - (rows + 0.1) * 0.1
    lons[:, 0] = 5.0                         # west of the DEM
    lats[:, 1] = 60.0                        # north of it

    found = _get_dem_swath(dem, 30.0, 50.0, 10.0, 40.0, lons, lats)

    expected = heights[rows, cols].astype(float)
    expected[:, :2] = np.nan
    np.testing.assert_array_equal(found, expected)


def a_swath_of_lonlats(lines=300):
    """Return lon/lats laid out like an AVHRR swath: lines along an inclined track, 2048 samples across."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    track_lons, track_lats, _ = geod.fwd(np.full(lines, 20.0), np.full(lines, -30.0),
                                         np.full(lines, -12.0), np.arange(lines) * 1100.0)
    across = np.tan(np.deg2rad(np.linspace(-55.37, 55.37, 2048))) * 830e3      # flat-earth spread, grows at the edges
    lons, lats, _ = geod.fwd(np.repeat(track_lons, 2048).reshape(lines, 2048),
                             np.repeat(track_lats, 2048).reshape(lines, 2048),
                             np.full((lines, 2048), 78.0), np.tile(across, (lines, 1)))
    return lons, lats


def test_the_supersampled_grid_follows_the_spline_through_every_pixel():
    """The grid the terrain is searched on is a cubic spline through each scan line's pixels.

    Fitted line by line it was most of the orthocorrection's time; however it is fitted,
    the points must be where that spline puts them.
    """
    import geotiepoints as gtp
    from georeferencer.georeferencer import lonlat_interpolator

    lons, lats = a_swath_of_lonlats()
    fine = np.arange(0, 2048, 0.25)
    expected_lons, expected_lats = gtp.SatelliteInterpolator(
        (lons, lats), (np.arange(len(lats)), np.arange(2048)), (np.arange(len(lats)), fine), 1, 3).interpolate()

    found_lons, found_lats = lonlat_interpolator(lons, lats, np.arange(2048), fine)

    np.testing.assert_allclose(found_lons, expected_lons, rtol=0, atol=1e-9)
    np.testing.assert_allclose(found_lats, expected_lats, rtol=0, atol=1e-9)
