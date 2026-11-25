"""The ground control points of a whole reference image, found once and kept."""

import numpy as np


def a_reference(path, height=640, width=960, seed=3):
    """Write a three-band reference GeoTIFF of textured terrain, and return its first band."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    ground = np.cumsum(np.cumsum(rng.normal(0, 1, (height, width)), axis=0), axis=1)
    ground = (255 * (ground - ground.min()) / np.ptp(ground)).astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=3, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(-20.0, 60.0, 1 / 240, 1 / 240)) as out:
        for band in (1, 2, 3):
            out.write(ground, band)
    return ground


def test_the_gcps_found_strip_by_strip_are_those_of_the_whole_reference(tmp_path):
    """Read a strip at a time, the reference yields the GCPs its whole first band does.

    A global reference is too large to hold at once, so it is read in strips and only
    the sums each variance window needs are kept. The GCPs, their thinning and their
    recovery, are then those the variance grid of the whole image gives.
    """
    import georeferencer.gcp_generation as gcp_gen
    from georeferencer.georeferencer import gcp_lonlats_of_reference, gcp_to_lonlat
    from rasterio.transform import from_origin

    ground = a_reference(tmp_path / "reference.tif")
    kilometre = gcp_gen.downsample_2x2(ground.astype(np.float64) / 2.55)
    variance = gcp_gen.get_variance_array(kilometre, 8, 48)
    gcps = np.array(gcp_gen.thin_gcp_candidates(variance, gcp_gen.get_gcp_candidates(variance, 3), 11))
    lons, lats = gcp_to_lonlat(gcps[:, 1], gcps[:, 0], from_origin(-20.0, 60.0, 1 / 240, 1 / 240))

    found = gcp_lonlats_of_reference(tmp_path / "reference.tif", lines_per_read=64)

    assert len(found) > 20
    np.testing.assert_allclose(found[np.lexsort(found.T)],
                               np.column_stack([lons, lats])[np.lexsort((lons, lats))], rtol=0, atol=1e-9)


def counting_searches(monkeypatch):
    """Count the searches of a whole reference for its GCPs."""
    import georeferencer.georeferencer as georef

    searches = []
    search = georef.gcps_of_reference

    def counted(path, **kwargs):
        searches.append(path)
        return search(path, **kwargs)

    monkeypatch.setattr(georef, "gcps_of_reference", counted)
    return searches


def test_a_references_gcps_are_found_once_and_read_back_afterwards(tmp_path, monkeypatch):
    """The first pass matched against a reference pays for its GCPs; every later one reads them."""
    from georeferencer.georeferencer import reference_gcp_lonlats

    a_reference(tmp_path / "reference.tif")
    searches = counting_searches(monkeypatch)

    first = reference_gcp_lonlats(tmp_path / "reference.tif", cache_dir=tmp_path / "cache")
    again = reference_gcp_lonlats(tmp_path / "reference.tif", cache_dir=tmp_path / "cache")

    assert len(searches) == 1
    np.testing.assert_array_equal(first, again)


def test_a_changed_reference_is_searched_again(tmp_path, monkeypatch):
    """GCPs kept for a reference are not used for another one written in its place."""
    import os

    from georeferencer.georeferencer import gcp_lonlats_of_reference, reference_gcp_lonlats

    a_reference(tmp_path / "reference.tif", seed=3)
    reference_gcp_lonlats(tmp_path / "reference.tif", cache_dir=tmp_path / "cache")
    a_reference(tmp_path / "reference.tif", seed=4)
    stat = os.stat(tmp_path / "reference.tif")
    os.utime(tmp_path / "reference.tif", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    searches = counting_searches(monkeypatch)

    found = reference_gcp_lonlats(tmp_path / "reference.tif", cache_dir=tmp_path / "cache")

    assert len(searches) == 1
    np.testing.assert_array_equal(found, gcp_lonlats_of_reference(tmp_path / "reference.tif"))


def test_the_gcps_are_kept_where_the_environment_says(tmp_path, monkeypatch):
    """Without a directory given, they are kept in GEOREFERENCER_CACHE_DIR."""
    from georeferencer.georeferencer import reference_gcp_lonlats

    a_reference(tmp_path / "reference.tif")
    monkeypatch.setenv("GEOREFERENCER_CACHE_DIR", str(tmp_path / "kept"))

    reference_gcp_lonlats(tmp_path / "reference.tif")

    assert len(list((tmp_path / "kept").iterdir())) == 1


def test_a_pass_takes_the_kept_gcps_it_sees_at_the_swath_pixel_nearest_each(tmp_path):
    """Each GCP in the swath goes to the pixel whose lon/lat is nearest its own.

    Those outside the swath's lon/lat box are left out, and so are those that land
    within twelve pixels of the swath's edge, where a matching block would not fit.
    """
    import xarray as xr
    from georeferencer.georeferencer import place_gcps_in_swath

    lines, samples = 200, 300
    lons, lats = np.meshgrid(np.linspace(10.0, 16.0, samples), np.linspace(50.0, 46.0, lines))
    swath = xr.Dataset({"longitude": (("y", "x"), lons), "latitude": (("y", "x"), lats)})
    rng = np.random.default_rng(5)
    gcps = np.column_stack([rng.uniform(8.0, 18.0, 400), rng.uniform(44.0, 52.0, 400)])

    coords, lonlats = place_gcps_in_swath(gcps, swath)

    inside = (gcps[:, 0] >= 10.0) & (gcps[:, 0] <= 16.0) & (gcps[:, 1] >= 46.0) & (gcps[:, 1] <= 50.0)
    nearest = [np.unravel_index(np.argmin((lons - lon) ** 2 + (lats - lat) ** 2), lons.shape)
               for lon, lat in gcps[inside]]
    clear = [12 <= y < lines - 12 and 12 <= x < samples - 12 for y, x in nearest]
    assert coords == [(int(y), int(x)) for (y, x), ok in zip(nearest, clear, strict=True) if ok]
    assert lonlats == [tuple(lonlat) for lonlat, ok in zip(gcps[inside], clear, strict=True) if ok]


def test_a_gcp_is_placed_at_the_right_pixel_past_missing_geolocation():
    """Scan lines without geolocation do not shift where the GCPs after them are placed."""
    import xarray as xr
    from georeferencer.georeferencer import place_gcps_in_swath

    lines, samples = 200, 300
    lons, lats = np.meshgrid(np.linspace(10.0, 16.0, samples), np.linspace(50.0, 46.0, lines))
    lons[:20], lats[:20] = np.nan, np.nan
    swath = xr.Dataset({"longitude": (("y", "x"), lons), "latitude": (("y", "x"), lats)})

    coords, _ = place_gcps_in_swath(np.array([[13.0, 47.0]]), swath)

    assert coords == [tuple(int(i) for i in np.unravel_index(np.nanargmin((lons - 13.0) ** 2 + (lats - 47.0) ** 2),
                                                             lons.shape))]


def test_the_kept_gcps_can_be_read_by_whoever_shares_the_cache(tmp_path):
    """The kept file gets the permissions the umask gives any new file, not a private one."""
    import os

    from georeferencer.georeferencer import reference_gcp_lonlats

    a_reference(tmp_path / "reference.tif")
    umask = os.umask(0o022)
    try:
        reference_gcp_lonlats(tmp_path / "reference.tif", cache_dir=tmp_path / "cache")
    finally:
        os.umask(umask)

    (kept,) = (tmp_path / "cache").iterdir()
    assert kept.stat().st_mode & 0o777 == 0o644


def test_the_gcps_are_kept_with_their_variance_grid_as_reference_data(tmp_path):
    """The kept file is georeferencer's reference data: the GCPs on the variance grid, and the grid.

    It is written and read with the reference data helpers, so it can be looked into,
    or made elsewhere, with them.
    """
    import georeferencer.gcp_generation as gcp_gen
    from georeferencer.georeferencer import gcps_of_reference, reference_gcp_lonlats

    a_reference(tmp_path / "reference.tif")
    gcps, variance = gcps_of_reference(tmp_path / "reference.tif")

    reference_gcp_lonlats(tmp_path / "reference.tif", cache_dir=tmp_path / "cache")

    (kept,) = (tmp_path / "cache").iterdir()
    np.testing.assert_array_equal(gcp_gen.get_gcp_points_from_file(kept), gcps)
    np.testing.assert_array_equal(gcp_gen.get_variance_array_from_file(kept), variance)
