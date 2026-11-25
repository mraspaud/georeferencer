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
    search = georef.gcp_lonlats_of_reference

    def counted(path, **kwargs):
        searches.append(path)
        return search(path, **kwargs)

    monkeypatch.setattr(georef, "gcp_lonlats_of_reference", counted)
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
