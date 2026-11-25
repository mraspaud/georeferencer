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
