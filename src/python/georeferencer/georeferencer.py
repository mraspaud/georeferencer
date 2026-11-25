"""Georeferencing and displacement calculation for satellite imagery.

This module provides functions for processing satellite image data,
georeferencing using ground control points (GCPs), and calculating
displacement between a swath image and a reference image.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import dask.array as da
import numpy as np
import rioxarray
import xarray as xr
from numba import njit
from pyproj import Geod
from pyresample import gradient
from pyresample.geometry import AreaDefinition, SwathDefinition
from rasterio.transform import xy
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

import georeferencer.displacement_calc as dc
import georeferencer.gcp_generation as gcp_gen

_geod = Geod(ellps="WGS84")
logger = logging.getLogger(__name__)
INVALID_DISPLACEMENT = (-100, -100)

#: The matcher compares a square window this many pixels across around each point.
COVARIANCE_WINDOW = 48

#: How far around a point it looks for the best match, in pixels. This is what
#: bounds how far out of place a swath may be and still be found.
SEARCH_RADIUS = 24


def subset_from_bounds(buffer, dataset, max_lat, max_lon, min_lat, min_lon):
    """Selects a subset from a dataset based on latitude and longitude bounds."""
    dataset_bounds = dataset.rio.bounds()
    dataset_min_lon, dataset_min_lat, dataset_max_lon, dataset_max_lat = dataset_bounds

    min_lat = max(min_lat - buffer, dataset_min_lat)
    max_lat = min(max_lat + buffer, dataset_max_lat)
    min_lon = max(min_lon - buffer, dataset_min_lon)
    max_lon = min(max_lon + buffer, dataset_max_lon)

    subset_window = from_bounds(
        left=min_lon, bottom=min_lat, right=max_lon, top=max_lat, transform=dataset.rio.transform()
    )

    col_off = int(subset_window.col_off)
    row_off = int(subset_window.row_off)
    width = int(subset_window.width / 2) * 2
    height = int(subset_window.height / 2) * 2

    subset = dataset.isel(x=slice(col_off, col_off + width), y=slice(row_off, row_off + height))
    return subset


def open_subset_tif(filepath, min_lat, max_lat, min_lon, max_lon, buffer=5, chunks=512):
    """Open a subset of a GeoTIFF file based on latitude and longitude bounds.

    This function extracts a subset of a GeoTIFF file by defining a bounding box
    around specified latitude and longitude coordinates, with an optional buffer.

    Args:
        filepath (str): Path to the GeoTIFF file.
        min_lat (float): Minimum latitude for the subset.
        max_lat (float): Maximum latitude for the subset.
        min_lon (float): Minimum longitude for the subset.
        max_lon (float): Maximum longitude for the subset.
        buffer (int, optional): Buffer (in degrees) to extend the bounding box. Default is 5.
        chunks (int or None, optional): Dask chunk size; None opens the file lazily without dask.

    Returns:
        xarray.DataArray: A lazily read xarray DataArray containing the subset of the image.
    """
    dataset = rioxarray.open_rasterio(filepath, engine="rasterio", chunks=chunks)
    subset = subset_from_bounds(buffer, dataset, max_lat, max_lon, min_lat, min_lon)

    return subset


def gcp_to_lonlat(gcp_x, gcp_y, geo_transform):
    """Convert Ground Control Point (GCP) pixel coordinates to geographic coordinates (longitude, latitude).

    Args:
        gcp_x (int): X-coordinate (pixel) of the GCP.
        gcp_y (int): Y-coordinate (pixel) of the GCP.
        geo_transform (Affine): Geotransform matrix from rasterio.

    Returns:
        tuple: (longitude, latitude) of the GCP in geographic coordinates.
    """
    gcp_x_original = gcp_x * 16 + 24 * 2
    gcp_y_original = gcp_y * 16 + 24 * 2

    xp, yp = xy(geo_transform, gcp_y_original, gcp_x_original)

    return xp, yp


def place_gcps_in_swath(gcp_lonlats, calibrated_ds):
    """Place each GCP the swath sees at the swath pixel whose lon/lat is nearest its own.

    GCPs outside the swath's lon/lat box are left out, and so are those landing within
    twelve pixels of the swath's edge.

    Args:
        gcp_lonlats (array-like): The GCPs' (lon, lat), one per row.
        calibrated_ds (xarray.Dataset): Calibrated swath dataset containing longitude and latitude data.

    Returns:
        (gcp coords, gcp lonlats): the GCPs placed, as (line, sample) and as (lon, lat).
    """
    if "tc_longitude" in calibrated_ds and "tc_latitude" in calibrated_ds:
        lons = calibrated_ds["tc_longitude"].values
        lats = calibrated_ds["tc_latitude"].values
    else:
        lons = calibrated_ds["longitude"].values
        lats = calibrated_ds["latitude"].values

    gcp_lonlats = np.asarray(gcp_lonlats, dtype=float)
    seen = ((gcp_lonlats[:, 0] >= np.nanmin(lons)) & (gcp_lonlats[:, 0] <= np.nanmax(lons))
            & (gcp_lonlats[:, 1] >= np.nanmin(lats)) & (gcp_lonlats[:, 1] <= np.nanmax(lats)))
    gcp_lonlats = gcp_lonlats[seen]

    # TODO change to another KDTree
    pixels = np.column_stack((lons.ravel(), lats.ravel()))
    valid = ~np.isnan(pixels).any(axis=1)
    # An unbalanced tree is built several times faster and finds the same neighbours.
    tree = cKDTree(pixels[valid], balanced_tree=False, compact_nodes=False)
    _, nearest = tree.query(gcp_lonlats)
    lines, samples = np.unravel_index(np.flatnonzero(valid)[nearest], lons.shape)

    margin = 24 // 2
    clear = ((lines >= margin) & (lines < lons.shape[0] - margin)
             & (samples >= margin) & (samples < lons.shape[1] - margin))
    swath_coords = [(int(line), int(sample)) for line, sample in zip(lines[clear], samples[clear], strict=True)]
    return swath_coords, [tuple(lonlat) for lonlat in gcp_lonlats[clear]]


def reproject_reference_into_swath(
    matrix,
    swath_longitudes,
    swath_latitudes,
):
    """Reproject a reference image to match the swath image coordinate system.

    This function aligns a reference image with the swath data by using its
    geospatial bounds, regridding, and resampling it to match the swath.

    Args:
        matrix (np.array): Reference image data as a NumPy array.
        swath_longitudes (xarray.DataArray): Swath image longitudes.
        swath_latitudes (xarray.DataArray): Swath image latitudes.

    Returns:
        xarray.DataArray: The reprojected reference image aligned with the swath.
    """
    swath_lons_values = swath_longitudes.values
    swath_lats_values = swath_latitudes.values
    lon_min_swath = np.nanmin(swath_lons_values)
    lon_max_swath = np.nanmax(swath_lons_values)
    lat_min_swath = np.nanmin(swath_lats_values)
    lat_max_swath = np.nanmax(swath_lats_values)
    sub_matrix = subset_from_bounds(0, matrix, lat_max_swath, lon_max_swath, lat_min_swath, lon_min_swath)
    bounds = sub_matrix.rio.bounds()
    sub_matrix = sub_matrix.coarsen(x=2, y=2, boundary="trim").mean()
    source_area = AreaDefinition(
        "subgrid_area",
        "reference image AD around swath lat,lons",
        "tif_projection",
        sub_matrix.rio.crs,
        sub_matrix.shape[1],
        sub_matrix.shape[0],
        bounds,
    )
    full_swath_def = SwathDefinition(lons=swath_longitudes, lats=swath_latitudes)

    res = gradient.create_gradient_search_resampler(source_area, full_swath_def)
    resampled_data = res.resample(sub_matrix)
    return resampled_data


def _build_swath_image(calibrated_ds, sun_zen):
    """Builds a composite swath image using day/night channels corrected for sun zenith."""
    day_swath = calibrated_ds.sel(channel_name="2").channels / np.cos(np.deg2rad(sun_zen))
    night_swath = calibrated_ds.sel(channel_name="4").channels

    max_val = night_swath.max()
    night_swath = max_val - night_swath
    swath = da.where(sun_zen >= 87, night_swath, day_swath)
    return da.nan_to_num(swath).astype(np.float32)


def _load_reference_image(reference_image_path, calibrated_ds):
    """Loads a subset of the reference image based on swath bounds."""
    if "tc_longitude" in calibrated_ds and "tc_latitude" in calibrated_ds:
        lons = calibrated_ds["tc_longitude"]
        lats = calibrated_ds["tc_latitude"]
    else:
        lons = calibrated_ds["longitude"]
        lats = calibrated_ds["latitude"]

    tif = open_subset_tif(
        reference_image_path,
        np.min(lats),
        np.max(lats),
        np.min(lons),
        np.max(lons),
    )
    return tif.sel(band=1).astype(np.float32) / 2.55


def _reproject_to_swath(ref_image, calibrated_ds):
    """Reprojects the reference image into the swath coordinate system."""
    if "tc_longitude" in calibrated_ds and "tc_latitude" in calibrated_ds:
        lons = calibrated_ds["tc_longitude"]
        lats = calibrated_ds["tc_latitude"]
    else:
        lons = calibrated_ds["longitude"]
        lats = calibrated_ds["latitude"]
    return reproject_reference_into_swath(
        ref_image,
        lons,
        lats,
    ).astype(np.float32)


def gcp_lonlats_of_reference(reference_image_path, lines_per_read=512):
    """Return the lon/lats of every ground control point a reference image offers.

    The GCPs depend on the reference image alone, so they can be found once for the
    whole image instead of for every pass. Its first band is read a strip at a time,
    averaged to about 1 km as the matching uses it, and only the mean and the mean of
    squares of each 8 x 8 block are kept: the 48-wide variance windows sampled every
    eighth pixel are made of whole blocks, so the variance grid of the whole image is
    had from them without holding the image. Candidates are then chosen, thinned and
    recovered on that grid as for a single pass.
    """
    import rasterio
    from rasterio.windows import Window

    step, window = 8, 48
    with rasterio.open(reference_image_path) as src:
        height, width = (src.height // (2 * step)) * 2 * step, (src.width // (2 * step)) * 2 * step
        lines_per_read = max(2 * step, lines_per_read // (2 * step) * 2 * step)
        means, means_of_squares = [], []
        for first in range(0, height, lines_per_read):
            raw = src.read(1, window=Window(0, first, width, min(lines_per_read, height - first)))
            lines = raw.shape[0]
            kilometre = gcp_gen.downsample_2x2(raw) / 2.55
            blocks = kilometre.reshape(lines // (2 * step), step, width // (2 * step), step)
            means.append(blocks.mean(axis=(1, 3)))
            means_of_squares.append((blocks ** 2).mean(axis=(1, 3)))
        transform = src.transform

    variance = gcp_gen.variance_from_block_moments(np.vstack(means), np.vstack(means_of_squares), window // step,
                                                   (height // 2 - window) // step + 1,
                                                   (width // 2 - window) // step + 1)
    gcps = np.array(gcp_gen.thin_gcp_candidates(variance, gcp_gen.get_gcp_candidates(variance, 3), 11))
    lons, lats = gcp_to_lonlat(gcps[:, 1], gcps[:, 0], transform)
    return np.column_stack([lons, lats])


#: Bumped whenever the GCPs found for a reference image change, so kept ones are found anew.
GCP_DATABASE_VERSION = 1


def reference_gcp_lonlats(reference_image_path, cache_dir=None):
    """Return the lon/lats of a reference image's GCPs, found on first use and kept.

    They are kept in *cache_dir*, or else in ``GEOREFERENCER_CACHE_DIR``, or else in
    georeferencer's directory of the user's cache. They are kept under the reference
    image's name, size and modification time, so a reference written in place of
    another is searched again; reading a checksum of the whole image would cost a good
    part of what keeping its GCPs saves.
    """
    reference = Path(reference_image_path)
    if cache_dir is None:
        cache_dir = os.environ.get("GEOREFERENCER_CACHE_DIR") or (
            Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "georeferencer")
    cache_dir = Path(cache_dir)
    stat = reference.stat()
    kept = cache_dir / f"{reference.stem}-{stat.st_size}-{stat.st_mtime_ns}-gcps-v{GCP_DATABASE_VERSION}.npz"
    if kept.exists():
        with np.load(kept) as stored:
            return stored["lonlats"]

    lonlats = gcp_lonlats_of_reference(reference)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Written beside and then moved into place, so a pass running alongside never
    # reads a half-written file.
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as unfinished:
        np.savez(unfinished, lonlats=lonlats, reference=str(reference.resolve()))
    # A temporary file is made private; the kept one is for whoever shares the cache.
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(unfinished.name, 0o666 & ~umask)
    os.replace(unfinished.name, kept)
    return lonlats


def displacement_scatter(points, displacements):
    """Return how much of a displacement field no smooth geometry explains, in pixels.

    Every parameter the registration solves for moves the swath smoothly: a time
    offset shifts it bodily, a yaw error by an amount proportional to the distance
    from nadir. What such a field leaves behind is small. Matched noise leaves
    behind everything, however few pixels each match happens to be out by.
    """
    y = np.asarray([point[0] for point in points], dtype=float)
    x = np.asarray([point[1] for point in points], dtype=float)
    smooth = np.column_stack([np.ones_like(x), x, y, x * y])
    displacements = np.asarray(displacements, dtype=float)
    left_over = [component - smooth @ np.linalg.lstsq(smooth, component, rcond=None)[0]
                 for component in displacements.T]
    return float(np.hypot(*[np.std(component) for component in left_over]))


def _aggregated(image, factor):
    """Return *image* averaged over blocks of *factor* by *factor* pixels."""
    rows = (image.shape[0] // factor) * factor
    columns = (image.shape[1] // factor) * factor
    return image[:rows, :columns].reshape(
        rows // factor, factor, columns // factor, factor).mean(axis=(1, 3))


def estimate_gross_displacement(swath, reference, points, factor=8):
    """Return the shift that carries *swath* onto *reference*, in swath pixels.

    The search looks a fixed distance around each point, which bounds how far out
    of place a swath may be and still be found. Averaging both images over blocks
    of *factor* pixels lets that same search reach *factor* times further, at the
    cost of locating the answer only to within *factor* pixels.
    """
    coarse = sorted({(int(y) // factor, int(x) // factor) for y, x in points})
    displacement = np.array(
        dc.calculate_covariance_displacement(
            coarse,
            np.ascontiguousarray(_aggregated(swath, factor), dtype=np.float32),
            np.ascontiguousarray(_aggregated(reference, factor), dtype=np.float32),
            COVARIANCE_WINDOW, SEARCH_RADIUS),
        dtype=np.float32)
    found = displacement[~(displacement[:, 0] <= INVALID_DISPLACEMENT[0])]
    if not len(found):
        raise ValueError("No displacement found at any point")
    return tuple(np.median(found, axis=0) * factor)


def footprint_sizes(slant_range, local_zenith, field_of_view):
    """Return the ground a single detector sees, along track and across track.

    A footprint is not the spacing between samples, and the two must not be
    confused: along track the line spacing is set by the platform's motion and is
    near enough constant, while the footprint itself grows with the distance the
    view has to travel.

    Both axes grow with *slant_range*. Only the across-track axis is stretched by
    the ground tilting away from the line of sight, which is what makes the
    footprint rectangular away from nadir.
    """
    along = field_of_view * slant_range
    return along, along / np.cos(local_zenith)


class PixelFractions(NamedTuple):
    """A miss expressed against both the sample spacing and the footprint.

    The direction each miss was measured in is carried alongside it, because a miss
    on its own is only one component of a displacement and cannot be aggregated
    without knowing which way it points.
    """

    of_spacing: np.ndarray
    of_footprint: np.ndarray
    normals: np.ndarray | None = None


def as_pixel_fractions(miss, spacing, footprint, normals=None):
    """Express *miss* as a fraction of a sample step and of a footprint.

    Both are given because they are not interchangeable and the difference between
    them is not a constant: across track the two grow together, but along track the
    spacing barely changes while the footprint roughly doubles towards the swath
    edge. A figure quoted against one and read as the other is wrong by a factor
    that depends on where in the swath it was measured.
    """
    return PixelFractions(np.asarray(miss) / np.asarray(spacing),
                          np.asarray(miss) / np.asarray(footprint),
                          None if normals is None else np.asarray(normals))


def footprint_towards(direction, along, across):
    """Return how wide the footprint is in *direction*, measured from the track.

    A footprint is not round: it is much wider across the track than along it, and
    an error has to be judged against the width in the direction it was actually
    measured. A shoreline crossing is measured across the coast, and the coast runs
    wherever it runs, so that direction is set by geography rather than by the
    instrument.

    *direction* is an angle in radians from the along-track axis, so zero gives the
    narrow side and a right angle the wide one. In between it combines the two in
    quadrature, treating the footprint as an ellipse.

    That shape is pinned at three directions -- along, across, and obliquely -- and
    those three are enough to separate it from the alternatives worth considering:
    a straight average of the two widths and an L1 combination both miss the
    oblique case by more than a per cent. It is believed rather than proven for the
    directions in between, and a per-cent error there would be paid on most
    crossings, because most coasts run obliquely.
    """
    return float(np.hypot(along * np.cos(direction), across * np.sin(direction)))


def summarise_misses(misses):
    """Return how far *misses* sit off centre, and how widely they scatter.

    These answer different questions and are reported separately. The offset is
    where the navigation systematically puts things, and it is a signed median, so
    misses that fall as often one way as the other cancel. The scatter is how much
    the individual measurements disagree with each other, and it says nothing about
    whether they are centred.

    Collapsing the two -- by taking the size of a typical miss regardless of its
    sign -- would report scatter as though it were bias, and would condemn a
    perfectly centred navigation for being imprecise.
    """
    return float(np.median(misses)), float(np.std(misses))


def _clear_of_the_wrapped_seam(points, shape, steps):
    """Say which *points* keep their matching window clear of the carried swath's far end.

    Carrying the swath brings its opposite edge round to meet the near one. That
    ground is real, but it is not the ground the reference holds there, and a
    window straddling the join matches it confidently and wrongly.
    """
    reach = COVARIANCE_WINDOW // 2 + SEARCH_RADIUS
    points = np.asarray(points, dtype=np.float32)
    clear = np.ones(len(points), dtype=bool)
    for axis, step in enumerate(steps):
        if step == 0:
            continue
        along = points[:, axis]
        if step > 0:
            clear &= along >= step + reach
        else:
            clear &= along < shape[axis] + step - reach
    return clear


def find_control_points(swath_coords, gcp_lonlats, swath, ref_swath):
    """Calculates valid GCPs based on displacement analysis between swath and reference.

    The fine search reaches only its own radius, so a swath further out of place
    than that is never found at all. A coarse pass over both images averaged into
    blocks reaches much further; carrying the swath that far first leaves the fine
    search a residual it can reach, and what the coarse pass took out is added
    back afterwards.

    How far the swath was carried is returned alongside the points, because the fit
    that follows searches only a few seconds for the time offset and needs to centre
    that reach on the coarse answer rather than on zero.
    """
    seen = swath.compute() if hasattr(swath, "compute") else swath
    gross = estimate_gross_displacement(seen, ref_swath, swath_coords)
    carried = (int(round(gross[0])), int(round(gross[1])))
    brought_near = np.roll(seen, carried, axis=(0, 1))
    clear_of_seam = _clear_of_the_wrapped_seam(swath_coords, seen.shape, carried)

    displacement = np.array(
        dc.calculate_covariance_displacement(swath_coords, brought_near, ref_swath,
                                             COVARIANCE_WINDOW, SEARCH_RADIUS), dtype=np.float32
    )
    swath_coords = np.array(swath_coords, dtype=np.float32)
    gcp_lonlats = np.array(gcp_lonlats, dtype=np.float32)

    valid_mask = ~(displacement[:, 0] <= INVALID_DISPLACEMENT[0]) & clear_of_seam
    valid_displacements = displacement[valid_mask] + np.array(carried, dtype=np.float32)
    if len(valid_displacements) == 0:
        raise ValueError("No valid displacements found")
    valid_swath_coords = swath_coords[valid_mask]
    valid_gcp_lonlats = gcp_lonlats[valid_mask]

    valid_gcps = np.column_stack(
        [valid_swath_coords[:, 0] - valid_displacements[:, 0], valid_swath_coords[:, 1] - valid_displacements[:, 1]]
    )
    return valid_gcps, valid_gcp_lonlats, valid_swath_coords, carried


def measure_swath_displacement(calibrated_ds, sun_zen, sat_zen, reference_image_path,
                               dem_path=None):
    """Calculate the displacement between a swath image and a reference image.

    This function extracts a subset of the reference image, identifies
    ground control points (GCPs), and calculates displacement based on
    covariance analysis.

    Args:
        calibrated_ds (xarray.Dataset): Calibrated swath dataset containing channel data.
        sun_zen (2d Matrix): A 2d matrix with the same shape as the channel data containing the sun zenith angles.
        sat_zen (2d Matrix): A 2d matrix with the same shape as the channel data containing the satellite zenith angles.
        reference_image_path (str): Path to the reference GeoTIFF image.
        dem_path (str, optional): Path to the DEM GeoTIFF image.

    Returns:
        tuple: The control points, the reference lon/lats they were matched to, and how far
            along its own track the swath sits, counted in scan lines, as
            `(gcps, gcp_lonlats, along_track_lines)`. This package reports what it
            measured and makes no claim about what the displacement means, nor about
            how fast the lines arrived -- turning lines into a duration needs the scan
            rate, which is the caller's knowledge.

    Raises:
        ValueError: If no valid displacement is found.
    """
    if dem_path:
        calibrated_ds = orthocorrection(calibrated_ds, sat_zen, dem_path)
    swath = _build_swath_image(calibrated_ds, sun_zen)
    ref_image = _load_reference_image(reference_image_path, calibrated_ds)
    target_area = _reproject_to_swath(ref_image, calibrated_ds)
    swath_coords, gcp_lonlats = place_gcps_in_swath(reference_gcp_lonlats(reference_image_path), calibrated_ds)
    gcps, valid_gcp_lonlats, valid_swath_coords, carried = find_control_points(
        swath_coords, gcp_lonlats, swath, np.nan_to_num(target_area.values, nan=0.0)
    )

    valid_swath_coords = np.array(valid_swath_coords)
    valid_gcp_lonlats = np.array(valid_gcp_lonlats)
    gcps = np.array(gcps)
    y, x = valid_swath_coords[:, 0], valid_swath_coords[:, 1]
    lon, lat = valid_gcp_lonlats[:, 0], valid_gcp_lonlats[:, 1]
    y_corrected, x_corrected = gcps[:, 0], gcps[:, 1]
    y_displacement = valid_swath_coords[:, 0] - gcps[:, 0]
    x_displacement = valid_swath_coords[:, 1] - gcps[:, 1]

    calibrated_ds["gcp_x"] = xr.DataArray(x, dims=["points"])
    calibrated_ds["gcp_y"] = xr.DataArray(y, dims=["points"])
    calibrated_ds["gcp_longitude"] = xr.DataArray(lon, dims=["points"])
    calibrated_ds["gcp_latitude"] = xr.DataArray(lat, dims=["points"])
    calibrated_ds["gcp_x_corrected"] = xr.DataArray(x_corrected, dims=["points"])
    calibrated_ds["gcp_y_corrected"] = xr.DataArray(y_corrected, dims=["points"])
    calibrated_ds["gcp_x_displacement"] = xr.DataArray(x_displacement, dims=["points"])
    calibrated_ds["gcp_y_displacement"] = xr.DataArray(y_displacement, dims=["points"])

    _translate_gcp_lines_to_scanline_offsets(calibrated_ds, gcps)
    logger.debug(f"Found {len(gcps)} valid gcps")
    return gcps, valid_gcp_lonlats, carried[0]


def _translate_gcp_lines_to_scanline_offsets(calibrated_ds, gcps):
    """Translate gcp line to scanline offsets (zero-based).

    Useful with passes that miss scanlines.
    """
    ints, decs = np.divmod(gcps[:, 0], 1)
    gcps[:, 0] = calibrated_ds.scan_line_index.values[ints.astype(int)] - calibrated_ds.scan_line_index.values[0] + decs


def measure_swath_displacement_with_filename(swath_file, tle_dir, tle_file, reference_image_path,
                                             dem_path=None):
    """Measure a swath's displacement from a reference image, given the swath's filename.

    Args:
        swath_file (str): Path to the swath data file.
        tle_dir (str): Directory containing Two-Line Elements (TLE) files.
        tle_file (str): Name of the specific TLE file to use.
        reference_image_path (str): Path to the reference GeoTIFF image.
        dem_path (str): Path to the DEM GeoTIFF image.

    Returns:
        tuple: The control points, the reference lon/lats they were matched to, and how far
            along its own track the swath sits, counted in scan lines. What that
            displacement means for the platform is the caller's judgement, not this
            package's.
    """
    from pygac import get_reader_class

    reader_cls = get_reader_class(swath_file)
    reader = reader_cls(tle_dir=tle_dir, tle_name=tle_file)
    reader.read(swath_file)
    calibrated_ds = reader.get_calibrated_dataset()
    _, sat_zen, _, sun_zen, _ = reader.get_angles()

    return measure_swath_displacement(calibrated_ds, sun_zen, sat_zen, reference_image_path, dem_path)


@njit(nogil=True)
def _furthest_hiding_point(elev, dists, tan_view, idx_nom, sup_nadir, reach):
    """Return the furthest point towards nadir whose terrain rises into the line of sight.

    Nothing further than *reach* can, so the search stops there.
    """
    best_idx = idx_nom
    step = 1 if idx_nom <= sup_nadir else -1
    for k in range(idx_nom, sup_nadir, step):
        delta = abs(dists[k] - dists[idx_nom])
        if delta > reach:
            break
        if elev[k] > (delta / tan_view):
            best_idx = k
    return best_idx


@njit(nogil=True)
def _orthocorrect_line(elev, dists, tanz, swath_lats_i, swath_lons_i, lon_sup, lat_sup, out_lat, out_lon):
    npix = swath_lats_i.shape[0]
    nadir = np.argmin(tanz)
    sup_nadir = nadir * 4
    # No point further from a pixel than the highest terrain on the line times the
    # pixel's view slope can rise into its line of sight.
    highest = 0.0
    for k in range(elev.shape[0]):
        if elev[k] > highest:
            highest = elev[k]
    for j in range(npix):
        if tanz[j] == 0:
            out_lat[j] = swath_lats_i[j]
            out_lon[j] = swath_lons_i[j]
            continue
        best_idx = _furthest_hiding_point(elev, dists, tanz[j], j * 4, sup_nadir, highest * tanz[j])
        out_lat[j] = lat_sup[best_idx]
        out_lon[j] = lon_sup[best_idx]


def _distances_along_the_lines(lons, lats):
    """Return each point's distance from the start of its line, along the line, in metres.

    The points are a quarter of a pixel apart, so the straight line between neighbours
    on the WGS84 ellipsoid is the geodesic to well under a millimetre over a whole scan
    line, at a small fraction of the cost of solving for geodesics.
    """
    lon, lat = np.deg2rad(lons), np.deg2rad(lats)
    sin_lat = np.sin(lat)
    normal = _geod.a / np.sqrt(1 - _geod.es * sin_lat ** 2)
    x = normal * np.cos(lat) * np.cos(lon)
    y = normal * np.cos(lat) * np.sin(lon)
    z = normal * (1 - _geod.es) * sin_lat
    steps = np.sqrt(np.diff(x, axis=-1) ** 2 + np.diff(y, axis=-1) ** 2 + np.diff(z, axis=-1) ** 2)
    distances = np.zeros(np.shape(lons), dtype=np.float64)
    distances[..., 1:] = np.cumsum(steps, axis=-1)
    return distances


def _process_scanline(i, dem_swath, lons, lats, swath_lats, swath_lons, sat_zen):
    elev = dem_swath[i]
    lon_sup = lons[i]
    lat_sup = lats[i]

    dists = _distances_along_the_lines(lon_sup, lat_sup)

    # Calculate tan of view angles
    tanz = np.tan(np.deg2rad(sat_zen[i]))

    out_lat_line = np.full_like(swath_lats[i], np.nan, dtype=np.float64)
    out_lon_line = np.full_like(swath_lons[i], np.nan, dtype=np.float64)

    _orthocorrect_line(elev, dists, tanz, swath_lats[i], swath_lons[i], lon_sup, lat_sup, out_lat_line, out_lon_line)

    return out_lat_line, out_lon_line


#: How many scan lines' worth of supersampled points the DEM is read for at a time.
DEM_LINES_PER_READ = 256

#: How many scan lines are orthocorrected at a time. Each is corrected on its own, so the
#: pass is taken in blocks rather than holding its four-times-finer grid whole. The blocks
#: run side by side, one per dask worker, so they are kept small enough for a few of them
#: together to hold no more than one large block did.
ORTHOCORRECTION_LINES_PER_BLOCK = 32


def _get_dem_swath(dem_file_path, min_lats, max_lats, min_lon, max_lon, lons_sup, lats_sup):
    return _heights_under(_dem_subset(dem_file_path, min_lats, max_lats, min_lon, max_lon), lons_sup, lats_sup)


def _dem_subset(dem_file_path, min_lats, max_lats, min_lon, max_lon):
    """Open, without reading, the part of the DEM around a swath.

    It is read window by window from within the orthocorrection's own dask tasks, so it
    is opened without dask: a nested compute would share the scheduler's workers out
    again inside each task.
    """
    return open_subset_tif(dem_file_path, min_lats, max_lats, min_lon, max_lon, chunks=None).squeeze(drop=True)


def _heights_under(src, lons_sup, lats_sup):
    """Return the DEM's height under each point, bilinear between pixel centres, NaN off the DEM.

    As Khlopenkov et al. (2010) do: the DEM is about as coarse as the swath pixels, so
    taking the value of one DEM pixel would put the terrain up to half a DEM pixel out of
    place. Between the outermost pixel centres and the DEM's edge, the edge is carried out.
    """
    transform = src.rio.transform()
    nrows, ncols = src.sizes["y"], src.sizes["x"]

    # Read the DEM a block of scan lines at a time, each only over the window its points
    # fall in: indexing points one by one through dask is slow and holds most of the DEM
    # in memory at once, while the swath crosses its bounding box as a narrow band.
    output = np.full(lons_sup.shape, np.nan)
    for first in range(0, lons_sup.shape[0], DEM_LINES_PER_READ):
        block = slice(first, first + DEM_LINES_PER_READ)
        # Positions in DEM pixels, whole numbers at pixel centres.
        across = (lons_sup[block] - transform.c) / transform.a - 0.5
        down = (lats_sup[block] - transform.f) / transform.e - 0.5
        on_the_dem = (across >= -0.5) & (across < ncols - 0.5) & (down >= -0.5) & (down < nrows - 0.5)
        if not on_the_dem.any():
            continue
        if not on_the_dem.all():
            # Off the DEM, look anywhere on it, so as not to widen the window read.
            across = np.where(on_the_dem, across, across[on_the_dem][0])
            down = np.where(on_the_dem, down, down[on_the_dem][0])
        np.clip(across, 0, ncols - 1, out=across)
        np.clip(down, 0, nrows - 1, out=down)
        left, top = across.astype(np.int32), down.astype(np.int32)
        across -= left
        down -= top
        row0, col0 = top.min(), left.min()
        window = src.isel(y=slice(row0, min(top.max() + 2, nrows)),
                          x=slice(col0, min(left.max() + 2, ncols))).values
        top -= row0
        left -= col0
        right = np.minimum(left + 1, window.shape[1] - 1)
        bottom = np.minimum(top + 1, window.shape[0] - 1)
        upper = window[top, left] + across * (window[top, right] - window[top, left].astype(np.float64))
        lower = window[bottom, left] + across * (window[bottom, right] - window[bottom, left].astype(np.float64))
        heights = upper + down * (lower - upper)
        heights[~on_the_dem] = np.nan
        output[block] = heights

    return output


def _orthocorrect_block(swath_lons, swath_lats, sat_zen, dem):
    """Return the orthocorrected lats and lons of a block of scan lines, stacked."""
    ncols = swath_lons.shape[1]
    lons, lats = lonlat_interpolator(swath_lons, swath_lats, np.arange(ncols), np.arange(0, ncols, 0.25))
    heights = _heights_under(dem, lons, lats)
    corrected = np.full((2,) + swath_lons.shape, np.nan)
    for i in range(swath_lons.shape[0]):
        corrected[0, i], corrected[1, i] = _process_scanline(i, heights, lons, lats, swath_lats, swath_lons, sat_zen)
    return corrected


def orthocorrection(calibrated_ds, sat_zen, dem_file_path):
    """Performs orthocorrection on latitude and longitude based on satellite viewing angles.

    Each block of scan lines is a dask task, so the blocks are corrected side by side on
    whichever scheduler, and with however many workers, the processing chain sets up.
    """
    swath_lons = np.asarray(calibrated_ds["longitude"], dtype=np.float64)
    swath_lats = np.asarray(calibrated_ds["latitude"], dtype=np.float64)
    dem = _dem_subset(dem_file_path, np.min(swath_lats), np.max(swath_lats), np.min(swath_lons), np.max(swath_lons))

    chunks = (ORTHOCORRECTION_LINES_PER_BLOCK, swath_lons.shape[1])
    lons, lats, zeniths = (da.from_array(np.asarray(values), chunks=chunks)
                           for values in (swath_lons, swath_lats, sat_zen))
    corrected = da.map_blocks(_orthocorrect_block, lons, lats, zeniths, dem=dem, new_axis=0,
                              chunks=((2,),) + lons.chunks, dtype=np.float64,
                              meta=np.array((), dtype=np.float64))
    # Each block is written into place as it is done, rather than gathered and joined.
    out_lats, out_lons = out = np.empty(corrected.shape)
    da.store(corrected, out, lock=False)

    calibrated_ds["tc_latitude"] = (calibrated_ds["latitude"].dims, out_lats)
    calibrated_ds["tc_longitude"] = (calibrated_ds["longitude"].dims, out_lons)
    return calibrated_ds


def lonlat_interpolator(lons, lats, cols_subset, cols_full):
    """Interpolate from lat-lon tie-points to pixel locations along each scan line.

    Each line is a cubic spline through its pixels in earth-centred coordinates, the one
    geotiepoints fits; it is fitted for every line at once rather than line by line,
    which was most of the orthocorrection's time on a full pass.
    """
    from geotiepoints.geointerpolator import lonlat2xyz, xyz2lonlat
    from scipy.interpolate import CubicSpline

    xyz = lonlat2xyz(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))
    fine = [CubicSpline(cols_subset, coordinate, axis=1, bc_type="not-a-knot")(cols_full) for coordinate in xyz]
    return xyz2lonlat(*fine)


# -----------------------------------------------------------------------------
# Pure-Python implementations (for testing)
# -----------------------------------------------------------------------------
def _orthocorrect_line_py(elev, dists, tanz, swath_lats_i, swath_lons_i, lon_sup, lat_sup, out_lat, out_lon):
    npix = swath_lats_i.shape[0]
    nadir = npix // 2
    sup_nadir = nadir * 4
    for j in range(npix):
        if tanz[j] == 0:
            out_lat[j] = swath_lats_i[j]
            out_lon[j] = swath_lons_i[j]
            continue
        idx_nom = j * 4
        best_idx = idx_nom
        if idx_nom <= sup_nadir:
            for k in range(idx_nom, sup_nadir):
                delta = abs(dists[k] - dists[idx_nom])
                if elev[k] > (delta / tanz[j]):
                    best_idx = k
        else:
            for k in range(idx_nom, sup_nadir, -1):
                delta = abs(dists[idx_nom] - dists[k])
                if elev[k] > (delta / tanz[j]):
                    best_idx = k

        out_lat[j] = lat_sup[best_idx]
        out_lon[j] = lon_sup[best_idx]


# -----------------------------------------------------------------------------
# Pure-Python implementations (for testing)
# -----------------------------------------------------------------------------
def _process_scanline_py(i, dem_swath, lons, lats, swath_lats, swath_lons, sat_zen):
    elev = dem_swath[i]
    lon_sup = lons[i]
    lat_sup = lats[i]

    dists = _distances_along_the_lines(lon_sup, lat_sup)

    # Calculate tan of view angles
    tanz = np.tan(np.deg2rad(sat_zen[i]))

    out_lat_line = np.full_like(swath_lats[i], np.nan, dtype=np.float64)
    out_lon_line = np.full_like(swath_lons[i], np.nan, dtype=np.float64)

    _orthocorrect_line_py(elev, dists, tanz, swath_lats[i], swath_lons[i], lon_sup, lat_sup, out_lat_line, out_lon_line)

    return out_lat_line, out_lon_line
