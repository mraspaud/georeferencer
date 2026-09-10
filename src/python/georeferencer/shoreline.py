"""Measuring where a swath places the shoreline, against coastlines it was not fitted to."""

import numpy as np
from pyproj import Geod
from scipy.ndimage import map_coordinates

EARTH = Geod(ellps="WGS84")


def _steps_along(profile):
    """Return how much *profile* changes at each step along it."""
    return np.abs(np.diff(profile))


def offset_to_shoreline(profile):
    """Return how far the shore lies from the middle of *profile*, in samples.

    The profile is sampled across the coast, so the shore is where water gives
    way to land: the steepest step along it.
    """
    steps = _steps_along(profile)
    middle = (len(profile) - 1) / 2
    return float(np.argmax(steps)) + 0.5 - middle


def profile_along(image, at, direction, reach):
    """Return the image values along *direction* through *at*, out to *reach* either side."""
    steps = np.arange(-reach, reach + 1)
    rows = at[0] + steps * direction[0]
    columns = at[1] + steps * direction[1]
    return map_coordinates(image, [rows, columns], order=1, mode="nearest")


def coastline_from(path):
    """Return the coastline in the shapefile at *path*, as sequences of lon/lat points.

    GSHHG is distributed as shapefiles, and reading that format is a solved problem:
    this hands it to pyshp rather than keeping a second reader of its own to get
    wrong. pycoast reads the same data but only to draw it, and its geometry is not
    part of its public surface.
    """
    import shapefile

    with shapefile.Reader(path) as reading:
        return [[(float(lon), float(lat)) for lon, lat in shape.points]
                for shape in reading.shapes()]


def _degrees_apart(one, other):
    """Return how far apart two lon/lat points lie, in degrees, allowing for the meridian.

    Either side may be a whole grid of points rather than a single one, so that the
    same measure serves both for comparing two places and for searching a swath.
    The eastward distance shrinks towards the poles, and it is *one*'s latitude that
    sets that scaling.
    """
    (one_lon, one_lat), (other_lon, other_lat) = one, other
    eastwards = ((one_lon - other_lon + 180) % 360 - 180) * np.cos(np.radians(one_lat))
    return np.hypot(eastwards, one_lat - other_lat)


def swath_pixel_of(lons, lats, point):
    """Return the swath pixel on which *point* falls, as a line and a column."""
    away = _degrees_apart((lons, lats), point)
    return np.unravel_index(np.argmin(away), away.shape)


def coast_normal(start, end):
    """Return the unit step that crosses the coast running from *start* to *end*.

    The coast is turned a quarter circle, so that walking the returned step
    leaves the coast behind on one consistent side.
    """
    along_coast = np.array(end, dtype=float) - np.array(start, dtype=float)
    across = np.array([-along_coast[1], along_coast[0]])
    return across / np.hypot(*across)


def ground_step(lons, lats, at, direction):
    """Return how far across the ground one step along *direction* carries, in metres."""
    _, here_east, beyond_east = profile_along(lons, at, direction, reach=1)
    _, here_north, beyond_north = profile_along(lats, at, direction, reach=1)
    _, _, distance = EARTH.inv(here_east, here_north, beyond_east, beyond_north)
    return distance


def crosses_a_coast(profile, least_prominence):
    """Say whether the largest step in *profile* rises *least_prominence* above the median step."""
    steps = _steps_along(profile)
    return bool(np.max(steps) - np.median(steps) >= least_prominence)


def coastline_within(coastline, lons, lats):
    """Return the points of *coastline* that fall on the swath *lons* and *lats* describe.

    A coastline database spans the globe and a pass sees a sliver of it. A point
    the swath never covered has no profile to read: sampling one would return
    whatever sits at the edge of the array and report it as a shore.

    Nearness decides it, not a box drawn round the swath: a pass is a band across
    the globe, so that box holds large corners it never imaged. Nor an array index:
    :func:`swath_pixel_of` is a nearest-pixel search, so it answers for any point on
    earth, and bounds-checking its answer can reject nothing.

    How near is near enough is asked of the grid rather than fixed, because the
    geolocation may be given on sample points rather than on every pixel. A point
    counts as covered when it lies no further from its nearest sample than that
    sample lies from the one beside it.
    """
    inside = []
    for point in coastline:
        line, column = swath_pixel_of(lons, lats, point)
        beside = min(column + 1, lons.shape[1] - 1)
        here = (lons[line, column], lats[line, column])
        neighbour = (lons[line, beside], lats[line, beside])
        if _degrees_apart(here, point) <= _degrees_apart(here, neighbour):
            inside.append(point)
    return inside


def shoreline_offset(image, lons, lats, segment, reach, least_prominence):
    """Return how far the swath places the shore from one *segment* of coastline.

    A segment is the two points spanning a single crossing. The offset is given
    in metres, positive where the swath places the shore further along the coast
    normal than the segment does, and is not a number where the profile holds no
    step prominent enough to be a shore.
    """
    start, end = segment
    entering = swath_pixel_of(lons, lats, start)
    leaving = swath_pixel_of(lons, lats, end)
    crossing = np.mean([entering, leaving], axis=0)
    normal = coast_normal(entering, leaving)
    profile = profile_along(image, crossing, normal, reach)
    if not crosses_a_coast(profile, least_prominence):
        return np.nan
    step = ground_step(lons, lats, crossing, normal)
    return offset_to_shoreline(profile) * step


def shoreline_offsets(image, lons, lats, coastline, reach, least_prominence):
    """Return the offset for each crossing along *coastline* that shows a shore, in metres.

    Crossings whose profile holds no step prominent enough to be a shore are
    left out, so the result is shorter than the number of segments given.
    """
    segments = zip(coastline, coastline[1:])
    measured = np.array([shoreline_offset(image, lons, lats, segment, reach, least_prominence)
                         for segment in segments])
    return measured[np.isfinite(measured)]


def shoreline_double_difference(swath, swath_lons, swath_lats,
                                reference, reference_lons, reference_lats,
                                coastline, reach, least_prominence):
    """Return how far the swath places the shore relative to the reference, per crossing.

    Each crossing is measured twice against the same piece of coastline: once on
    the swath, once on the reference. The coastline's own error -- how it defines a
    shore, tides, its own accuracy -- is identical in the two and subtracts out, so
    what remains is the swath against the reference alone, free of a disagreement
    that is otherwise the same size as the whole requirement.

    A crossing that only one of the two resolves is dropped from both. Keeping it
    would let it enter the difference with nothing to cancel against, and bring
    back on that crossing exactly the error this construction removes.
    """
    segments = list(zip(coastline, coastline[1:]))
    ours = np.array([shoreline_offset(swath, swath_lons, swath_lats, segment,
                                      reach, least_prominence) for segment in segments])
    theirs = np.array([shoreline_offset(reference, reference_lons, reference_lats, segment,
                                        reach, least_prominence) for segment in segments])
    both = np.isfinite(ours) & np.isfinite(theirs)
    return ours[both] - theirs[both]
