"""Measuring where a swath places the shoreline, against coastlines it was not fitted to."""

import numpy as np
from pyproj import Geod
from scipy.ndimage import map_coordinates

EARTH = Geod(ellps="WGS84")


def _steps_along(profile):
    """Return how much *profile* changes at each step along it."""
    return np.abs(np.diff(profile))


def _peak_between_samples(steps, at):
    """Return where the true peak of *steps* lies relative to the sample *at*.

    Three samples fix a parabola, and its apex falls between them. This is the same
    refinement the control-point matcher applies to its correlation peak; without it
    a shore can only ever be placed at a whole sample, which is three times coarser
    than this record is asked to be.

    At either end of the profile there is no third sample to fit through, and the
    shore stays where the whole samples put it. Both ends need saying: running off
    the far end raises, but running off the near one does not -- counting back from
    the first step wraps round to the last, which belongs to another piece of coast,
    and the shore would be placed from it in silence.

    No guard is needed against a flat parabola. *at* is the first of the steepest
    steps, so the step before it is strictly smaller and the curvature is always
    negative; a branch for it could never be taken.
    """
    if at == 0 or at == len(steps) - 1:
        return 0.0
    before, here, after = steps[at - 1], steps[at], steps[at + 1]
    curvature = before - 2.0 * here + after
    return float(0.5 * (before - after) / curvature)


def offset_to_shoreline(profile):
    """Return how far the shore lies from the middle of *profile*, in samples.

    The profile is sampled across the coast, so the shore is where water gives
    way to land: the steepest step along it, located between samples rather than
    at one.
    """
    steps = _steps_along(profile)
    steepest = int(np.argmax(steps))
    middle = (len(profile) - 1) / 2
    return steepest + _peak_between_samples(steps, steepest) + 0.5 - middle


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


#: How coarsely the swath is scanned before the nearest sample is pinned down exactly.
#: Geolocation varies smoothly, so the coarse winner's neighbourhood holds the true
#: nearest sample; the window searched afterwards is wide enough to cover a whole
#: coarse step in each direction.
COARSE_STEP = 16


def swath_pixel_of(lons, lats, point):
    """Return the swath pixel on which *point* falls, as a line and a column.

    Searched in two passes, because a swath holds millions of samples and a global
    coastline asks this question thousands of times. A coarse scan of every
    sixteenth sample finds the right neighbourhood, and only that neighbourhood is
    then searched sample by sample.
    """
    if lons.shape[0] <= COARSE_STEP or lons.shape[1] <= COARSE_STEP:
        away = _degrees_apart((lons, lats), point)
        return np.unravel_index(np.argmin(away), away.shape)

    coarse = _degrees_apart((lons[::COARSE_STEP, ::COARSE_STEP],
                             lats[::COARSE_STEP, ::COARSE_STEP]), point)
    line, column = np.unravel_index(np.argmin(coarse), coarse.shape)
    line, column = line * COARSE_STEP, column * COARSE_STEP

    lines = slice(max(line - COARSE_STEP, 0), line + COARSE_STEP + 1)
    columns = slice(max(column - COARSE_STEP, 0), column + COARSE_STEP + 1)
    near = _degrees_apart((lons[lines, columns], lats[lines, columns]), point)
    closest = np.unravel_index(np.argmin(near), near.shape)
    return closest[0] + lines.start, closest[1] + columns.start


def coast_normal(start, end):
    """Return the unit step that crosses the coast running from *start* to *end*.

    The coast is turned a quarter circle, so that walking the returned step
    leaves the coast behind on one consistent side.
    """
    along_coast = np.array(end, dtype=float) - np.array(start, dtype=float)
    across = np.array([-along_coast[1], along_coast[0]])
    return across / np.hypot(*across)


def direction_from_track(step):
    """Return the angle of *step* from the along-track axis, in radians.

    A step is given in lines and columns: a line runs along the track and a column
    across it, so a step down the lines is at no angle to the track, a step along
    the columns is at a right angle to it, and one stepped equally in both lies
    halfway between. Real coasts run at every angle and the axis cases are the rare
    ones, so what happens between them decides most measurements.
    """
    return float(np.arctan2(step[1], step[0]))


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


def _as_unit_vectors(lons, lats):
    """Return points on the unit sphere, where angles are honest at any longitude."""
    lon, lat = np.radians(np.asarray(lons)), np.radians(np.asarray(lats))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _cone_around(lons, lats):
    """Return the middle of a swath and the angle it spans, as seen from the centre."""
    directions = _as_unit_vectors(lons, lats).reshape(3, -1)
    middle = directions.mean(axis=1)
    middle = middle / np.linalg.norm(middle)
    return middle, float(np.arccos(np.clip(middle @ directions, -1.0, 1.0)).max())


def _angle_from(middle, point):
    """Return how far *point* lies from the direction *middle*, in radians."""
    return float(np.arccos(np.clip(middle @ _as_unit_vectors(*point), -1.0, 1.0)))


def _widest_gap(lons, lats):
    """Return the largest step between neighbouring samples of a swath, in degrees."""
    return float(_degrees_apart((lons[:, :-1], lats[:, :-1]), (lons[:, 1:], lats[:, 1:])).max())


def coastline_within(coastline, lons, lats):
    """Return the points of *coastline* that fall on the swath *lons* and *lats* describe.

    A coastline database spans the globe and a pass sees a sliver of it. A point
    the swath never covered has no profile to read: sampling one would return
    whatever sits at the edge of the array and report it as a shore.

    Points far from the pass are dropped first by a cone drawn around it, which is
    only an optimisation: the cone is grown by the widest gap between neighbouring
    samples, so it cannot exclude anything the test below would have kept. It
    matters because the test below searches the whole swath for each point it is
    given, and a global coastline holds far more points than a pass can see.

    Nearness decides it, not a box drawn round the swath: a pass is a band across
    the globe, so that box holds large corners it never imaged. Nor an array index:
    :func:`swath_pixel_of` is a nearest-pixel search, so it answers for any point on
    earth, and bounds-checking its answer can reject nothing.

    How near is near enough is asked of the grid rather than fixed, because the
    geolocation may be given on sample points rather than on every pixel. A point
    counts as covered when it lies no further from its nearest sample than that
    sample lies from the one beside it -- taken on whichever side there is one, since
    at the last sample of a row there is nothing further out and a reach measured
    against the sample itself would be zero, refusing the whole trailing edge.
    """
    middle, spread = _cone_around(lons, lats)
    reach = np.radians(_widest_gap(lons, lats))
    inside = []
    for point in coastline:
        if _angle_from(middle, point) > spread + reach:
            continue
        line, column = swath_pixel_of(lons, lats, point)
        beside = column + 1 if column + 1 < lons.shape[1] else column - 1
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

def measure_against_reference(swath, swath_lons, swath_lats,
                              reference, reference_lons, reference_lats,
                              coastline, reach, least_prominence,
                              field_of_view, slant_range, local_zenith):
    """Return how far the swath sits from the reference, per crossing, in pixels.

    The whole measurement: cut the coastline to what the pass covered, measure the
    shore on both images against it, difference the two so the coastline's own error
    cancels, and express each crossing as a fraction of the pixel it was measured
    against -- both of the sample spacing and of the footprint, since the two differ
    and earlier figures for this record were quoted against the spacing.

    Each crossing is judged against the footprint at its own place in the swath, in
    the direction it was measured. The footprint is not one size for a pass: it
    grows towards the edge of the scan, and it is wider across the track than along
    it, so a single width for a whole pass would flatter the edges and punish the
    centre. The geometry comes from the caller -- the slant range and the local
    zenith angle the pass already carries -- rather than being derived here.

    A crossing that either image failed to resolve is passed over here as well as in
    the difference, so that every miss keeps the geometry of the crossing it came
    from. The reference fails as readily as the swath: it is masked and mosaicked,
    and a coast the pass sees plainly can be unreadable there.
    Without that, one miss meets two geometries and is quietly divided by both,
    reporting a measurement at a scan angle where nothing was measured.
    """
    from georeferencer.georeferencer import (
        as_pixel_fractions,
        footprint_sizes,
        footprint_towards,
    )

    covered = coastline_within(coastline, swath_lons, swath_lats)
    misses = shoreline_double_difference(swath, swath_lons, swath_lats,
                                         reference, reference_lons, reference_lats,
                                         covered, reach, least_prominence)
    spacings, footprints = [], []
    for start, end in zip(covered, covered[1:]):
        if not np.isfinite(shoreline_offset(swath, swath_lons, swath_lats, (start, end),
                                            reach, least_prominence)):
            continue
        if not np.isfinite(shoreline_offset(reference, reference_lons, reference_lats,
                                            (start, end), reach, least_prominence)):
            continue
        entering = swath_pixel_of(swath_lons, swath_lats, start)
        leaving = swath_pixel_of(swath_lons, swath_lats, end)
        normal = coast_normal(entering, leaving)
        crossing = np.mean([entering, leaving], axis=0)
        spacings.append(ground_step(swath_lons, swath_lats, crossing, normal))
        line, column = int(round(crossing[0])), int(round(crossing[1]))
        along, across = footprint_sizes(slant_range[line, column],
                                        local_zenith[line, column], field_of_view)
        footprints.append(footprint_towards(direction_from_track(normal), along, across))
    return as_pixel_fractions(misses, np.array(spacings), np.array(footprints))

def displacement_from(readings, normals):
    """Return the displacement, along track and across, that *readings* imply.

    Each reading is how far the shore moved measured across its own coast, which is
    one component of a two-component displacement -- the part along that coast is
    not measured at all. Solved together, crossings that bend away from each other
    fix both directions.

    They must be solved rather than averaged. Coastlines hold the same bearing over
    whole regions, so the directions a pass happens to sample are not spread evenly,
    and the mean of the readings leans towards whichever way the coast runs instead
    of cancelling.
    """
    return tuple(np.linalg.lstsq(np.asarray(normals, dtype=float),
                                 np.asarray(readings, dtype=float), rcond=None)[0])
