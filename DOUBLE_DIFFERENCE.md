# Measuring how well a swath is georeferenced, by double difference

## What this is for

The goal is one number, honestly obtained: **how far our navigated swath sits from
the Blue Marble reference, in local swath pixels, along track and across track
separately.** That is the quantity a geolocation requirement is written against,
and it is the quantity this tool measures.

It is a *measurement*, not a fit. It belongs in the georeferencer for the same
reason the control-point search does: this package measures displacements and
makes no claim about what they mean. Nothing here optimises anything, and nothing
here needs to know about clocks, attitudes or orbital elements.

## Why the obvious measurement is not good enough

The obvious thing is to take each chip, match it against the navigated swath, and
report the displacement. Call that the **single difference**. It is what the
`gcp_x_displacement` / `gcp_y_displacement` variables on a product already hold.

The trouble is that a single difference does not measure only geolocation error.
It measures geolocation error *plus everything the measurement process itself
contributes*:

- **Resampling.** The swath is warped into the reference's grid before matching.
  Interpolation shifts edges, and it shifts them differently depending on where a
  pixel falls relative to the grid.
- **Matcher bias.** The correlation surface is not symmetric. Peak location has a
  bias that depends on the texture in the window, on how the Laplacian responds to
  it, and on the sub-pixel interpolation used to refine the peak.
- **Chip content.** A coastline chip with a strong straight edge is located
  differently from a chip with a diffuse one, whatever imagery it is matched
  against.

None of these are properties of our navigation, and all of them are baked into a
single difference. They are the most likely explanation for the chip-to-chip
scatter we have never been able to attribute.

## The double difference

Measure each chip **twice, in the same projection, through the same matcher**:

| term | what is matched | what it contains |
|---|---|---|
| **A** | chip against the **navigated swath** | geolocation error + resampling + matcher bias + chip content |
| **B** | chip against the **Blue Marble reference** | resampling + matcher bias + chip content |

Then

```
double difference = A − B
```

Everything common to the two cancels. What is left is the part that differs
between the swath and the reference — which is our geolocation error, and is the
number we want. This is the same reasoning that makes a differential measurement
preferable to an absolute one anywhere else: the shared systematics drop out.

For the cancellation to be real, the two matches must be made **as identically as
possible**:

- the same chip, at the same location, with the same window size;
- the same projection and the same grid, so both images are sampled the same way;
- the same matcher, the same Laplacian pre-filter, the same sub-pixel refinement;
- the same acceptance criteria, so a chip that is rejected in one term is dropped
  from both rather than contributing to one alone.

Term **B** is not expected to be zero. If it were, it would not be worth
measuring. It is expected to be a small, structured bias that varies with chip
content and grid alignment — and that is exactly the part we want removed from A.

## Units, and why they are per chip

The result is reported in **local swath pixels**, along track and across track
separately, because the AVHRR footprint is strongly anisotropic:

- the along-track footprint roughly doubles from nadir to the swath edge;
- the across-track footprint grows about five-fold over the same span.

A miss of 500 m at nadir and a miss of 500 m at the swath edge are therefore not
the same error, and averaging them in metres hides that. Each chip's displacement
is divided by the size of *its own* footprint, **separately in each axis**.

### Footprint, not sample spacing

These are not the same thing, and using the wrong one silently biases the result:

- **Along track**, the line **spacing** is essentially constant across the whole
  swath — one scan line, wherever you are. The along-track **footprint** is not:
  it grows by about a factor of two from nadir to the swath edge, because the
  instantaneous field of view is projected down a longer slant range.
- **Across track**, both grow, and the footprint grows fastest — by roughly a
  factor of five from nadir to the edge, because the ground is also tilted away
  from the line of sight.

So dividing an along-track miss by the line spacing would understate the pixel at
the swath edge by up to a factor of two, and flatter the result exactly where the
geometry is worst. The divisor must be the footprint.

With `beta` the instantaneous field of view, `s` the slant range from the platform
to that chip, and `psi` the local zenith angle where the line of sight meets the
ground:

```
along-track footprint   =  beta * s
across-track footprint  =  beta * s / cos(psi)
```

The first term, `beta * s`, is the pure range effect and applies to both axes. The
extra `1 / cos(psi)` on the across-track axis is the foreshortening of a tilted
surface, and it is why the across-track footprint outgrows the along-track one.
At nadir both reduce to `beta * h` and the footprint is square; at the edge of an
AVHRR swath they are roughly 2.4 km along by 6.5 km across, against 1.1 km square
at nadir.

Both `s` and `psi` come from the pass's own geolocation and the platform position
at that scan line, so nothing is assumed about the instrument beyond its field of
view.

### Take these from pyorbital; do not re-derive them

Every quantity above already exists upstream, and re-deriving any of it here would
be a second implementation to keep in step with the first. Specifically:

- **`psi`, the local zenith angle, is already an argument.**
  `measure_swath_displacement` receives `sat_zen`, computed by the reader from the
  same geolocation the chips were placed with. Use it. Recomputing it here would
  risk measuring against a different geometry than the one being assessed, which
  would quietly invalidate the whole comparison.
- Where a zenith angle is *not* to hand, `pyorbital.geoloc.get_sensor_angles`
  returns it (and the azimuth) for ground coordinates, given the orbit.
- **The platform and pixel positions** come from `pyorbital.geoloc.compute_pixels`
  and the orbit's own `get_position`, both in earth-centred coordinates, so the
  slant range is the distance between them and needs no spherical geometry of our
  own. `pyorbital.geoloc.get_lonlatalt` converts those positions when longitudes
  and latitudes are wanted instead.
- The **ellipsoid** is `pyorbital.geoloc`'s, not a radius written down here.

The rule this follows is the one the package already lives by: the georeferencer
measures displacements in imagery. Orbital geometry is pyorbital's subject, and
this tool should import it rather than restate it.

### Axes

Because the matching is done in the satellite's native geometry, the two axes of
the measured displacement are already along track and across track; no rotation is
applied. The across-track axis is the one the footprint stretches, so the two
divisors must not be swapped — a result that improves when they are swapped is a
sign the axes have been mixed up somewhere.

## What it reports

Per pass:

- the number of chips that survived acceptance in **both** terms;
- the median and the scatter of the double difference, per axis, in local pixels;
- the same for terms A and B separately, so the size of the correction is visible
  rather than hidden.

Across a set of passes, the same aggregated, plus the systematic offset (the
median, which is the accuracy reading) reported separately from the chip-to-chip
scatter (the spread, which is the precision reading). Those two answer different
questions and must not be collapsed into one figure.

## What this does NOT measure

Stating the limits plainly, because the number is only as good as its provenance:

- **It does not validate Blue Marble.** The double difference measures our swath
  against the reference *as it is*. If the reference is itself displaced, this
  measurement will not say so. That is a separate study — comparing the reference
  against an independent coastline database — and its result belongs alongside
  this one, combined in quadrature, not folded into it.
- **It does not account for terrain.** Chips are located on coastlines at sea
  level. Elevation displaces a pixel at any non-zero view angle, and that
  displacement is real geolocation error which this measurement will attribute to
  us — correctly, but it is worth knowing which part of the total it is.
- **It does not measure passes with no chips.** A pass over cloud or open ocean
  yields nothing, and the set that *can* be measured is not a random sample of the
  record.
- **It is not a fit quality.** The residual left over by a least-squares fit is
  not this number and must not be reported as if it were. A looser fit always
  leaves a smaller residual while navigating no better.

## Shape of the interface

The tool needs, per pass: the calibrated dataset (for the swath imagery and its
geolocation), and the path to the reference image. It returns the per-chip double
differences and their per-axis summaries. It does not take, and must not take, a
set of fitted parameters — measuring against a fit is what the pipeline does
elsewhere, and mixing the two is how a measurement quietly becomes a
self-assessment.
