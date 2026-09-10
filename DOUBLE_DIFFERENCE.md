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

The trouble is what those chips are cut from. They are selected out of the Blue
Marble reference itself, so a single difference measures the swath against the
reference and nothing more. If the reference is displaced, every pass corrected
against it is displaced identically: the passes agree beautifully with each other,
with the reference, and with every internal consistency check available — and all
of them are wrong together. Withheld control points cannot see it either, because
they share the same reference.

Only an independent source reveals that common error. Matching a chip cut from the
reference back against the reference is an autocorrelation and is identically
zero, so it can never serve as the second term.

## The double difference

Use a template neither image was fitted to: a **coastline chip from GSHHG**, an
independent shoreline database. Match it against both images, at the same place,
through the same method:

| term | what is measured | what it contains |
|---|---|---|
| **A** | where the **navigated swath** puts the shoreline, against the GSHHG chip | our geolocation error + the chip's own error |
| **B** | where **Blue Marble** puts the shoreline, against the same GSHHG chip | the reference's error + the same chip error |

Then

```
offset(pass -> GSHHG)  -  offset(Blue Marble -> GSHHG)  =  offset(pass -> Blue Marble)
```

The GSHHG chip is *identical* in the two terms, so everything it contributes —
coastline definition, tides, the database's own accuracy, the sampling of the
profile — is common mode and subtracts out exactly. What remains is how well the
pass matches the reference, per chip, free of the reference-versus-database
disagreement that otherwise dominates the scatter.

This is the whole point: that disagreement is large. Blue Marble against GSHHG
runs to several hundred metres, which is the same order as the entire requirement,
and it currently sits inside every per-chip number we quote. It does not belong
there, because it is not our navigation.

For the cancellation to be exact, the two measurements must be made **as
identically as possible**:

- the same GSHHG chip, at the same location, sampled the same way;
- the same projection and grid, so both images are read at the same points;
- the same shore-finding, the same sub-sample refinement;
- the same acceptance criteria, so a chip rejected in one term is dropped from
  both rather than contributing to one alone.

### The decomposition this gives the report

| term | what it is | how it is measured | does it average down? |
|---|---|---|---|
| pass -> Blue Marble | our fit quality | per chip; GSHHG cancels | no — this is the per-pixel term |
| Blue Marble -> truth | the reference's own error | many patches, once, for the campaign | yes — one number for the whole record |

The two are combined in quadrature and reported separately, never folded together.

### What cannot serve as the independent source

The ocean mask shipped alongside the reference (`c1_b1_c2_b2_oceanmask.tif`) is at
Blue Marble's own posting and is co-registered to it. Whatever geolocation error
Blue Marble carries, that mask carries identically. It is a good matching aid and
useless as a check: validating against it would confirm the reference by
construction.

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
- **It does not remove tides or coastline change.** Those are in the GSHHG chip,
  and because the chip is common to both terms they cancel from the double
  difference — but they do *not* cancel from the separate Blue-Marble-to-truth
  study, where they are part of the irreducible noise.
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
geolocation), the reference image, and the coastline database the chips are drawn
from. It returns the per-chip double differences and their per-axis summaries.
The coastline source is an argument rather than an assumption, because the whole
construction rests on it being independent of both images -- a caller that passes
the reference's own ocean mask would get a measurement that confirms the reference
by construction, and the interface should make that an obvious mistake rather than
a silent one. It does not take, and must not take, a
set of fitted parameters — measuring against a fit is what the pipeline does
elsewhere, and mixing the two is how a measurement quietly becomes a
self-assessment.
