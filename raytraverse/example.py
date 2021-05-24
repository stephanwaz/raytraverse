# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""example script for raytraverse."""

import numpy as np
from raytraverse.mapper import PlanMapper, SkyMapper, ViewMapper
from raytraverse.scene import Scene
from raytraverse.renderer import Rtrace, Rcontrib
from raytraverse.sampler import SkySamplerPt, SamplerArea, SamplerSuns
from raytraverse.sky import SkyData, skycalc
from raytraverse.lightfield import LightResult
from raytraverse.evaluate import MetricSet


# -------------------
# update these values
# -------------------

# output directory where are simulation results are written
out = "outdir"
# the radiance scene files (all materials and geometries, no sources)
scene_files = "room.rad"
# a horizontal analysis plane, for this demo should be 2x2 (or change ptres)
zone = "plane.rad"
# an 8760 epw or wea file with location data. although note that wea files
# do not include dew-point, a potentially important parameter of the perez
# sky model.
epw = "weather.epw"
# an output file path for writing compressed binary results
output = "metrics.npz"


# NOTE: many of the setting overrides are for demonstration only
# and may not yield accurate or meaningful results, they are accuracy
# reductions made in order for this script to run quickly (less than 1 minute on
# a reasonably fast laptop).
# In this example, only directions are dynamically sampled, but position and
# solar sources are sampled for 1 level only at a low resolution. to sample
# points dynamically, set SamplerArea(nlev=3, jitter=True).  To dynamically
# sample suns, change  SamplerSuns(nlev=3) or other appropriate level.
# refer to the documentation for adjusting the sampling scheme used in
# directional sampling (SamplerPt, SkySamplerPt, SunSamplerPt).
def main():
    loc = skycalc.get_loc_epw(epw)
    # Make octree, manage output file directory
    scn = Scene(out, scene_files)
    # initialize sampling schemes and boundaries for position and solar source
    # sampling
    pm = PlanMapper(zone, ptres=2.0)
    sm = SkyMapper(loc=loc)
    # initialize rendering engines (note settings are appended to
    # Renderer.defaultargs, which are different from rtrace/rcontrib defaults)
    rcontrib = Rcontrib("-ab 2 -ad 4 -c 1000", scene=scn.scene)
    rtrace = Rtrace("-ab 2 -c 1", scene=scn.scene)

    # initialize sky point sampler and then call an area sampler to simulate
    # sky contribution
    sk_engine = SkySamplerPt(scn, rcontrib, accuracy=2.0)
    skysampler = SamplerArea(scn, sk_engine, accuracy=2.0, nlev=1, jitter=False)
    skyfield = skysampler.run(pm)
    print(rcontrib)
    rcontrib.reset()

    # to modify parameters for sun/pt sampler pass arguments to the ptkwargs
    # argument of SamperSuns
    ptkwargs = dict(accuracy=2.0)
    areakwargs = dict(accuracy=2.0, nlev=1, jitter=False)
    sunsampler = SamplerSuns(scn, rtrace, nlev=1,
                             ptkwargs=ptkwargs, areakwargs=areakwargs)

    daylightfield = sunsampler.run(sm, pm, specguide=skyfield)
    rtrace.reset()
    # calculate sky patch and sun contributions
    sd = SkyData(epw)

    # make a set of points to evaluate (here a regular grid at the final
    # sampling level (assuming nlev=2)
    pts = pm.point_grid(False, 1)

    # The raytraverse.lightfield.DayLightPlaneKD object holds the complete
    # sampling results. individual point data can be loaded by querying with
    # a 6 element solar position and plan cooordinate
    # the first sun in skydata:
    sun = sd.sun[0, 0:3]
    # the first point in our grid:
    pt = pts[0]
    i, d = daylightfield.query((*sun, *pt))
    sun_lightpoint = daylightfield.data[i[0]]
    # we can also query for the closest skypoint:
    j, d = daylightfield.skyplane.query(pt)
    sky_lightpoint = daylightfield.skyplane.data[j]
    # for energy conserving operations (avg luminance, illuminance, an image)
    # we can evaluate the lightpoints seperately and then add, but for
    # general analysis we need to combiine the points first:
    sun_sky_point = sky_lightpoint.add(sun_lightpoint)
    # because the sources are combined now we need to concatenate the solar
    # value onto the rest of the skyvector:
    skyvec = np.concatenate((sd.smtx[0], sd.sun[0, 3]), axis=None)

    # and now we can make an hdr image:
    vm = ViewMapper((0, -1, 0), viewangle=180)
    # file named by hour of year
    outf = f"hour_{sd.maskindices[0]:04d}.hdr"
    sun_sky_point.make_image(outf, skyvec, vm)

    # or calculate DGP and UGP:
    vol = sun_sky_point.evaluate(skyvec, vm)
    metrics = MetricSet(*vol, vm)
    print(f"DGP: {metrics.dgp} UGP: {metrics.ugp}")

    # for larger sets of metric evaluations, DaylightPlaneKD has its' own
    # evaluate function to do bulk processing (also will use multiprocessing)
    # Note that you can mask the SkyData object to limit the evaluation
    # such as to only the first day of the year:
    sd.mask = np.arange(24)

    # view directions
    vdirs = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],dtype=float)
    # choose from ["illum", "avglum", "gcr", "ugp", "dgp", "tasklum", "backlum",
    # "dgp_t1", "log_gc", "dgp_t2", "ugr", "threshold", "pwsl2", "view_area",
    # "backlum_true", "srcillum", "srcarea", "maxlum"] or inherit
    # raytraverse.evaluate.MetricSet and pass metricclass=YourMetricClass
    metrics = ['illum', 'dgp', 'ugp']
    result = daylightfield.evaluate(sd, pts, vdirs, metrics=metrics)
    result.write(output)

    # to reload these results:
    result = LightResult(output)

    # this result object stores a 4D array:
    # (sky, point, view, metric)
    # to reshape/slice results for viewing or analysis, use the pull method:
    # here we get  the south facing view for the first point in our grid:
    data, axes, names = result.pull("sky", "metric", findices=[[0], [3]])
    print("hour", *axes[2])
    for d, h in zip(data[0], axes[1]):
        print(h, *d)


if __name__ == '__main__':
    main()
