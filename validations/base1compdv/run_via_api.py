#!/usr/bin/env python

"""Demo of using the api to script sampling and evaluation for siz configurations
of a simple office model with three desk locations. reports simulation time
and accuracy metrics against a high quality reference simulation. Duplicates
the results of the command line calls in run.sh located in the same directory."""

import os
import shutil
import time

import numpy as np
from clasp.script_tools import try_mkdir, sglob

from raytools.mapper import ViewMapper
from raytraverse.integrator import Integrator
from raytraverse.lightfield import LightPlaneKD, ResultAxis, LightResult
from raytraverse.mapper import PlanMapper
from raytraverse.renderer import Rcontrib, Rtrace
from raytraverse.sampler import SkySamplerPt, SamplerArea
from raytraverse.scene import Scene
from raytraverse.sky import SkyData


def get_sky_sampler(rayargs, scn, skyres, accuracy):
    """load a static point sky sampler

    make sure rcontrib is reset when calling for a second (or more) time!

    Parameters
    ----------
    rayargs : str
        arguments to pass to Rcontrib
    scn : raytraverse.scene.Scene
        the scene to sample
    skyres : int
        grid size for skypatch division, number of sky patches will be skyres^2
    accuracy : float
        accuracy parameter for direction sampling

    Returns
    -------
    raytraverse.sampler.SamplerArea
        initialized SamplerArea with SkySamplerPt engine

    """
    rcontrib = Rcontrib(rayargs=rayargs, scene=scn.scene, skyres=skyres)
    # for scenes with complex and small opennings, do not override the default
    # nlev (5)
    skyengine = SkySamplerPt(scn, rcontrib, accuracy=accuracy, nlev=4)
    # for dynamic area sampling, set nlev > 1 and jitter to True.
    return SamplerArea(scn, skyengine, nlev=1, jitter=False)


def sample_with_check(scn, pm, skysampler, overwrite=False):
    """pattern to only run sampling if results do not exist

    Parameters
    ----------
    scn : raytraverse.scene.Scene
    pm : raytraverse.mapper.PlanMapper
        planmapper initialized with a list of points (the same as the
        sensors used for evaluation. it is ok if these are 6-component as the
        planmapper will ignore the directions.
    skysampler : raytraverse.sampler.SamplerArea
        initialized SamplerArea with SkySamplerPt engine
    overwrite : bool
        set to true to force resampling even if output files exist

    Returns
    -------
    skyf : raytraverse.lightfield.LightPlaneKD
        representing the full sky patch contributions
    dskyf : raytraverse.lightfield.LightPlaneKD
        matching the structure/sampling of skyf (made via skysampler.repeat())
        and representing the direct sky patch contributions
    """

    ###################################################################
    ### first load/run full ambient bounce sky coefficient sampling ###
    ###################################################################
    try:
        if overwrite:
            raise OSError
        skyfield = LightPlaneKD(scn, f"{scn.outdir}/{pm.name}/sky_points.tsv",
                                pm, "sky")
    except OSError:
        overwrite = True
        # turn logging on here since this steps can take a little while
        # so I want to see the progress bar
        scn.dolog = True
        skyfield = skysampler.run(pm)
        scn.dolog = False
        try_mkdir(f"{scn.outdir}/{pm.name}")
        # look for reflections (this is important if you have specular materials
        # that could cause glare. with really complex geometry, but known normals
        # that have specular sources of concern, it is better to manually generate
        # this file and copy it to this location. the format of the file is:
        # 1 0 0
        # 0 0 1
        # 0 -1 0
        # where each line is a surface normal to check for specular reflection
        # of the sun (this happens at both the sun sampling (3-component) or
        # evaluation (1-comp DV) steps.
        # if you do not have specular materials that could reflect the sun,
        # simply skip/comment out these lines.
        reflf = f"{scn.outdir}/{pm.name}/reflection_normals.txt"
        if not os.path.isfile(reflf):
            refl = scn.reflection_search(skyfield.vecs)
            if refl.size > 0:
                np.savetxt(reflf, refl)

    ############################################################
    ### Now run -ab 0 sky patch run for 1-comp DV evaluation ###
    ############################################################
    try:
        if overwrite:
            raise OSError
        dskyfield = LightPlaneKD(scn,
                                 f"{scn.outdir}/{pm.name}/skydcomp_points.tsv",
                                 pm, "skydcomp")
    except OSError:
        # copy values before initializing new sampler
        skyres = skysampler.engine.engine.skyres
        accuracy = skysampler.engine.accuracy
        skysampler.engine.engine.reset()
        skysampler = get_sky_sampler("-ab 0 -ss 0 -lr 1", scn, skyres, accuracy)
        dskyfield = skysampler.repeat(skyfield, "skydcomp")
    # reset rcontrib, otherwise segmentation faults abound...
    skysampler.engine.engine.reset()
    return skyfield, dskyfield


def run_opt(opt, geo, pm, skyres=12, vlt=0.7):
    """initialize components to sample sky and directsky for 1compdv run

    Parameters
    ----------
    opt : str
        the name of the option/geometry to run, should be unique for each
        seperate geometry configuration
    geo : str
        space seperated list of radiance scene files (passed to oconv)
    pm : raytraverse.mapper.PlanMapper
        planmapper initialized with a list of points (the same as the
        sensors used for evaluation. it is ok if these are 6-component as the
        planmapper will ignore the directions.
    skyres : int
        grid size for skypatch division, number of sky patches will be skyres^2
    vlt : float
        visible light tranmission of primary daylight apertures, used to scale
        the accuracy, matters especially when sampling very low transmission
        scenes when accurate contrast is important.

    Returns
    -------
    skyf : raytraverse.lightfield.LightPlaneKD
        representing the full sky patch contributions
    dskyf : raytraverse.lightfield.LightPlaneKD
        matching the structure/sampling of skyf (made via skysampler.repeat())
        and representing the direct sky patch contributions
    """

    # set log to true to get progress bars, but this can create a lot of noise
    # on the stderr, so if you are printing other messages better to make quiet
    # this can also be toggled on and off throughout the process by setting
    # scn.dolog = False, scn.dolog = True
    scn = Scene(opt, geo, log=False)
    accuracy = vlt / 0.64
    # default rcontrib arguments in raytraverse are quite sufficient for
    # accurate results across a wide range of conditions, here the parameters
    # are relaxed as this was part of a comparison study of other methods that
    # used similar parameters.
    skysampler = get_sky_sampler("-ab 6 -ss 0", scn, skyres, accuracy)
    return sample_with_check(scn, pm, skysampler)


def evaluate_opt(skyf, dskyf, sensors, sd, skymask=None, srad=1.0):
    """perform 1compdv evaluation for sensors with individual directions

    Parameters
    ----------
    skyf : raytraverse.lightfield.LightPlaneKD
        representing the full sky patch contributions
    dskyf : raytraverse.lightfield.LightPlaneKD
        matching the structure/sampling of skyf (made via skysampler.repeat())
        and representing the direct sky patch contributions
    sensors : np.array
        6 component location and direction vectors
    sd : raytraverse.sky.SkyData
        the full skydata
    skymask : None, np.array, int, list, tuple
        indices (referring to rows of initializing skydata) to include in
        evaluation
    srad : float
        sampling radius around direct sun / reflection vectors for direct view
        sampling, for specular transmission/reflection, leave at 1, for
        semi-specular scattering, this value needs to be larger (20) in the
        cases demonstrated here (trans, nmt)

    Returns
    -------

    raytraverse.lightfield.LightResult
        a light result with three axes: (sky, point, metric)
    """

    # these are the objects needed for a 1-compDV run, look at
    # raytraverse.api.get_integrator() for other patterns, and to load the
    # lightplanes from a file directory. since here we are passing the
    # lightplanes directly it is easier to directly initialize
    rtrace = Rtrace(rayargs="-ab 0", scene=skyf.scene.scene)
    itg = Integrator(skyf, dskyf, sunviewengine=rtrace, dv=True)
    result = []
    # in this case I am not masking the sky at all, but this, by passing None
    # any mask is reset, but this is where one could pass a single index, or
    # list of indices to reduce the evaluation
    sd.mask = skymask
    # because the sensors are 6-component (each with their own direction) I
    # loop through each one for evaluation. If I were to pass locations and
    # directions seperately to evaluate (so every location would be run for
    # every direction) then I could do it like this:
    # return itg.evaluate(sd, sensors, sdirs, viewangle=180, suntol=0.0,
    #                     metrics=("illum", "dgp", "ugp"), blursun=False,
    #                     resamprad=srad, coercesumsafe=True)
    for sensor in sensors:
        point = sensor[0:3]
        vm = ViewMapper(sensor[3:6], 180)
        result.append(itg.evaluate(sd, point, vm, metrics=("illum", "dgp", "ugp"),
                                   suntol=0.0, blursun=False, resamprad=srad,
                                   coercesumsafe=True))
    data = np.squeeze(np.concatenate([r.data for r in result], axis=1), 2)
    ptaxis = ResultAxis(sensors, "point")
    return LightResult(data, result[0].axes[0], ptaxis, result[0].axes[3])


def main():
    """this script duplicates the command line runs in run.sh which
    performs an annual 1compdv analysis of three static desk positions under
    6 different facade conditions."""
    t0 = time.time()

    # global settings:
    skyres = 12

    configurations = dict(
            glz="scene/options/clear.rad scene/RAD/office.rad",
            ecg="scene/options/ec.rad scene/RAD/office.rad",
            shd="scene/options/fabric.rad scene/RAD/office.rad",
            trn="scene/options/20per_toptranslucent.rad scene/RAD/office.rad",
            ngl="scene/options/north_glass.rad scene/RAD/office_n.rad",
            nmt="scene/options/north_metal.rad scene/RAD/office_n.rad")

    skydata = SkyData("scene/refs/skydata.txt", minalt=0, mindiff=0, mindir=50,
                      skyres=skyres)

    # since some configurations are rotated,
    # specify the two different sensor grids
    pt_s = np.loadtxt("scene/POINTS/sensors.txt")
    pt_n = np.loadtxt("scene/POINTS/sensors_n.txt")

    for opt, scene_geo in configurations.items():
        t = time.time()
        # specific configuration options to match what the command line
        # calls I made are doing.
        vlt = 0.7
        resamprad = 1.0
        if opt in ("ecg", "shd"):
            vlt = 0.1
        elif opt == "trn":
            vlt = 0.3
        if "_n" in scene_geo:
            pts = pt_n
        else:
            pts = pt_s
        if opt in ("nmt, trn"):
            resamprad = 20.0
        pm = PlanMapper(pts)

        # these are the two main function calls that do all the work
        skyf, dskyf = run_opt(f"{opt}_api", scene_geo, pm, skyres=skyres, vlt=vlt)
        lr = evaluate_opt(skyf, dskyf, pts, skydata, srad=resamprad)
        # do stuff with the results here.... (including saving)

        # this part is just for validation
        opttime = time.time() - t
        refd = []
        for f in sglob(f"scene/refrt/*_{opt}_v*_refrt.tsv"):
            refd.append(np.loadtxt(f, skiprows=1)[:, [4, 5, 12]])
        refd = np.stack(refd, axis=1)
        d = lr.data - refd
        # make illuminance relative
        d[..., 0] = d[..., 0]/refd[..., 0]
        np.set_printoptions(3, suppress=True)
        print(opt, f"runtime: {opttime:.01f}", "MSD:", np.average(d, axis=(0,1)), "MAD:",
              np.average(np.abs(d), axis=(0,1)))
        # Watch out! I'm deleting everything here...
        shutil.rmtree(f"{opt}_api")
    totaltime = time.time() - t0
    print(f"totaltime: {totaltime:.01f}, perpoint: {totaltime/18:.01f}")


if __name__ == "__main__":
    main()
