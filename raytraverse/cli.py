# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""Console script for raytraverse."""
import os
import shutil
from glob import glob

import numpy as np

from clasp import click
import clasp.click_ext as clk

import raytraverse
from raytraverse import translate
from raytraverse import api
from raytraverse.lightfield import LightPlaneKD, ResultAxis
from raytraverse.lightfield import LightResult
from raytraverse.mapper import PlanMapper, SkyMapper, ViewMapper
from raytraverse.utility.cli import np_load, np_load_safe, shared_pull, pull_decs
from raytraverse.scene import Scene
from raytraverse.renderer import Rtrace, Rcontrib
from raytraverse.sampler import SkySamplerPt, SamplerArea, SamplerSuns
from raytraverse.sky import SkyData, skycalc


@click.group(chain=True, invoke_without_command=True)
@click.option('-out', type=click.Path(file_okay=False))
@click.option('-config', '-c', type=click.Path(exists=True),
              help="path of config file to load")
@click.option('--template/--no-template', is_eager=True,
              callback=clk.printconfigs,
              help="write default options to std out as config")
@click.option('-n', default=None, type=int,
              help='sets the environment variable RAYTRAVERSE_PROC_CAP set to'
                   ' 0 to clear (parallel processes will use cpu_limit)')
@click.option('--opts', '-opts', is_flag=True,
              help="check parsed options")
@click.option('--debug', is_flag=True,
              help="show traceback on exceptions")
@click.version_option(version=raytraverse.__version__)
@click.pass_context
def main(ctx, out=None, config=None, n=None,  **kwargs):
    """the raytraverse executable is a command line interface to the raytraverse
    python package for running and evaluating climate based daylight models.
    sub commands of raytraverse can be chained but should be invoked in the
    order given.

    the easiest way to manage options is to use a configuration file,
    to make a template::

        raytraverse --template > run.cfg

    after adjusting the settings, than each command can be invoked in turn and
    any dependencies will be loaded with the correct options, a complete run
    and evaluation can then be called by::

        raytraverse -c run.cfg skyrun sunrun

    as all required precursor commands will be invoked automatically as needed.
    """
    raytraverse.io.set_nproc(n)
    ctx.info_name = 'raytraverse'
    clk.get_config_chained(ctx, config, None, None, None)
    ctx.obj = dict(out=out)


@main.command()
@click.option('-out', type=click.Path(file_okay=False))
@click.option('-scene',
              help='space separated list of radiance scene files (no sky) or '
                   'precompiled octree')
@click.option('--reload/--no-reload', default=True,
              help='if a scene already exists at OUT reload it, note that if'
                   ' this is False and overwrite is False, the program will'
                   ' abort')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Warning! if set to True all files in'
                   'OUT will be deleted')
@click.option('--log/--no-log', default=True,
              help='log progress to stderr')
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def scene(ctx, out=None, opts=False, debug=False, version=None, **kwargs):
    """define scene files for renderer and output directory

    Effects
    ~~~~~~~
        - creates outdir and outdir/scene.oct
    """
    if out is not None:
        ctx.obj['out'] = out
    s = Scene(ctx.obj['out'], **kwargs)
    ctx.obj['scene'] = s


engine_opts = [
 click.option('-accuracy', default=1.0,
              help='a generic accuracy parameter that sets the threshold'
                   ' variance to sample. A value of 1 will have a sample count'
                   ' at the final sampling level equal to the number of'
                   ' directions with a contribution variance greater than .25'),
 click.option("-vlt", default=0.64,
              help="primary transmitting vlt, used to scale the accuracy "
                   "parameter to the expected scene variance. Optional, but "
                   "helpful with, for example, electrochromic glazing or "
                   "shades"),
 click.option('-idres', default=5,
              help='the initial directional sampling resolution. each side'
                   ' of the sampling square (representing a hemisphere) will'
                   ' be subdivided 2^idres, yielding 2^(2*idres) samples and'
                   ' a resolution of 2^(2*idres)/(2pi) samples/steradian. this'
                   ' value should be smaller than 1/2 the size of the smallest'
                   ' view to an aperture that should be captured with 100%'
                   ' certainty'),
 click.option('-rayargs',
              help='additional arguments to pass to the  rendering engine'),
 click.option('--default-args/--no-default-args', default=True,
              help='use raytraverse defaults before -rayargs, if False, uses'
                   ' radiance defaults'),
    ]


@main.command()
@click.option("-static_points", callback=np_load,
              help="points to simulate, this can be a .npy file, a whitespace "
                   "seperated text file or entered as a string with commas "
                   "between components of a point and spaces between points. "
                   "points should either all be 3 componnent (x,y,z) or 6"
                   " component (x,y,z,dx,dy,dz) but the dx,dy,dz is ignored")
@click.option("-zone", callback=np_load_safe,
              help="zone boundary to dynamically sample. can either be a "
                   "radiance scene file defining a plane to sample or an array "
                   "of points (same input options as -static_points). Points"
                   "are used to define a convex hull with an offset of "
                   "1/2*ptres in which to sample. Note that if static_points"
                   "and zone are both give, static_points is silently ignored")
@click.option("-ptres", default=1.0,
              help="initial sampling resolution for points")
@click.option("-rotation", default=0.0,
              help="positive Z rotation for point grid alignment")
@click.option("-jitterrate", default=0.5,
              help="fraction of each axis to jitter over")
@click.option("-zheight", type=float, default=None,
              help="replaces z in points or zone")
@click.option("-name", default="plan",
              help="name for zone/point group (impacts file naming)")
@click.option("--printdata/--no-printdata", default=False,
              help="if True, print areamapper positions (either boundary or "
                   "static points)")
@click.option("-printlevel", type=int,
              help="print a set of sun positions at sampling level "
                   "(overrides printdata)")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def area(ctx, static_points=None, zone=None, opts=False, debug=False,
         version=None, printdata=False, printlevel=None, **kwargs):
    """define sampling area

    Effects
    ~~~~~~~
        - None

    """
    if zone is None:
        zone = static_points
        # used to override jitter and sampling levels in samplers
        ctx.obj['static_points'] = True
    else:
        ctx.obj['static_points'] = False
    if zone is None:
        raise ValueError("one of 'static_points' or 'zone' must be defined")
    ctx.obj['planmapper'] = PlanMapper(zone, **kwargs)
    if printdata or printlevel is not None:
        pm = ctx.obj['planmapper']
        if printlevel is not None:
            xyz = pm.point_grid(level=printlevel, jitter=False)
        elif ctx.obj['static_points']:
            xyz = zone
        else:
            xyz = np.array(pm.borders()).reshape(-1, 3)
        for x in xyz:
            print("{: >10.05f} {: >10.05f} {: >10.05f}".format(*x))


@main.command()
@click.option("-loc", callback=np_load_safe,
              help="""can be a number of formats:

    1. a string of 3 space seperated values (lat lon mer)
       where lat is +west and mer is tz*15 (matching gendaylit).
    2. a string of comma seperated sun positions with multiple items
       seperated by spaces: "0,-.7,.7 .7,0,.7" following the shape
       requirements of 3.
    3. a file loadable with np.loadtxt) of shape
       (N, 2), (N,3), (N,4), or (N,5):

            a. 2 elements: alt, azm (angles in degrees)
            b. 3 elements: dx,dy,dz of sun positions
            c. 4 elements: alt, azm, dirnorm, diffhoriz (angles in degrees)
            d. 5 elements: dx, dy, dz, dirnorm, diffhoriz.

    4. path to an epw or wea formatted file: solar positions are generated
       and used as candidates unless --epwloc is True.
    5. None (default) all possible sun positions are considered

in the case of a location, sun positions are considered valid when
in the solar transit for that location. for candidate options (2., 3., 4.),
sun positions are drawn from this set (with one randomly chosen from all
candidates within adaptive grid.""")
@click.option("-skyro", default=0.0,
              help="counterclockwise sky-rotation in degrees (equivalent to "
                   "clockwise project north rotation)")
@click.option("-sunres", default=20.0,
              help="initial sampling resolution for suns")
@click.option("-jitterrate", default=0.5,
              help="fraction of each axis to jitter over")
@click.option("-name", default="suns",
              help="name for solar sourcee group (impacts file naming)")
@click.option("--epwloc/--no-epwloc", default=False,
              help="if True, use location from epw/wea argument to -loc as a"
                   " transit mask (like -loc option 1.) instead of as a list"
                   " of candidate sun positions.")
@click.option("--printdata/--no-printdata", default=False,
              help="if True, print skymapper sun positions (either boundary or "
                   "candidates in xyz coordinates)")
@click.option("-printlevel", type=int,
              help="print a set of sun positions at sampling level "
                   "(overrides printdata)")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def suns(ctx, loc=None, opts=False, debug=False, version=False, epwloc=False,
         printdata=False, printlevel=None, **kwargs):
    """define solar sampling space

    Effects
    ~~~~~~~
        - None
    """
    if hasattr(loc, "shape"):
        if loc.shape[1] == 1:
            loc = loc.ravel()[0:3]
            loc = (loc[0], loc[1], int(loc[2]))
    if epwloc:
        try:
            loc = skycalc.get_loc_epw(loc)
        except ValueError:
            pass
    ctx.obj['skymapper'] = SkyMapper(loc=loc, **kwargs)
    if printdata or printlevel is not None:
        sm = ctx.obj['skymapper']
        if printlevel is not None:
            xyz = sm.solar_grid(level=printlevel)
        elif sm.solarbounds is None:
            xyz = sm.uv2xyz(sm.candidates)
        else:
            xyz = sm.uv2xyz(sm.solarbounds.vertices)
            xyz = xyz[np.logical_not(np.isnan(xyz[:, 0]))]
        click.echo("columns: 0:x 1:y 2:z 3:x(anglular) 4:y(angular) 5:u 6:v", err=True)
        uv = sm.xyz2uv(xyz)
        axy = translate.xyz2xy(xyz)
        for x, u, a in zip(xyz, uv, axy):
            print("{: >10.05f} {: >10.05f} {: >10.05f} {: >10.05f} {: >10.05f} "
                  "{: >10.05f} {: >10.05f}".format(*x, *a, *u))


@main.command()
@click.option("-wea", callback=np_load_safe,
              help="path to epw, wea, .npy file or np.array, or .npz file,"
                   "if loc not set attempts to extract location data "
                   "(if needed).")
@click.option("-loc", default=None, callback=clk.split_float,
              help="location data given as 'lat lon mer' with + west of prime "
                   "meridian overrides location data in wea")
@click.option("-skyro", default=0.0,
              help="angle in degrees counter-clockwise to rotate sky (to "
                   "correct model north, equivalent to clockwise rotation of "
                   "scene)")
@click.option("-ground_fac", default=0.2, help="ground reflectance")
@click.option("-skyres", default=12.0,
              help="approximate square patch size in degrees (must match "
                   "argument given to skyengine)")
@click.option("-minalt", default=2.0,
              help="minimum solar altitude for daylight masking")
@click.option("-mindiff", default=5.0,
              help="minumum diffuse horizontal irradiance for daylight masking")
@click.option("-mindir", default=0.0,
              help="minumum direct normal irradiance for daylight masking")
@click.option("--reload/--no-reload", default=True,
              help="reload saved skydata if it exists in scene directory")
@click.option("-name", default="skydata",
              help="output file name for skydata")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def skydata(ctx, wea=None, name="skydata", loc=None, reload=True, opts=False,
            debug=False, version=None, **kwargs):
    """define sky conditions for evaluation

    Effects
    ~~~~~~~
        - Invokes scene
        - write outdir/name.npz (SkyData initialization object)
    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    file = f"{scn.outdir}/{name}.npz"
    reloaded = reload and os.path.isfile(file)
    if reloaded:
        wea = file
    if wea is None:
        raise ValueError(f"'wea' must be given if {file} does not exist")
    if loc is not None:
        loc = (loc[0], loc[1], int(loc[2]))
    sd = SkyData(wea, loc=loc, **kwargs)
    if not reloaded:
        sd.write(name, scn)
    ctx.obj['skydata'] = sd


@main.command()
@clk.shared_decs(engine_opts)
@click.option("-skyres", default=12.0,
              help="approximate resolution for skypatch subdivision (in "
                   "degrees). Patches will have (rounded) size skyres x skyres."
                   " So if skyres=10, each patch will be 100 sq. degrees "
                   "(0.03046174197 steradians) and there will be 18 * 18 = "
                   "324 sky patches. Must match argument givein to skydata")
@click.option('-fdres', default=9,
              help='the final directional sampling resolution, yielding a'
                   ' grid of potential samples at 2^fdres x 2^fdres per'
                   ' hemisphere')
@click.option('-dcompargs', default='-ab 1',
              help="additional arguments for running direct component. when "
                   "using, set -ab in sunengine.rayargs to this ab minus one.")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def skyengine(ctx, accuracy=1.0, vlt=0.64, idres=5, rayargs=None,
              default_args=True, skyres=10.0, fdres=9, dcompargs='-ab 1',
              usedecomp=False, opts=False, debug=False, version=None, **kwargs):
    """initialize engine for skyrun

    Effects
    ~~~~~~~
        - Invokes scene
        - creates outdir/scene_sky.oct
    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    if usedecomp:
        if rayargs is not None:
            rayargs += " " + dcompargs
        else:
            rayargs = dcompargs
    rcontrib = Rcontrib(rayargs=rayargs, scene=scn.scene, skyres=skyres,
                        default_args=default_args)
    accuracy = accuracy * vlt / 0.64
    ctx.obj['skyengine'] = SkySamplerPt(scn, rcontrib, accuracy=accuracy,
                                        idres=idres, fdres=fdres)


@main.command()
@clk.shared_decs(engine_opts)
@click.option('-fdres', default=10,
              help='the final directional sampling resolution, yielding a'
                   ' grid of potential samples at 2^fdres x 2^fdres per'
                   ' hemisphere')
@click.option('-speclevel', default=9,
              help="at this sampling level, pdf is made from brightness of sky "
                   "sampling rather than progressive variance to look for fine "
                   "scale specular highlights, this should be atleast 1 level "
                   "from the end and the resolution of this level should be "
                   "smaller than the size of the source")
@click.option('-slimit', default=0.01,
              help="the minimum value in the specular guide considered as a "
                   "potential specular reflection source, in the case of low "
                   "vlt glazing, make sure to set -vlt.")
@click.option('-maxspec', default=0.2,
              help="the maximum value in the specular guide considered as a "
                   "specular reflection source. Above this value it is "
                   "assumed that these are direct view rays to the source so "
                   "are not sampled. in the case of low vlt glazing, "
                   "set -vlt. In mixed (high-low) vlt scenes "
                   "the specular guide will either over sample (including "
                   "direct views when maxspec is large) or under sample "
                   "(miss specular reflections when maxspec is small) "
                   "depending on this setting.")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sunengine(ctx, accuracy=1.0, vlt=0.64, idres=5, rayargs=None,
              default_args=True, fdres=10, slimit=0.01, maxspec=0.2,
              opts=False, debug=False, version=None, **kwargs):
    """initialize engine for sunrun

    Effects
    ~~~~~~~
        - Invokes scene
    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    rtrace = Rtrace(rayargs=rayargs, scene=scn.scene, default_args=default_args)
    accuracy = accuracy*vlt/0.64
    maxspec = maxspec*vlt/0.64
    slimit = slimit*vlt/0.64
    ptkwargs = dict(slimit=slimit, maxspec=maxspec, accuracy=accuracy,
                    idres=idres, fdres=fdres)
    ctx.obj['sunengine'] = dict(engine=rtrace, ptkwargs=ptkwargs)


sample_opts = [
 click.option("-accuracy", default=1.0,
              help="parameter to set threshold at sampling level relative to "
                   "final level threshold (smaller number will increase "
                   "sampling)"),
 click.option("-edgemode", default='constant',
              type=click.Choice(['constant', 'reflect', 'nearest', 'mirror',
                                 'wrap']),
              help="if 'constant' value is set to -self.t1, so edge is "
                   "always seen as detail. Internal edges (resulting from "
                   "PlanMapper borders) will behave like 'nearest' for all "
                   "options except 'constant'"),
 click.option("-nlev", default=3,
              help="number of levels to sample (final resolution will be "
                   "ptres/2^(nlev-1))"),
 click.option("--jitter/--no-jitter", default=True,
              help="jitter samples on plane within adaptive sampling grid"),
 click.option("--plotp/--no-plotp", default=False,
              help="plot pdfs and sample vecs for each level")
    ]


@main.command()
@clk.shared_decs(sample_opts)
@click.option("--overwrite/--no-overwrite", default=False,
              help="If True, reruns sampler when invoked, otherwise will first"
                   " attempt to load results")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def skyrun(ctx, accuracy=1.0, nlev=3, jitter=True, overwrite=False, plotp=False,
           edgemode='constant', opts=False, debug=False, version=None):
    """run scene under sky for a set of points (defined by area)

    Effects
    ~~~~~~~
        - Invokes scene
        - Invokes area (no effects)
        - Invokes skyengine
        - creates outdir/area.name/sky_points.tsv
            - contents: 5cols x N rows: [sample_level idx x y z]
        - creates outdir/area.name/sky/######.rytpt
            - each file is a LightPointKD initialization object
    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    if 'planmapper' not in ctx.obj:
        clk.invoke_dependency(ctx, area)
    pm = ctx.obj['planmapper']
    if 'skyengine' not in ctx.obj:
        clk.invoke_dependency(ctx, skyengine)
    if ctx.obj['static_points']:
        nlev = 1
        jitter = False
    skengine = ctx.obj['skyengine']
    skysampler = SamplerArea(scn, skengine, accuracy=accuracy, nlev=nlev,
                             jitter=jitter, edgemode=edgemode)
    try:
        if overwrite:
            raise OSError
        skyfield = LightPlaneKD(scn, f"{scn.outdir}/{pm.name}/sky_points"
                                f".tsv", pm, "sky")
    except OSError:
        skyfield = skysampler.run(pm, plotp=plotp)
    else:
        click.echo(f"Sky Lightfield reloaded from {scn.outdir}/{pm.name} "
                   f"use --overwrite to rerun", err=True)
    ctx.obj['skyfield'] = skyfield


@main.command()
@click.option("--overwrite/--no-overwrite", default=False,
              help="If True, reruns sampler when invoked, otherwise will first"
                   " attempt to load results")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def directskyrun(ctx, overwrite=False, opts=False, debug=False, version=None):
    if 'skyfield' not in ctx.obj:
        clk.invoke_dependency(ctx, skyrun, overwrite=False)
    scn = ctx.obj['scene']
    skyfield = ctx.obj['skyfield']
    skengine = ctx.obj['skyengine']
    pm = skyfield.pm
    try:
        if overwrite:
            raise OSError
        dskyfield = LightPlaneKD(scn,
                                 f"{scn.outdir}/{pm.name}/skydcomp_points"
                                 f".tsv", pm, "skydcomp")
    except OSError:
        skengine.engine.reset()
        clk.invoke_dependency(ctx, skyengine, usedecomp=True)
        skysampler = SamplerArea(scn, skengine)
        dskyfield = skysampler.repeat(skyfield, "skydcomp")
    ctx.obj['dskyfield'] = dskyfield


@main.command()
@clk.shared_decs(sample_opts)
@click.option("-srcaccuracy", default=1.0,
              help="parameter to set threshold at sampling level relative to "
                   "final level threshold (smaller number will increase "
                   "sampling)")
@click.option("-srcnlev", default=3,
              help="number of levels to sample (final resolution will be "
                   "sunres/2^(nlev-1))")
@click.option("--srcjitter/--no-srcjitter", default=True,
              help="jitter solar source within adaptive sampling grid for "
                   "candidate SkyMappers, only affects weighting of selecting"
                   " candidates in the same grid true positions are still "
                   "used")
@click.option("--recover/--no-recover", default=True,
              help="If True, recovers existing sampling")
@click.option("--overwrite/--no-overwrite", default=False,
              help="If True, reruns sampler when invoked, otherwise will first"
                   " attempt to load results")
@click.option("--guided/--no-guided", default=True,
              help="If True, uses skysampling results to guide sun sampling "
                   "this is necessary if the model has any specular "
                   "reflections, will raise an error if skyrun has not been"
                   " called yet.")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sunrun(ctx, srcaccuracy=1.0, srcnlev=3, srcjitter=True, recover=False,
           overwrite=False, guided=True, plotp=False, opts=False, debug=False,
           version=None, **kwargs):
    """run scene for a set of suns (defined by suns) for a set of points
    (defined by area)

    Effects
    ~~~~~~~
        - Invokes scene
        - Invokes area (no effects)
        - Invokes sunengine (no effects)
        - invokes skyrun (if guided=True)
        - creates outdir/area.name/sun_####_points.tsv
            - contents: 5cols x N rows: [sample_level idx x y z]
        - creates outdir/area.name/sky/sun_####/######.rytpt
            - each file is a LightPointKD initialization object
    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    if 'planmapper' not in ctx.obj:
        clk.invoke_dependency(ctx, area)
    pm = ctx.obj['planmapper']
    if 'skymapper' not in ctx.obj:
        clk.invoke_dependency(ctx, suns)
    skmapper = ctx.obj['skymapper']
    if guided and 'skyfield' not in ctx.obj:
        clk.invoke_dependency(ctx, skyrun, overwrite=False)
    if guided:
        specguide = ctx.obj['skyfield']
    else:
        specguide = None
    if 'sunengine' not in ctx.obj:
        clk.invoke_dependency(ctx, sunengine)
    snengine = ctx.obj['sunengine']['engine']
    ptkwargs = ctx.obj['sunengine']['ptkwargs']
    if ctx.obj['static_points']:
        kwargs.update(nlev=1, jitter=False)
    sunsampler = SamplerSuns(scn, snengine, accuracy=srcaccuracy, nlev=srcnlev,
                             jitter=srcjitter, ptkwargs=ptkwargs,
                             areakwargs=kwargs)
    sunfile = f"{scn.outdir}/{pm.name}/{skmapper.name}_sunpositions.tsv"
    if overwrite:
        shutil.rmtree(sunfile, ignore_errors=True)
        for src in glob(f"{scn.outdir}/{pm.name}/{skmapper.name}_sun_*"):
            shutil.rmtree(src, ignore_errors=True)
        for src in glob(f"{scn.outdir}/{pm.name}/i_{skmapper.name}_sun_*"):
            shutil.rmtree(src, ignore_errors=True)
    dfield = sunsampler.run(skmapper, pm, specguide=specguide, recover=recover,
                            plotp=plotp, pfish=False)
    ctx.obj['daylightfield'] = dfield


eval_opts = [
 click.option("-sensors", callback=np_load,
              help="sensor points, this can be a .npy file, a whitespace "
                   "seperated text file or entered as a string with commas "
                   "between components of a point and spaces between points. "
                   "points should either all be 3 componnent (x,y,z) or 6"
                   " component (x,y,z,dx,dy,dz). If 3 component, -sdirs is "
                   "required, if 6-component, -sdirs is ignored"),
 click.option("-sdirs", callback=np_load,
              help="sensor directions, this can be a .npy file, a whitespace "
                   "seperated text file or entered as a string with commas "
                   "between components of a point and spaces between points. "
                   "vectors should all be 3 componnent (dx,dy,dz). used with"
                   "3-component -sensors argument, all points are run for all"
                   "views, creating len(sensors)*len(sdirs) results. this"
                   "is the preferred option for multiple view directions, as"
                   "the calculations are grouped more efficiently"),
 click.option("-viewangle", default=180.),
 click.option("-skymask", callback=clk.split_int,
              help="""mask to reduce output from full SkyData, enter as index
rows in wea/epw file using space seperated list or python range notation:

    - 370 371 372 (10AM-12PM on jan. 16th)
    - 12:8760:24 (everyday at Noon)

"""),
 click.option("-simtype", default="3comp",
              type=click.Choice(api.stypes),
              help="simulation process/integration type"),
 click.option("-resuntol", default=5.0,
              help="tolerance for resampling sun views"),
 click.option("--blursun/--no-blursun", default=False,
              help="for simulating point spread function for direct sun view"),
 click.option("--resampleview/--no-resampleview", default=False,
              help="resample direct sun view directions")
    ]


@main.command()
@clk.shared_decs(eval_opts)
@click.option("-res", default=800, help="image resolution")
@click.option("-interpolate", type=click.Choice(['linear', 'fast', 'high', '', 'None', 'False']))
@click.option("--namebyindex/--no-namebyindex", default=False,
              help="if False (default), names images by: "
                   "<prefix>_sky-<row>_pt-<x>_<y>_<z>_vd-<dx>_<dy>_<dz>.hdr "
                   "if True, names images by: "
                   "<prefix>_sky-<row>_pt-<pidx>_vd-<vidx>.hdr, "
                   "where pidx, vidx refer to the order of points, and vm.")
@click.option("-basename", default="results",
              help="prefix of namebyindex.")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def images(ctx, sensors=None, sdirs=None, viewangle=180., skymask=None,
           basename="results", res=800, interpolate=None, namebyindex=False,
           simtype="2comp", resuntol=5.0, blursun=False, resampleview=False,
           **kwargs):
    """render images

    Prequisites
    ~~~~~~~~~~~
        - skyrun and sunrun must be manually invoked prior to this

    Effects
    ~~~~~~~
        - Invokes scene
        - Invokes skydata
        - invokes area (no effects)
        - invokes suns (no effects)
        - writes: output images according to --namebyindex

    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    if 'skydata' not in ctx.obj:
        clk.invoke_dependency(ctx, skydata, reload=True)
    sd = ctx.obj['skydata']
    if 'planmapper' not in ctx.obj:
        clk.invoke_dependency(ctx, area)
    pm = ctx.obj['planmapper']
    if 'skymapper' not in ctx.obj:
        clk.invoke_dependency(ctx, suns)
    skmapper = ctx.obj['skymapper']
    if resampleview:
        if 'sunengine' not in ctx.obj:
            clk.invoke_dependency(ctx, sunengine)
        sunviewengine = ctx.obj['sunengine']['engine']
    else:
        sunviewengine = None

    itg = api.get_integrator(scn, pm, skmapper.name, simtype,
                             sunviewengine=sunviewengine)

    if skymask is not None:
        sd.mask = skymask
    sensors = np.atleast_2d(sensors)

    if interpolate == "linear":
        interpolate = True
    elif interpolate not in ['fast', 'high']:
        interpolate = False

    if sensors.shape[1] == 6:
        result = []
        for sensor in sensors:
            point = sensor[0:3]
            vm = ViewMapper(sensor[3:6], viewangle)
            result.append(itg.make_images(sd, point, vm, res=res,
                                          interp=interpolate, prefix=basename,
                                          namebyindex=namebyindex,
                                          suntol=resuntol, blursun=blursun))
        result = np.concatenate(result)
    elif sdirs is None:
        raise ValueError("if sensors do not have directions, sdirs cannot be "
                         "None")
    else:
        result = itg.make_images(sd, sensors, sdirs, viewangle=viewangle,
                                 res=res, interp=interpolate, prefix=basename,
                                 namebyindex=namebyindex, suntol=resuntol,
                                 blursun=blursun)
    for d in result:
        print(d)
    sd.mask = None


@main.command()
@clk.shared_decs(eval_opts)
@click.option("-metrics", callback=clk.split_str, default="illum dgp ugp",
              help='metrics to compute, choices: ["illum", '
                   '"avglum", "gcr", "ugp", "dgp", "tasklum", "backlum", '
                   '"dgp_t1", "log_gc", "dgp_t2", "ugr", "threshold", "pwsl2", '
                   '"view_area", "backlum_true", "srcillum", "srcarea", '
                   '"maxlum"]')
@click.option("-basename", default="results",
              help="LightResult object is written to basename.npz.")
@click.option("--npz/--no-npz", default=True,
              help="write LightResult object to .npz, use 'raytraverse pull'"
                   "or LightResult('basename.npz') to access results")
@click.option("--serr/--no-serr", default=False,
              help="include columns of sampling info/errors columns are: "
                   "sun_pt_err, sun_pt_bin, sky_pt_err, sky_pt_bin, sun_err, "
                   "sun_bin. 'err' is distance from queried vector to actual. "
                   "'bin' is the unraveled idx of source vector at a 500^2 "
                   "resolution of the mapper.")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def evaluate(ctx, sensors=None, sdirs=None, viewangle=180., skymask=None,
             metrics=None, basename="results", simtype="2comp", npz=True,
             serr=False, resuntol=5.0, blursun=False, resampleview=False,
             **kwargs):
    """evaluate metrics

    Prequisites
    ~~~~~~~~~~~
        - skyrun and sunrun must be manually invoked prior to this

    Effects
    ~~~~~~~
        - Invokes scene
        - Invokes skydata
        - invokes area (no effects)
        - invokes suns (no effects)
        - writes: <basename>.npz light result file (use "raytraverse pull" to
          extract data views)

    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    if 'skydata' not in ctx.obj:
        clk.invoke_dependency(ctx, skydata, reload=True)
    sd = ctx.obj['skydata']
    if 'planmapper' not in ctx.obj:
        clk.invoke_dependency(ctx, area)
    pm = ctx.obj['planmapper']
    if 'skymapper' not in ctx.obj:
        clk.invoke_dependency(ctx, suns)
    if resampleview:
        if 'sunengine' not in ctx.obj:
            clk.invoke_dependency(ctx, sunengine)
        sunviewengine = ctx.obj['sunengine']['engine']
    else:
        sunviewengine = None
    skmapper = ctx.obj['skymapper']
    zonal = sensors is None
    itg = api.get_integrator(scn, pm, skmapper.name, simtype,
                             sunviewengine=sunviewengine, zonal=zonal)
    if skymask is not None:
        sd.mask = skymask
    sensors = np.atleast_2d(sensors)
    if sensors.shape[1] == 6:
        result = []
        for sensor in sensors:
            point = sensor[0:3]
            vm = ViewMapper(sensor[3:6], viewangle)
            result.append(itg.evaluate(sd, point, vm, metrics=metrics,
                                       datainfo=serr, suntol=resuntol,
                                       blursun=blursun))
        data = np.squeeze(np.concatenate([r.data for r in result], axis=1), 2)
        ptaxis = ResultAxis(sensors, "point")
        result = LightResult(data, result[0].axes[0], ptaxis, result[0].axes[3])
    else:
        if zonal and sdirs is not None:
            sensors = pm
        elif sdirs is None:
            raise ValueError("if sensors do not have directions, sdirs cannot be "
                             "None")
        result = itg.evaluate(sd, sensors, sdirs, viewangle=viewangle,
                              suntol=resuntol, metrics=metrics, datainfo=serr,
                              srconly=simtype == 'directview', blursun=blursun)
    if npz:
        result.write(f"{basename}.npz")
    sd.mask = None
    ctx.obj['lightresult'] = result


@main.command()
@clk.shared_decs(pull_decs)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def pull(*args, **kwargs):
    return shared_pull(*args, **kwargs)


@main.result_callback()
@click.pass_context
def printconfig(ctx, returnvalue, **kwargs):
    """callback to cleanup any temp files"""
    try:
        clk.tmp_clean(ctx)
    except Exception:
        pass


if __name__ == '__main__':
    main()
