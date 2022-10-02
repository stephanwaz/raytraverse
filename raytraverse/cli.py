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
import sys
from glob import glob

import numpy as np

from clasp import click
import clasp.click_ext as clk
from clasp.script_tools import try_mkdir

import raytraverse
from raytraverse import translate
from raytraverse import api
from raytraverse.lightfield import LightPlaneKD, ResultAxis
from raytraverse.lightfield import LightResult
from raytraverse.mapper import PlanMapper, SkyMapper, ViewMapper
from raytraverse.utility.cli import np_load, np_load_safe, shared_pull, pull_decs
from raytraverse.scene import Scene
from raytraverse.renderer import Rtrace, Rcontrib
from raytraverse.sampler import SkySamplerPt, SamplerArea, SamplerSuns, \
    SrcSamplerPt
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

    after adjusting the settings, then each command can be invoked in turn and
    any dependencies will be loaded with the correct options, a complete run
    and evaluation can then be called by::

        raytraverse -c run.cfg skyrun sunrun evaluate

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

    Effects:
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
 click.option('-idres', default=32,
              help='the initial directional sampling resolution '
                   '(as sqrt of samples per hemisphere)')
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
              help="initial sampling resolution for points (in model units")
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

    Effects:
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
@click.option("-sunres", default=9,
              help="initial sampling resolution for suns "
                   "(as sqrt of samples per hemisphere)")
@click.option("-jitterrate", default=0.5,
              help="fraction of each axis to jitter over")
@click.option("-name", default="suns",
              help="name for solar source group (impacts file naming)")
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

    Effects:
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
        click.echo("columns: 0:x 1:y 2:z 3:x(anglular) 4:y(angular) 5:u 6:v",
                   err=True)
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
@click.option("-skyres", default=15,
              help="resolution of sky patches (sqrt(patches / hemisphere)).")
@click.option("-minalt", default=2.0,
              help="minimum solar altitude for daylight masking")
@click.option("-mindiff", default=5.0,
              help="minumum diffuse horizontal irradiance for daylight masking")
@click.option("-mindir", default=0.0,
              help="minumum direct normal irradiance for daylight masking")
@click.option("--reload/--no-reload", default=True,
              help="reload saved skydata if it exists in scene directory")
@click.option("--printdata/--no-printdata", default=False,
              help="if True, print solar position and dirnorm/diff of loaded "
                   "data")
@click.option("--printfull/--no-printfull", default=False,
              help="with printdata, if True, print full unmasked skydata")
@click.option("-name", default="skydata",
              help="output file name for skydata")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def skydata(ctx, wea=None, name="skydata", loc=None, reload=True, opts=False,
            printdata=False, printfull=False, debug=False, version=None, **kwargs):
    """define sky conditions for evaluation

    Effects:
        - Invokes scene
        - write outdir/name.npz (SkyData initialization object)
    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    if scn.outdir is None:
        print("Warning! attempting to use skydata from outside scene, "
              "make sure scene -outdir is set", file=sys.stderr)
        file = f"{name}.npz"
    else:
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
        if scn.outdir is None:
            sd.write(name)
        else:
            sd.write(name, scn)
    ctx.obj['skydata'] = sd
    if printdata:
        ris = np.arange(sd.skydata.shape[0])
        if printfull:
            rows = sd.fill_data(sd.skydata[sd.daymask], rowlabels=True)
        else:
            rows = sd.label(sd.skydata[sd.daymask])
            ris = ris[sd.daymask]
        for j, (ri, r) in enumerate(zip(ris, rows)):
            ot = " ".join([f"{i: >10.05g}" for i in r])
            print(f"{ri: >5} {j: >5} {ot}")


@main.command()
@clk.shared_decs(engine_opts)
@click.option('-rayargs',
              help='additional arguments to pass to the rendering engine')
@click.option('--default-args/--no-default-args', default=True,
              help='use raytraverse defaults before -rayargs, if False, uses'
                   ' radiance defaults. defaults are: -u+ -ab 16 -av 0 0 0 '
                   '-aa 0 -as 0 -dc 1 -dt 0 -lr -14 -ad adpatch*(skyres^2+1) '
                   '-lw 0.008/(skyres^2+1)/adpatch -st 0 -ss 16 -c 1. note that'
                   ' if this is false -ad and -lw will not be automatically '
                   'set')
@click.option("-skyres", default=15,
              help="resolution of sky patches (sqrt(patches / hemisphere))."
                   "Must match argument givein to skydata")
@click.option("-adpatch", default=50,
              help="prefered instead of -ad/-lw in rayargs to better coordinate"
                   " settings of ad/lw and skypatch division, consider doubling"
                   " this with each halving of accuracy and in cases with high"
                   " proportion indirect contributions, such as deep spaces or"
                   " complex fenestrations")
@click.option('-nlev', default=5,
              help='number of directional sampling levels, yielding a final'
                   'resolution of idres^2 * 2^(nlev) samples per hemisphere')
@click.option('-dcompargs', default='-ab 1',
              help="additional arguments for running direct component. when "
                   "using, set -ab in sunengine.rayargs to this ab minus one.")
@click.option("-vlt", default=0.64,
              help="primary transmitting vlt, used to scale the accuracy "
                   "parameter to the expected scene variance. Optional, but "
                   "helpful with, for example, electrochromic glazing or "
                   "shades")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def skyengine(ctx, accuracy=1.0, vlt=0.64, idres=32, rayargs=None,
              default_args=True, skyres=15, nlev=5, dcompargs='-ab 1',
              usedecomp=False, opts=False, debug=False, version=None, **kwargs):
    """initialize engine for skyrun

    Effects:
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
                                        idres=idres, nlev=nlev)


@main.command()
@clk.shared_decs(engine_opts)
@click.option('-rayargs', default="-ab 0",
              help='additional arguments to pass to the rendering engine,'
                   ' by default sets -ab 0, pass "" to clear')
@click.option('--default-args/--no-default-args', default=True,
              help='use raytraverse defaults before -rayargs, if False, uses'
                   ' radiance defaults. defaults are: -u+ -ab 16 -av 0 0 0 '
                   '-aa 0 -as 0 -dc 1 -dt 0 -lr -14 -ad 1000 -lw 0.00004 -st 0 '
                   '-ss 16 -w-')
@click.option('-nlev', default=6,
              help='number of directional sampling levels, yielding a final'
                   'resolution of idres^2 * 2^(nlev) samples per hemisphere')
@click.option("-vlt", default=0.64,
              help="primary transmitting vlt, used to scale the accuracy "
                   "parameter to the expected scene variance. Optional, but "
                   "helpful with, for example, electrochromic glazing or "
                   "shades")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sunengine(ctx, accuracy=1.0, vlt=0.64, idres=32, rayargs=None,
              default_args=True, nlev=6, opts=False,
              debug=False, version=None, **kwargs):
    """initialize engine for sunrun

    Effects:
        - Invokes scene
    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    if 'sourceengine' in ctx.obj:
        ctx.obj['sourceengine']['engine'].reset()
        ctx.obj.pop('sourceengine')
    scn = ctx.obj['scene']
    rtrace = Rtrace(rayargs=rayargs, scene=scn.scene, default_args=default_args)
    accuracy = accuracy*vlt/0.64
    ptkwargs = dict(accuracy=accuracy, idres=idres, nlev=nlev)
    ctx.obj['sunengine'] = dict(engine=rtrace, ptkwargs=ptkwargs)


@main.command()
@click.option('-idres', default=32,
             help='the initial directional sampling resolution '
                  '(as sqrt of samples per hemisphere)')
@click.option('-t0', default=20.0,
              help='initial sample  threshold (in cd/m^2), use instead of '
                   'accuracy when source luminance and meaningful difference'
                   'is known')
@click.option('-t1', default=400.0,
              help='final sample  threshold (in cd/m^2), use instead of '
                   'accuracy when source luminance and meaningful difference'
                   'is known')
@click.option('-rayargs',
              help='additional arguments to pass to the rendering engine')
@click.option('--default-args/--no-default-args', default=True,
              help='use raytraverse defaults before -rayargs, if False, uses'
                   ' radiance defaults. defaults are: -u+ -ab 16 -av 0 0 0 '
                   '-aa 0 -as 0 -dc 1 -dt 0 -lr -14 -ad 1000 -lw 0.00004 -st 0 '
                   '-ss 16 -w-')
@click.option("-srcfile", default=None, type=click.Path(dir_okay=False),
              help="scene source description (required)")
@click.option("-source", default="source", help="name for this source")
@click.option('-nlev', default=6,
              help='number of directional sampling levels, yielding a final'
                   'resolution of idres^2 * 2^(nlev) samples per hemisphere')
@click.option("-vlt", default=1.0,
              help="Leave at 1.0 for interior light sources. primary "
                   "transmitting vlt, used to scale the accuracy "
                   "parameter to the expected scene variance. Optional, but "
                   "helpful with, for example, electrochromic glazing or "
                   "shades")
@click.option("--color/--no-color", default=True)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sourceengine(ctx, srcfile=None, source="source", idres=32, rayargs=None,
                 default_args=True, nlev=6, color=True, t0=20.0, t1=400.0,
                 opts=False, debug=False, version=None,
                 **kwargs):
    """initialize engine for sunrun

    Effects:
        - Invokes scene
    """
    if srcfile is None:
        click.echo("srcfile is required by sourcerun", err=True)
        raise click.Abort
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    if 'sunengine' in ctx.obj:
        ctx.obj['sunengine']['engine'].reset()
        ctx.obj.pop('sunengine')
    scn = ctx.obj['scene']
    srcscn = scn.source_scene(srcfile, source)
    if rayargs is None:
        rayargs = ""
    rayargs += f" -af {scn.outdir}/{source}.amb"
    rtrace = Rtrace(rayargs=rayargs, scene=srcscn, default_args=default_args)
    if color:
        rtrace.update_ospec("v")
    ptkwargs = dict(t0=t0, t1=t1, idres=idres, nlev=nlev, stype=source)
    ctx.obj['sourceengine'] = dict(engine=rtrace, ptkwargs=ptkwargs,
                                   source=(srcfile, source))


sample_opts = [
 click.option("-accuracy", default=1.0,
              help="parameter to set threshold at sampling level relative to "
                   "final level threshold (smaller number will increase "
                   "sampling)"),
 click.option("-edgemode", default='reflect',
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

    Effects:
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
        skyfield = skysampler.run(pm, plotp=plotp, pfish=False)
        try_mkdir(f"{scn.outdir}/{pm.name}")
        reflf = f"{scn.outdir}/{pm.name}/reflection_normals.txt"
        if not os.path.isfile(reflf):
            refl = scn.reflection_search(skyfield.vecs)
            if refl.size > 0:
                np.savetxt(reflf, refl)
    else:
        if scn.dolog:
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
                   "sunres*2^(nlev-1))")
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
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sunrun(ctx, srcaccuracy=1.0, srcnlev=3, srcjitter=True, recover=False,
           overwrite=False, plotp=False, opts=False, debug=False,
           version=None, **kwargs):
    """run scene for a set of suns (defined by suns) for a set of points
    (defined by area)

    Effects:
        - Invokes scene
        - Invokes area (no effects)
        - Invokes sunengine (no effects)
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
    try_mkdir(f"{scn.outdir}/{pm.name}")
    reflf = f"{scn.outdir}/{pm.name}/reflection_normals.txt"
    if not os.path.isfile(reflf):
        refl = scn.reflection_search(pm.point_grid(False))
        if refl.size > 0:
            np.savetxt(reflf, refl)
    dfield = sunsampler.run(skmapper, pm, specguide=True, recover=recover,
                            plotp=plotp, pfish=False)
    ctx.obj['daylightfield'] = dfield


@main.command()
@clk.shared_decs(sample_opts)
@click.option("--overwrite/--no-overwrite", default=False,
              help="If True, reruns sampler when invoked, otherwise will first"
                   " attempt to load results")
@click.option("--scenedetail/--no-scenedetail", default=False,
              help="If True, includes scene details (distance, surface normal,"
                   "and modifier as features). Increases sampling rate to"
                   " improve image reconstruction")
@click.option("-distance", default=0.5,
              help="when using scene detail, the difference in ray length "
                   "equivalent to final sampling luminance threshold")
@click.option("-normal", default=5.0,
              help="when using scene detail, the difference in surface normal "
                   "(degrees) equivalent to final sampling luminance threshold")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sourcerun(ctx, accuracy=1.0, nlev=3, jitter=True, overwrite=False,
              plotp=False, edgemode='constant', scenedetail=False, distance=0.5,
              normal=5.0, opts=False, debug=False, version=None):
    """run scene for a single source (or multiple defined in a single scene file)

    - Do not run as part of the same call as sunrun
    - make sure rayargs are properly set in sunengine (not -ab 0)

    Effects:
        - Invokes scene
        - Invokes area (no effects)
        - Invokes sunengine (no effects)
        - creates outdir/area.name/SOURCE_points.tsv
            - contents: 5cols x N rows: [sample_level idx x y z]
        - creates outdir/area.name/sky/SOURCE/######.rytpt
            - each file is a LightPointKD initialization object
    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    if 'planmapper' not in ctx.obj:
        clk.invoke_dependency(ctx, area)
    pm = ctx.obj['planmapper']
    if 'sourceengine' not in ctx.obj:
        clk.invoke_dependency(ctx, sourceengine)
    srcrtrace = ctx.obj['sourceengine']['engine']
    ptkwargs = ctx.obj['sourceengine']['ptkwargs']
    srcfile, source = ctx.obj['sourceengine']['source']
    if ctx.obj['static_points']:
        nlev = 1
        jitter = False
    srcengine = SrcSamplerPt(scn, srcrtrace, srcfile, scenedetail=scenedetail,
                             distance=distance, normal=normal, **ptkwargs)
    srcsampler = SamplerArea(scn, srcengine, accuracy=accuracy, nlev=nlev,
                             jitter=jitter, edgemode=edgemode, metricset=('avglum', 'loggcr'))
    try:
        if overwrite:
            raise OSError
        skyfield = LightPlaneKD(scn, f"{scn.outdir}/{pm.name}/{source}_points"
                                f".tsv", pm, source)
    except OSError:
        try_mkdir(f"{scn.outdir}/{pm.name}")
        reflf = f"{scn.outdir}/{pm.name}/reflection_normals.txt"
        if not os.path.isfile(reflf) and len(srcengine.sources) > 0:
            refl = scn.reflection_search(pm.point_grid(False))
            if refl.size > 0:
                np.savetxt(reflf, refl)
        skyfield = srcsampler.run(pm, plotp=plotp, pfish=False,
                                  specguide=True)
    else:
        if scn.dolog:
            click.echo(f"Source Lightfield reloaded from {scn.outdir}/{pm.name} "
                       f"use --overwrite to rerun", err=True)
    ctx.obj[f'{source}_field'] = skyfield



eval_opts = [
 click.option("-sensors", callback=np_load,
              help="sensor points, this can be a .npy file, a whitespace "
                   "seperated text file or entered as a string with commas "
                   "between components of a point and spaces between points. "
                   "points should either all be 3 componnent (x,y,z) or 6"
                   " component (x,y,z,dx,dy,dz). If 3 component, -sdirs is "
                   "required, if 6-component, -sdirs is ignored. leave as None"
                   " for zonal evaluation (sdirs required)"),
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
 click.option("--maskfull/--maskday", default=True,
              help="if false, skymask assumes daystep indices"),
 click.option("-simtype", default="3comp",
              help=f"simulation process/integration type:\n\n"
                   f"{api.stypedocstring}\n\nor source name (overrides "
                   f"--resampleview, --directview, etc."),
 click.option("-resuntol", default=1.0,
              help="tolerance for resampling sun views"),
 click.option("-resamprad", default=0.0,
              help="radius for resampling sun vecs"),
 click.option("--blursun/--no-blursun", default=False,
              help="for simulating point spread function for direct sun view"),
 click.option("--resampleview/--no-resampleview", default=False,
              help="resample direct sun view directions")
    ]


@main.command()
@clk.shared_decs(eval_opts)
@click.option("-res", default=800, help="image resolution")
@click.option("-interpolate", type=click.Choice(['linear', 'fast', 'high',
                                                 'fastc', 'highc', '', 'None',
                                                 'False']))
@click.option("--namebyindex/--no-namebyindex", default=False,
              help="if False (default), names images by: "
                   "<prefix>_sky-<row>_pt-<x>_<y>_<z>_vd-<dx>_<dy>_<dz>.hdr "
                   "if True, names images by: "
                   "<prefix>_sky-<row>_pt-<pidx>_vd-<vidx>.hdr, "
                   "where pidx, vidx refer to the order of points, and vm.")
@click.option("-basename", default="results",
              help="prefix of namebyindex.")
@click.option("-bandwidth", default=20,
              help="used by interpolation.")
@click.option("--directview/--no-directview", default=False,
              help="if True, ignore sky data and use daylight factors directly")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def images(ctx, sensors=None, sdirs=None, viewangle=180., skymask=None,
           basename="results", res=800, interpolate=None, namebyindex=False,
           simtype="2comp", resuntol=5.0, blursun=False, resampleview=False,
           directview=False, maskfull=True, resamprad=0.0, bandwidth=20,
           **kwargs):
    """render images

    Prerequisites:

        - skyrun and sunrun must be manually invoked prior to this

    Effects:

        - Invokes scene
        - Invokes skydata
        - invokes area (no effects)
        - invokes suns (no effects)
        - writes: output images according to --namebyindex

    """
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene, reload=True, overwrite=False)
    scn = ctx.obj['scene']
    if 'planmapper' not in ctx.obj:
        clk.invoke_dependency(ctx, area)
    pm = ctx.obj['planmapper']
    if simtype not in api.stypes:
        sd = SkyData(None, skyres=1, ground=False, srcname=simtype)
        resampleview = False
        skymask = None
        srcname = simtype
        if interpolate in ['fastc', 'highc']:
            if 'sourceengine' not in ctx.obj:
                clk.invoke_dependency(ctx, sourceengine)
            sunviewengine = ctx.obj['sourceengine']['engine']
        else:
            sunviewengine = None
    else:
        if 'skydata' not in ctx.obj:
            clk.invoke_dependency(ctx, skydata, reload=True)
        sd = ctx.obj['skydata']
        if 'skymapper' not in ctx.obj:
            clk.invoke_dependency(ctx, suns)
        skmapper = ctx.obj['skymapper']
        if resampleview or interpolate in ['fastc', 'highc']:
            if 'sunengine' not in ctx.obj:
                clk.invoke_dependency(ctx, sunengine)
            sunviewengine = ctx.obj['sunengine']['engine']
        else:
            sunviewengine = None
        if directview:
            sd = SkyData(None, skyres=sd.skyres)
        srcname = skmapper.name
    if not resampleview:
        resuntol = 180
    itg = api.get_integrator(scn, pm, srcname, simtype,
                             sunviewengine=sunviewengine)
    if skymask is not None:
        if maskfull:
            sd.mask = skymask
        else:
            sd.mask = sd.maskindices[skymask]
    sensors = np.atleast_2d(sensors)

    if interpolate == "linear":
        interpolate = True
    elif interpolate not in ['fastc', 'highc', 'fast', 'high']:
        interpolate = False

    if sensors.shape[1] == 6:
        result = []
        for sensor in sensors:
            point = sensor[0:3]
            vm = ViewMapper(sensor[3:6], viewangle)
            result.append(itg.make_images(sd, point, vm, res=res,
                                          interp=interpolate, prefix=basename,
                                          namebyindex=namebyindex,
                                          suntol=resuntol, blursun=blursun,
                                          resamprad=resamprad,
                                          bandwidth=bandwidth))
        result = np.concatenate(result)
    elif sdirs is None:
        raise ValueError("if sensors do not have directions, sdirs cannot be "
                         "None")
    else:
        result = itg.make_images(sd, sensors, sdirs, viewangle=viewangle,
                                 res=res, interp=interpolate, prefix=basename,
                                 namebyindex=namebyindex, suntol=resuntol,
                                 blursun=blursun,  resamprad=resamprad,
                                 bandwidth=bandwidth)
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
@click.option("-threshold", default=2000.,
              help="same as the evalglare -b option. if factor is larger than "
                   "100, it is used as constant threshold in cd/m2, else this "
                   "factor is multiplied by the average task luminance. task "
                   "position is center of image with a 30 degree field of view")
@click.option("--npz/--no-npz", default=True,
              help="write LightResult object to .npz, use 'raytraverse pull'"
                   "or LightResult('basename.npz') to access results")
@click.option("--lowlight/--no-lowlight", default=False,
              help="use lowlight correction for dgp")
@click.option("--coercesumsafe/--no-coercesumsafe", default=False,
              help="to speed up evaluation, treat sources seperately,"
                   "only compatible with illum, avglum, ugp (but note this is "
                   "often WRONG!!!), dgp")
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
             lowlight=False, coercesumsafe=False, threshold=2000.,
             maskfull=True, resamprad=0.0, **kwargs):
    """evaluate metrics

    Prequisites

        - skyrun and sunrun must be manually invoked prior to this

    Effects:

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
                             sunviewengine=sunviewengine)
    if skymask is not None:
        if maskfull:
            sd.mask = skymask
        else:
            sd.mask = sd.maskindices[skymask]
    sensors = np.atleast_2d(sensors)
    if sensors.shape[1] == 6:
        result = []
        for sensor in sensors:
            point = sensor[0:3]
            vm = ViewMapper(sensor[3:6], viewangle)
            result.append(itg.evaluate(sd, point, vm, metrics=metrics,
                                       datainfo=serr, suntol=resuntol,
                                       blursun=blursun, lowlight=lowlight,
                                       threshold=threshold, resamprad=resamprad,
                                       coercesumsafe=coercesumsafe))
        data = np.squeeze(np.concatenate([r.data for r in result], axis=1), 2)
        ptaxis = ResultAxis(sensors, "point")
        result = LightResult(data, result[0].axes[0], ptaxis, result[0].axes[3])
    else:
        if zonal and sdirs is not None:
            sensors = None
        elif sdirs is None:
            raise ValueError("if sensors do not have directions, sdirs cannot be "
                             "None")
        result = itg.evaluate(sd, sensors, sdirs, viewangle=viewangle,
                              suntol=resuntol, metrics=metrics, datainfo=serr,
                              srconly=simtype == 'directview', blursun=blursun,
                              lowlight=lowlight, threshold=threshold,
                              coercesumsafe=coercesumsafe, resamprad=resamprad)
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
