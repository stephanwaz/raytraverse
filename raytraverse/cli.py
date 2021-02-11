# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""Console script for raytraverse."""
import os
import re
import sys

import numpy as np

from clasp import click
import clasp.click_ext as clk
import clasp.script_tools as cst

import raytraverse
from raytraverse.scene import Scene


@clk.pretty_name("NPY, TSV, FLOATS,FLOATS")
def np_load(ctx, param, s):
    """read np array from command line

    trys np.load (numpy binary), then np.loadtxt (space seperated txt file)
    then split row by spaces and columns by commas.
    """
    if s is None:
        return s
    if os.path.exists(s):
        try:
            ar = np.load(s)
        except ValueError:
            ar = np.loadtxt(s)
        if len(ar.shape) == 1:
            ar = ar.reshape(1, -1)
        return ar
    else:
        return np.array([[float(i) for i in j.split(',')] for j in s.split()])


@click.group(chain=True, invoke_without_command=True)
@click.argument('out')
@click.option('--config', '-c', type=click.Path(exists=True),
              help="path of config file to load")
@click.option('--template/--no-template', is_eager=True,
              callback=clk.printconfigs,
              help="write default options to std out as config")
@click.option('-n', default=None, type=int,
              help='sets the environment variable RAYTRAVERSE_PROC_CAP set to'
                   '0 to clear (parallel processes will use cpu_limit)')
@click.option('--opts', '-opts', is_flag=True,
              help="check parsed options")
@click.option('--debug', is_flag=True,
              help="show traceback on exceptions")
@click.version_option(version=raytraverse.__version__)
@click.pass_context
def main(ctx, out, config, n=None,  **kwargs):
    """the raytraverse executable is a command line interface to the raytraverse
    python package for running and evaluating climate based daylight models.
    sub commands of raytraverse can be chained but should be invoked in the
    order given.

    the easiest way to manage options and sure that Scene and SunSetter classes
    are properly reloaded is to use a configuration file, to make a template::

        raytraverse --template > run.cfg

    after adjusting the settings, than each command can be invoked in turn and
    any dependencies will be loaded with the correct options, a complete run
    and evaluation can then be called by::

        raytraverse -c run.cfg OUT sky sunrun integrate

    as both scene and sun will be invoked automatically as needed.

    Arguments:
        * ctx: click.Context
        * out: path to new or existing directory for raytraverse run
        * config: path to config file
        * n: max number of processes to spawn
    """
    raytraverse.io.set_nproc(n)
    ctx.info_name = 'raytraverse'
    clk.get_config_chained(ctx, config, None, None, None)
    ctx.obj = dict(out=out)


@main.command()
@click.option('-skyro', default=0.0,
              help='counter clockwise rotation (in degrees) of the sky to'
                   ' rotate true North to project North, so if project North'
                   ' is 10 degrees East of North, skyro=10')
@click.option('-sunres', default=10.0,
              help='resolution in degrees of the sky patch grid in which to'
                   ' stratify sun samples. Suns are randomly located within'
                   ' the grid, so this corresponds to the average distance'
                   ' between sources. The average error to a randomly selected'
                   ' sun position will be on average ~0.4 times this value')
@click.option('-loc', callback=clk.split_float,
              help='specify the scene location (if not specified in -wea or to'
                   ' override. give as "lat lon mer" where lat is + North, lon'
                   ' is + West and mer is the timezone meridian (full hours are'
                   ' 15 degree increments)')
@click.option('-wea',
              help="path to weather/sun position file. possible formats are:\n"
                   "\n"
                   "1. .wea file\n"
                   "#. .wea file without header (require -loc and "
                   "--no-usepositions)\n"
                   "#. .epw file\n"
                   "#. .epw file without header (require -loc and "
                   "--no-usepositions)\n"
                   "#. 3 column tsv file, each row is dx, dy, dz of candidate"
                   " sun position (requires --usepositions)\n"
                   "#. 4 column tsv file, each row is altitude, azimuth, direct"
                   " normal, diff. horizontal of canditate suns (requires "
                   "--usepositions)\n"
                   "#. 5 column tsv file, each row is dx, dy, dz, direct "
                   "normal, diff. horizontal of canditate suns (requires "
                   "--usepositions)\n\n"
                   "tsv files are loaded with loadtxt")
@click.option('--plotdview/--no-plotdview', default=False,
              help="creates a png showing sun positions on an angular fisheye"
                   " projection of the sky. sky patches are colored by the"
                   " maximum contributing ray to the scene")
@click.option('--reload/--no-reload', default=True,
              help="if False, regenerates sun positions, because positions may"
                   " be randomly selected this will make any sunrun results"
                   " obsolete")
@click.option('--printsuns/--no-printsuns', default=False,
              help="print sun positions to stdout")
@click.option('--usepositions/--no-usepositions', default=False,
              help='if True, sun positions will be chosen from the positions'
                   ' listed in wea. if more than one position is a candidate'
                   ' for that particular sky patch (as determined by sunres)'
                   ' than a random choice will be made. by using one of the'
                   ' tsv format options for wea, and preselecting sun positions'
                   ' such that there is 1 per patch a deterministic result can'
                   'be achieved.')
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def suns(ctx, loc=None, wea=None, usepositions=False, plotdview=False,
         printsuns=False, **kwargs):
    """the suns command provides a number of options for creating sun positions
    used by sunrun see wea and usepositions options for details

    Note:

    the wea and skyro parameters are used to reduce the number of suns in cases
    where a specific site is known. Only suns within the solar transit (or
    positions if usepositions is True will be selected. It is important to note
    that when integrating, if a sun position outside this range is queried than
    results will not include the more detailed simulations involved in sunrun
    and will instead place the suns energy within the nearest sky patch. if
    skyres is small and or the patch is directly visible this will introduce
    significant bias in most metrics.
    """
    pass
    # if usepositions:
    #     if wea is None:
    #         raise ValueError('option -wea is required when use positions is '
    #                          'True')
    #     s = SunsPos(ctx.obj['out'], wea, **kwargs)
    # elif loc is not None:
    #     s = SunsLoc(ctx.obj['out'], loc, **kwargs)
    # elif wea is not None:
    #     loc = raytraverse.skycalc.get_loc_epw(wea)
    #     s = SunsLoc(ctx.obj['out'], loc, **kwargs)
    # else:
    #     s = Suns(ctx.obj['out'], **kwargs)
    # if plotdview:
    #     s.direct_view()
    # ctx.obj['suns'] = s
    # if printsuns:
    #     for pt in s.suns:
    #         print('{}\t{}\t{}'.format(*pt))


@main.command()
@click.option('-scene', help='space separated list of radiance scene files '
              '(no sky) or precompiled octree')
@click.option('--reload/--no-reload', default=True,
              help='if a scene already exists at OUT reload it, note that if'
                   'this is False and overwrite is False, the program will'
                   'abort')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Warning! if set to True and reload is False all files in'
                   'OUT will be deleted')
@click.option('--frozen/--no-frozen', default=True,
              help='create frozen octree from scene files')
@click.option('--log/--no-log', default=True,
              help='log progress to <out>/log.txt')
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def scene(ctx, opts=False, debug=False, **kwargs):
    """The scene commands creates a Scene object which holds geometric
    information about the model including object geometry (and defined
    materials), the analysis plane and the desired resolutions for sky and
    analysis plane subdivision"""
    print(kwargs)
    s = Scene(ctx.obj['out'], **kwargs)
    ctx.obj['scene'] = s
    ctx.obj['initlf'] = []
#
#
# run_opts = [
#  click.option('-accuracy', default=1.0,
#               help='a generic accuracy parameter that sets the threshold'
#                    ' variance to sample. A value of 1 will have a sample count'
#                    ' at the final sampling level equal to the number of'
#                    ' directions with a contribution variance greater than .25'),
#  click.option('-idres', default=4,
#               help='the initial directional sampling resolution. each side'
#                    ' of the sampling square (representing a hemisphere) will'
#                    ' be subdivided 2^idres, yielding 2^(2*idres) samples and'
#                    ' a resolution of 2^(2*idres)/(2pi) samples/steradian. this'
#                    ' value should be smaller than 1/2 the size of the smallest'
#                    ' view to an aperture that should be captured with 100%'
#                    ' certainty'),
#  click.option('--plotp/--no-plotp', default=False,
#               help='for diagnostics only, plots the pdf at each level for'
#                    ' point[0,0] in an interactive display (note that program'
#                    ' will hang until the user closes the plot window at each'
#                    ' level)'),
#  click.option('--plotdview/--no-plotdview', default=False,
#               help='plot a direct view of the sky field (as a .hdr file),'
#                    ' this is equivalent to integrating with a value of 1 for'
#                    ' all sky patches with no interpolation, plots pixels of'
#                    ' actualsample vectors in red'),
#  click.option('--run/--no-run', default=True,
#               help='if True calls sampler.run()'),
#  click.option('--rmraw/--no-rmraw', default=True,
#               help='if True removes output of sampler.run(), after SCBinField'
#                    ' is constructed. Note that SCBinField cannot be rebuilt'
#                    ' once raw files are removed'),
#  click.option('--overwrite/--no-overwrite', default=False,
#               help='execute run even if simulations exist'),
#  click.option('--rebuild/--no-rebuild', default=False,
#               help='force rebuild kdtree'),
#  click.option('--showsample/--no-showsample', default=True,
#                  help='show samples on dviews'),
#  click.option('--showweight/--no-showweight', default=True,
#                  help='show weights on dviews'),
#  click.option('-dpts', default=None, callback=np_load,
#               help="if given points to evaluate with --plotdview, this can be "
#                    "a .npy file, a whitespace "
#                    "seperated text file or entered as a string with commas "
#                    "between components of a point and spaces between points. "
#                    "in all cases each point requires 6 numbers x,y,z,dx,dy,dz "
#                    "so the shape of the array will be (N, 6)")
#     ]
#
#
# @main.command()
# @click.option('-fdres', default=9,
#               help='the final directional sampling resolution, yielding a'
#                    ' grid of potential samples at 2^fdres x 2^fdres per'
#                    ' hemisphere')
# @click.option('-rcopts',
#               default='-ab 7 -ad 60000 -as 30000 -lw 1e-7 -st 0 -ss 16',
#               help='rtrace options to pass to the rcontrib call'
#                    ' see the man pages for rtrace, rcontrib, and '
#                    ' rcontrib -defaults for more information')
# @click.option('-dviewpatch',
#               callback=clk.split_float,
#               help="to plot direct view for a single patch,"
#                    "give an x,y,z direction")
# @clk.shared_decs(run_opts)
# @clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
# def sky(ctx, plotdview=False, run=True, rmraw=True, overwrite=False,
#         rebuild=False, showsample=True, showweight=True, dpts=None,
#         dviewpatch=None, **kwargs):
#     """the sky command intitializes and runs a sky sampler and then readies
#     the results for integration by building a SCBinField. sky should be invoked
#     before calling suns, as the sky contributions are used to select the
#     necessary sun positions to run"""
#     if 'scene' not in ctx.obj:
#         clk.invoke_dependency(ctx, scene)
#     s = ctx.obj['scene']
#     lumf = f'{s.outdir}/sky_kd_lum.dat'
#     exists = os.path.isfile(lumf) and os.stat(lumf).st_size > 1000
#     run = run and (overwrite or not exists)
#     if run:
#         sampler = SkySampler(s, **kwargs)
#         sampler.run()
#         try:
#             os.remove(f"{s.outdir}/sunpdfidxs.npz")
#         except IOError:
#             pass
#     sk = SCBinField(s, rebuild=run or rebuild, rmraw=rmraw)
#     ctx.obj['initlf'].append(sk)
#     if plotdview:
#         sk.direct_view(showsample=showsample, showweight=showweight, dpts=dpts,
#                        srcidx=dviewpatch, res=1024)
#
# @main.command()
# @click.option('-fdres', default=10,
#               help='the final directional sampling resolution, yielding a'
#                    ' grid of potential samples at 2^fdres x 2^fdres per'
#                    ' hemisphere')
# @click.option('-speclevel', default=9,
#               help='at this sampling level, pdf is made from brightness of sky '
#                    'sampling rather than progressive variance to look for fine '
#                    'scale specular highlights, this should be atleast 1 level '
#                    'from the end and the resolution of this level should be '
#                    'smaller than the size of the source')
# @click.option('-rcopts',
#               default='-ab 6 -ad 3000 -as 1500 -st 0 -ss 16 -aa .1',
#               help='rtrace options for sun reflection runs'
#                    ' see the man pages for rtrace, and '
#                    ' rtrace -defaults for more information')
# @click.option('--view/--no-view', default=True,
#               help="run/build/plot direct sun views")
# @click.option('--sunviewstats/--no-sunviewstats', default=False,
#               help="run/build/plot direct sun views")
# @click.option('--ambcache/--no-ambcache', default=True,
#               help='whether the rcopts indicate that the calculation will use '
#                    'ambient caching (and thus should write an -af file argument'
#                    ' to the engine)')
# @click.option('--keepamb/--no-keepamb', default=False,
#               help='whether to keep ambient files after run, if kept, a '
#                    'successive call will load these ambient files, so care '
#                    'must be taken to not change any parameters')
# @click.option('--reflection/--no-reflection', default=True,
#               help="run/build/plot reflected sun components")
# @click.option('--checkviz/--no-checkviz', default=True,
#               help="precheck for visable sun based on sky run for sunview "
#                    "sampler")
# @clk.shared_decs(run_opts)
# @clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
# def sunrun(ctx, plotdview=False, run=True, rmraw=False, overwrite=False,
#            rebuild=False, showsample=True, showweight=True, dpts=None,
#            sunviewstats=False, checkviz=True, **kwargs):
#     """the sunrun command intitializes and runs a sun sampler and then readies
#     the results for integration by building a SunField."""
#     if 'suns' not in ctx.obj:
#         clk.invoke_dependency(ctx, suns)
#     scn = ctx.obj['scene']
#     sns = ctx.obj['suns']
#     if kwargs['view']:
#         sampler = SunViewSampler(scn, suns, checkviz=checkviz)
#         lumf = f'{scn.outdir}/sunview_kd_data.pickle'
#         exists = os.path.isfile(lumf) and os.stat(lumf).st_size > 1000
#         vrun = run and (overwrite or not exists)
#         if vrun:
#             sampler.run()
#         try:
#             sv = SunViewField(scn, sns, rebuild=vrun or rebuild, rmraw=rmraw)
#         except FileNotFoundError as ex:
#             print(f'Warning: {ex}', file=sys.stderr)
#         else:
#             ctx.obj['initlf'].append(sv)
#             if sunviewstats:
#                 vs = sv.point_stats()
#                 print(f"# Visibility of {sns.suns.shape[0]} suns from "
#                       f"{scn.area.npts} points", file=sys.stderr)
#                 print("x y z n-suns min-tvis max-tvis min-vis max-vis",
#                       file=sys.stderr)
#                 for v in vs:
#                     print(" ".join([f"{i:.03f}" for i in v]))
#             if plotdview:
#                 sv.direct_view()
#     if kwargs['reflection']:
#         sampler = SunSampler(scn, sns, **kwargs)
#         lumf = f'{scn.outdir}/sun_kd_lum.dat'
#         exists = os.path.isfile(lumf) and os.stat(lumf).st_size > 1000
#         rrun = run and (overwrite or not exists)
#         if rrun:
#             sampler.run(replace=overwrite)
#         # try:
#         #     su = SunField(scn, sns, rebuild=rrun or rebuild, rmraw=rmraw)
#         # except FileNotFoundError as ex:
#         #     print(f'Warning: {ex}', file=sys.stderr)
#         # else:
#         #     ctx.obj['initlf'].append(su)
#         #     if plotdview:
#         #         items = list(su.items())
#         #         if len(items) >= 20:
#         #             items = None
#         #         su.direct_view(items=items, showsample=showsample,
#         #                        showweight=showweight, dpts=dpts, res=1024)
#
#
# @main.command()
# @clk.shared_decs(run_opts)
# @click.option('-skydef', help='sky scene file (.rad)',
#               type=click.Path(exists=True, dir_okay=False))
# @click.option('-skyname',
#               help='basename for result files')
# @click.option('-fdres', default=10,
#               help='the final directional sampling resolution, yielding a'
#                    ' grid of potential samples at 2^fdres x 2^fdres per'
#                    ' hemisphere')
# @click.option('-rcopts',
#               default='-ab 6 -ad 3000 -as 1500 -st 0 -ss 16 -aa .1',
#               help='rtrace options for sun reflection runs'
#                    ' see the man pages for rtrace, and '
#                    ' rtrace -defaults for more information')
# @click.option('--ambcache/--no-ambcache', default=True,
#               help='whether the rcopts indicate that the calculation will use '
#                    'ambient caching (and thus should write an -af file argument'
#                    ' to the engine)')
# @click.option('--keepamb/--no-keepamb', default=False,
#               help='whether to keep ambient files after run, if kept, a '
#                    'successive call will load these ambient files, so care '
#                    'must be taken to not change any parameters')
# @clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
# def onesky(ctx, skydef=None, skyname=None, plotdview=False, run=True,
#            rmraw=False, overwrite=False, rebuild=False, showsample=True,
#            showweight=True, dpts=None, **kwargs):
#     """the onesky command is for running one off sky definitions."""
#     if skydef is None:
#         print("Error: '-skydef' is required", file=sys.stderr)
#         raise click.Abort
#     if 'scene' not in ctx.obj:
#         clk.invoke_dependency(ctx, scene)
#     scn = ctx.obj['scene']
#     if skyname is None:
#         skyname = re.split(r"[/.]", skydef)[-2]
#     sampler = SkySampler(scn, skydef, skyname, **kwargs)
#     lumf = f'{scn.outdir}/{skyname}_kd_lum.dat'
#     exists = os.path.isfile(lumf) and os.stat(lumf).st_size > 1000
#     rrun = run and (overwrite or not exists)
#     if rrun:
#         sampler.run()
#     try:
#         su = StaticField(scn, sampler.sources, prefix=skyname,
#                          rebuild=rrun or rebuild, rmraw=rmraw)
#     except FileNotFoundError as ex:
#         print(f'Warning: {ex}', file=sys.stderr)
#     else:
#         ctx.obj['initlf'].append(su)
#         if plotdview:
#             items = list(su.items())
#             if len(items) >= 20:
#                 items = None
#             su.direct_view(items=items, showsample=showsample,
#                            showweight=showweight, dpts=dpts, res=1024)
#             if su.view is not None:
#                 su.view.direct_view()
#
#
# allmetrics = ["illum", "avglum", "gcr", "ugr", "dgp", "tasklum",
#               "backlum", "dgp_t1", "dgp_t2", "threshold", "pwsl2",
#               "view_area", "density", "reldensity", "lumcenter"]
#
#
# @main.command()
# @click.option('-loc', callback=clk.split_float,
#               help='specify the scene location (if not specified in -wea or to'
#                    ' override. give as "lat lon mer" where lat is + North, lon'
#                    ' is + West and mer is the timezone meridian (full hours are'
#                    ' 15 degree increments)')
# @click.option('-wea',
#               help="path to weather/sun position file. possible formats are:\n"
#                    "\n"
#                    "1. .wea file\n"
#                    "#. .wea file without header (requires -loc)\n"
#                    "#. .epw file\n"
#                    "#. .epw file without header (requires -loc)\n"
#                    "#. 4 column tsv file, each row is altitude, azimuth, direct"
#                    " normal, diff. horizontal of canditate suns (requires "
#                    "--usepositions)\n"
#                    "#. 5 column tsv file, each row is dx, dy, dz, direct "
#                    "normal, diff. horizontal of canditate suns (requires "
#                    "--usepositions)\n\n"
#                    "tsv files are loaded with loadtxt"
#               )
# @click.option('-pts', default='0,0,0,0,-1,0', callback=np_load,
#               help="points to evaluate, this can be a .npy file, a whitespace "
#                    "seperated text file or entered as a string with commas "
#                    "between components of a point and spaces between points. "
#                    "in all cases each point requires 6 numbers x,y,z,dx,dy,dz "
#                    "so the shape of the array will be (N, 6)")
# @click.option('-res', default=800,
#               help="the resolution of hdr output in pixels")
# @click.option('-interp', default=12,
#               help='number of nearby rays to use for interpolation of hdr'
#                    'output (weighted by a gaussian filter). this does'
#                    'not apply to metric calculations')
# @click.option('-blursun', default=1,
#               help='spread sun to mimic camera or human eye, 1 is no blur'
#                    ' a value of 4 will double the apparent radius')
# @click.option('-vname', default='view',
#               help='name to include with hdr outputs')
# @click.option('-static',
#               help='name of static field to integrate (overrides other opts)')
# @click.option('-skyro', default=0.0,
#               help='counter clockwise rotation (in degrees) of the sky to'
#                    ' rotate true North to project North, so if project North'
#                    ' is 10 degrees East of North, skyro=10')
# @click.option('-ground_fac', default=0.15,
#               help='ground reflectance')
# @click.option('--skyonly/--no-skyonly', default=False,
#               help="if True, only integrate on Sky Field, useful for "
#                    "diagnostics")
# @click.option('--sunonly/--no-sunonly', default=False,
#               help="if True, only integrate on Sun Field, useful for "
#                    "diagnostics. Note: only runs if --skyonly is False")
# @click.option('--hdr/--no-hdr', default=True,
#               help="produce an hdr output for each point and line in wea")
# @click.option('--metric/--no-metric', default=True,
#               help="calculate metrics for each point and wea file output"
#                    " is ordered by point than sky")
# @click.option('-metricset', default="illum avglum gcr dgp ugr",
#               callback=clk.split_str,
#               help='which metrics to return items must be in: '
#                    f'{", ".join(allmetrics)}')
# @click.option('-threshold', default=2000.0,
#               help='Threshold factor; if factor is larger than 100, it is used '
#                    'as constant threshold in cd/m2. If factor is less or equal '
#                    'than 100,  this  factor  multiplied  by the average task '
#                    'luminance will be used as  threshold for detecting the '
#                    'glare sources. task luminance is taken from the center'
#                    'of the view and encompasses tradius (see parameter '
#                    '-tradius)')
# @click.option('-tradius', default=30.0,
#               help='task radius in degrees for calculating task luminance')
# @click.option('--header/--no-header', default=False,
#               help='print column headings on metric output')
# @click.option('--statidx/--no-statidx', default=False,
#               help='print index columns on metric output')
# @click.option('--statsensor/--no-statsensor', default=False,
#               help='print sensor position and direction columns on metric '
#                    'output')
# @click.option('--staterr/--no-staterr', default=False,
#               help='print interpolation error info on metric output')
# @click.option('--statsky/--no-statsky', default=False,
#               help='print sky info (sun x,y,z dirnorm, dirdiff) on metric '
#                    'output')
# @clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
# def integrate(ctx, loc=None, wea=None, skyro=0.0, ground_fac=0.15, pts=None,
#               skyonly=False, sunonly=False, hdr=True, metric=True, res=800, interp=12,
#               vname='view', static=None, header=False, metricset="",
#               tradius=30.0, blursun=1, **kwargs):
#     """the integrate command combines sky and sun results and evaluates the
#     given set of positions and sky conditions"""
#     if 'scene' not in ctx.obj:
#         clk.invoke_dependency(ctx, scene)
#     scn = ctx.obj['scene']
#
#     if static is not None:
#         sk = StaticField(scn, prefix=static)
#         itg = BaseIntegrator(sk)
#     elif skyonly:
#         sk = SCBinField(scn)
#         itg = Integrator(sk, wea=wea, loc=loc, skyro=skyro,
#                          ground_fac=ground_fac)
#     else:
#         if 'suns' not in ctx.obj:
#             clk.invoke_dependency(ctx, suns)
#         sns = ctx.obj['suns']
#         # su = SunField(scn, sns, blursun=blursun)
#         # if sunonly:
#         #     itg = Integrator(su, wea=wea, loc=loc, skyro=skyro,
#         #                      ground_fac=ground_fac)
#         # else:
#         sk = SCBinField(scn)
#         itg = SunSkyIntegrator(sk, wea=wea, loc=loc, skyro=skyro,
#                                ground_fac=ground_fac)
#     metric_return_opts = {"idx": kwargs['statidx'],
#                           "sensor": kwargs['statsensor'],
#                           "err": kwargs['staterr'], "sky": kwargs['statsky']}
#     if header and metric:
#         print("\t".join(itg.metricheader))
#     data = itg.integrate(pts, interp=interp, res=res, vname=vname, scale=179,
#                          metricset=metricset, dohdr=hdr, dometric=metric,
#                          tradius=tradius, metric_return_opts=metric_return_opts)
#     #
#     # for d in data:
#     #     print("\t".join([f"{i}" for i in d]))
#
#
# def cross_interpolate(sk_kd, sk_vec, sk_lum, su_kd, su_vec, su_lum):
#     lum_sk = LightFieldKD.interpolate_query(sk_kd, sk_lum, sk_vec, su_vec, k=1)
#     lum_su = LightFieldKD.interpolate_query(su_kd, su_lum, su_vec, sk_vec, k=1)
#     vecs = np.vstack((su_vec, sk_vec))
#     lum_sk = np.vstack((lum_sk, sk_lum))
#     lum_su = np.vstack((su_lum, lum_su))
#     lums = np.hstack((lum_su, lum_sk))
#     kd, omega = LightFieldKD.mk_vector_ball(vecs)
#     return vecs, lums, np.squeeze(omega)
#
#
# def metric_out(sk_kd, sk_vlo, pt, perr, pi, scn, sns, skydat, dirs, header, metricset, outname, **kwargs):
#     metric_return_opts = {"idx"   : kwargs['statidx'],
#                           "sensor": kwargs['statsensor'],
#                           "err"   : kwargs['staterr'], "sky": kwargs['statsky']
#                           }
#     mcalc = Metric(scene, metricset, metric_return_opts)
#     viewf = SunViewField(scn, sns)
#     spi_o = None
#     hassun = False
#     f = open(f"{scn.outdir}_{outname}_{pi:04d}.out", 'w')
#     if header:
#         print("\t".join(mcalc.metricheader), file=f)
#     j = -1
#     for i, sd in enumerate(skydat.skydata):
#         if skydat.daysteps[i]:
#             j += 1
#             sunbin = skydat.sunproxy[j]
#             spi = (pi, sunbin[1])
#             if spi != spi_o:
#                 spi_o = spi
#                 pref = f"sun_{sunbin[1]:04d}"
#                 hassun = False
#                 if os.path.isfile(f"{scn.outdir}/{pref}_kd_lum.dat"):
#                     sf = LightFieldKD(scn, prefix=pref, fvrays=scn.maxspec, log=False)
#                     if pi in sf.d_kd.keys():
#                         if sf.d_kd[pi] is not None:
#                             su_vlo = cross_interpolate(sk_kd, sk_vlo[0],
#                                                        sk_vlo[1], sf.d_kd[pi],
#                                                        sf.vec[pi], sf.lum[pi])
#                             hassun = True
#             if sd[-2] > 0 and hassun:
#                 skyvec = np.concatenate((skydat.sun[j][-2:-1], skydat.smtx[j]))
#                 use_vlo = su_vlo
#                 serr = skydat._serr[j]
#             else:
#                 skyvec = np.copy(skydat.smtx[j])
#                 skyvec[sunbin[0]] += skydat.sun[j][-1]
#                 use_vlo = sk_vlo
#                 serr = -90
#             lum = np.einsum('j,kj->k', skyvec, use_vlo[1])
#             rays = use_vlo[0]
#             omega = use_vlo[2]
#             for vd in dirs[:, 0:3]:
#                 vm = ViewMapper(vd, 180)
#                 info = mcalc._metric_info([pi, i], [*pt, *vd],
#                                           [perr, serr],
#                                           skydat.skydata[j])
#                 svw = viewf.get_ray(spi, vm, skydat.sun[j])
#                 if svw is not None:
#                     rays = np.vstack((rays, svw[0][None, :]))
#                     lum = np.concatenate((lum, [svw[1]]))
#                     omega = np.concatenate((omega, [svw[2]]))
#                 fmetric = MetricSet(vm, rays, omega, lum, metricset)()
#                 data = np.nan_to_num(
#                     np.concatenate((info[0], fmetric, info[1])))
#                 for d in data:
#                     f.write(f"{d}\t")
#                 f.write("\n")
#     f.close()
#
#
#
#
# @main.command()
# @click.option('-loc', callback=clk.split_float,
#               help='specify the scene location (if not specified in -wea or to'
#                    ' override. give as "lat lon mer" where lat is + North, lon'
#                    ' is + West and mer is the timezone meridian (full hours are'
#                    ' 15 degree increments)')
# @click.option('-wea',
#               help="path to weather/sun position file. possible formats are:\n"
#                    "\n"
#                    "1. .wea file\n"
#                    "#. .wea file without header (requires -loc)\n"
#                    "#. .epw file\n"
#                    "#. .epw file without header (requires -loc)\n"
#                    "#. 4 column tsv file, each row is altitude, azimuth, direct"
#                    " normal, diff. horizontal of canditate suns (requires "
#                    "--usepositions)\n"
#                    "#. 5 column tsv file, each row is dx, dy, dz, direct "
#                    "normal, diff. horizontal of canditate suns (requires "
#                    "--usepositions)\n\n"
#                    "tsv files are loaded with loadtxt"
#               )
# @click.option('-pts', default='0,0,0', callback=np_load,
#               help="points to evaluate, this can be a .npy file, a whitespace "
#                    "seperated text file or entered as a string with commas "
#                    "between components of a point and spaces between points. "
#                    "in all cases each point requires 3 numbers x,y,z "
#                    "so the shape of the array will be (N, 3)")
# @click.option('-dirs', default='0,-1,0', callback=np_load,
#               help="directions to evaluate, this can be a .npy file, a whitespace "
#                    "seperated text file or entered as a string with commas "
#                    "between components of a point and spaces between points. "
#                    "in all cases each point requires 3 numbers dx,dy,dz "
#                    "so the shape of the array will be (N, 3)")
# @click.option('-skyro', default=0.0,
#               help='counter clockwise rotation (in degrees) of the sky to'
#                    ' rotate true North to project North, so if project North'
#                    ' is 10 degrees East of North, skyro=10')
# @click.option('-outname', default="metric",
#               help='naming for outfiles (combined with point index and scene)')
# @click.option('-ground_fac', default=0.15,
#               help='ground reflectance')
# @click.option('-metricset', default="illum avglum gcr dgp ugr",
#               callback=clk.split_str,
#               help='which metrics to return items must be in: '
#                    f'{", ".join(allmetrics)}')
# @click.option('-threshold', default=2000.0,
#               help='Threshold factor; if factor is larger than 100, it is used '
#                    'as constant threshold in cd/m2. If factor is less or equal '
#                    'than 100,  this  factor  multiplied  by the average task '
#                    'luminance will be used as  threshold for detecting the '
#                    'glare sources. task luminance is taken from the center'
#                    'of the view and encompasses tradius (see parameter '
#                    '-tradius)')
# @click.option('-tradius', default=30.0,
#               help='task radius in degrees for calculating task luminance')
# @click.option('--header/--no-header', default=False,
#               help='print column headings on metric output')
# @click.option('--statidx/--no-statidx', default=False,
#               help='print index columns on metric output')
# @click.option('--statsensor/--no-statsensor', default=False,
#               help='print sensor position and direction columns on metric '
#                    'output')
# @click.option('--staterr/--no-staterr', default=False,
#               help='print interpolation error info on metric output')
# @click.option('--statsky/--no-statsky', default=False,
#               help='print sky info (sun x,y,z dirnorm, dirdiff) on metric '
#                    'output')
# @clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
# def computemetric(ctx, loc=None, wea=None, skyro=0.0, ground_fac=0.15, pts=None,
#                   dirs=None, header=False, metricset="",
#                   tradius=30.0, outname="metric", **kwargs):
#     """the integrate command combines sky and sun results and evaluates the
#     given set of positions and sky conditions"""
#     if 'scene' not in ctx.obj:
#         clk.invoke_dependency(ctx, scene)
#     scn = ctx.obj['scene']
#     if 'suns' not in ctx.obj:
#         clk.invoke_dependency(ctx, suns)
#     sns = ctx.obj['suns']
#     skydat = SkyData(wea, scn, sns, loc=loc, skyro=skyro, ground_fac=ground_fac)
#     perrs, pis = scn.area.pt_kd.query(pts[:, 0:3])
#     skyf = SCBinField(scn)
#
#     for pt, perr, pi in zip(pts, perrs, pis):
#         sk_kd = skyf.d_kd[pi]
#         sk_vlo = (skyf.vec[pi], skyf.lum[pi], np.squeeze(skyf.omega[pi]))
#         metric_out(sk_kd, sk_vlo, pt, perr, pi, scn, sns, skydat, dirs, header, metricset, outname, **kwargs)
#
#
#
# @main.command()
# @click.option('-loc', callback=clk.split_float,
#               help='specify the scene location (if not specified in -wea or to'
#                    ' override. give as "lat lon mer" where lat is + North, lon'
#                    ' is + West and mer is the timezone meridian (full hours are'
#                    ' 15 degree increments)')
# @click.option('-wea',
#               help="path to weather/sun position file. possible formats are:\n"
#                    "\n"
#                    "1. .wea file\n"
#                    "#. .wea file without header (requires -loc)\n"
#                    "#. .epw file\n"
#                    "#. .epw file without header (requires -loc)\n"
#                    "#. 4 column tsv file, each row is altitude, azimuth, direct"
#                    " normal, diff. horizontal of canditate suns (requires "
#                    "--usepositions)\n"
#                    "#. 5 column tsv file, each row is dx, dy, dz, direct "
#                    "normal, diff. horizontal of canditate suns (requires "
#                    "--usepositions)\n\n"
#                    "tsv files are loaded with loadtxt"
#               )
# @click.option('-pts', default='0,0,0,0,-1,0', callback=np_load,
#               help="points to evaluate, this can be a .npy file, a whitespace "
#                    "seperated text file or entered as a string with commas "
#                    "between components of a point and spaces between points. "
#                    "in all cases each point requires 6 numbers x,y,z,dx,dy,dz "
#                    "so the shape of the array will be (N, 6)")
# @click.option('-res', default=800,
#               help="the resolution of hdr output in pixels")
# @click.option('-interp', default=12,
#               help='number of nearby rays to use for interpolation of hdr'
#                    'output (weighted by a gaussian filter). this does'
#                    'not apply to metric calculations')
# @click.option('-blursun', default=1,
#               help='spread sun to mimic camera or human eye, 1 is no blur'
#                    ' a value of 4 will double the apparent radius')
# @click.option('-vname', default='view',
#               help='name to include with hdr outputs')
# @click.option('-skyro', default=0.0,
#               help='counter clockwise rotation (in degrees) of the sky to'
#                    ' rotate true North to project North, so if project North'
#                    ' is 10 degrees East of North, skyro=10')
# @click.option('-ground_fac', default=0.15,
#               help='ground reflectance')
# @clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
# def computehdr(ctx, loc=None, wea=None, skyro=0.0, ground_fac=0.15, pts=None, **kwargs):
#     """the integrate command combines sky and sun results and evaluates the
#     given set of positions and sky conditions"""
#     if 'scene' not in ctx.obj:
#         clk.invoke_dependency(ctx, scene)
#     scn = ctx.obj['scene']
#     if 'suns' not in ctx.obj:
#         clk.invoke_dependency(ctx, suns)
#     sns = ctx.obj['suns']
#     skydat = SkyData(wea, scn, sns, loc=loc, skyro=skyro, ground_fac=ground_fac)
#     perrs, pis = scn.area.pt_kd.query(pts[:, 0:3])
#     skyf = SCBinField(scn)
#     for pt, perr, pi in zip(pts, perrs, pis):
#         for i, sd in enumerate(skydat.skydata):
#             sunbin = skydat.sunproxy[i]
#             spi = (pi, sunbin[1])
#             pref = f"sun_{sunbin[1]:04d}"
#             hassun = False
#             if os.path.isfile(f"{scn.outdir}/{pref}_kd_lum.dat"):
#                 sf = LightFieldKD(scn, prefix=pref, fvrays=scn.maxspec, log=False)
#                 sk = SunSkyPt(skyf, sf, pi)
#                 print("here")
#             else:
#                 print("not here")




@main.resultcallback()
@click.pass_context
def printconfig(ctx, returnvalue, **kwargs):
    """callback to print additional info and cleanup any temp files"""
    rv = dict([i[0:2] for i in returnvalue])
    try:
        if 'scene' in rv:
            info = rv['scene']['info']
        else:
            info = (ctx.command.commands['scene'].
                    context_settings['default_map']['info'])
        if str(info).lower() in ('1', 'yes', 'y', 't', 'true'):
            s = ctx.obj['scene']
            print('\nCallback Scene Info:', file=sys.stderr)
            print('='*60 + '\n', file=sys.stderr)
            try:
                suncount = ctx.obj['suns'].suns.shape[0]
                print(f'scene has {suncount} suns', file=sys.stderr)
            except KeyError:
                print('sun setter not initialized', file=sys.stderr)
            print('\n Lightfield Data:', file=sys.stderr)
            print('-'*60, file=sys.stderr)
            print('Has sky lightfield data:',
                  os.path.isfile(f'{s.outdir}/sky_kd_data.pickle'),
                  file=sys.stderr)
            print('Has sunview lightfield data:',
                  os.path.isfile(f'{s.outdir}/sunview_kd_data.pickle'),
                  file=sys.stderr)
            for lf in ctx.obj['initlf']:
                print(f"lightfield \"{lf.prefix}\" covers {lf.size['lfang']:.02f} "
                      f"steradians with {lf.size['lfcnt']} rays", file=sys.stderr)
    except KeyError:
        pass
    try:
        clk.tmp_clean(ctx)
    except Exception:
        pass


if __name__ == '__main__':
    main()
