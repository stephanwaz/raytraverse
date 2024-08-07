# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""Console script for raytraverse utilities."""
import re

import numpy as np

from clasp import click
import clasp.click_ext as clk

import raytraverse
from raytools import translate
from raytraverse import api
from raytraverse.lightfield import ResultAxis, ZonalLightResult
from raytraverse.lightfield import LightResult
from raytraverse.evaluate import hvsgsm
from raytraverse.sky import SkyData, skycalc
from raytools.utility import pool_call
from raytools import imagetools
from raytraverse.cli_funcs import np_load, shared_pull, pull_decs, np_load_safe


@click.group(chain=True, invoke_without_command=True)
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
    """the raytu executable is a command line interface to utility commands
    as part of the raytraverse python package.

    the easiest way to manage options is to use a configuration file,
    to make a template::

        raytu --template > run.cfg

    after adjusting the settings, than each command can be invoked in turn and
    any dependencies will be loaded with the correct options, for example::

        raytu -c run.cfg imgmetric pull

    will calculate metrics on a set of images and then print to the stdout.
    """
    raytraverse.io.set_nproc(n)
    ctx.info_name = 'raytu'
    clk.get_config_chained(ctx, config, None, None, None)
    ctx.obj = {}


@main.command()
@click.option("-d", callback=np_load,
              help="a .npy file, a whitespace "
                   "seperated text file (can be - for stdin) or "
                   "entered as a string with commas  between components of a "
                   "point and spaces between rows.")
@click.option("--flip/--no-flip", default=False,
              help="transpose matrix before transform (after reshape)")
@click.option("-reshape", callback=clk.split_int,
              help="reshape before transform (before flip)")
@click.option("-cols", callback=clk.split_int,
              help="coordinate columns (if none uses first N as required)")
@click.option("-op", default='xyz2xy',
              type=click.Choice(['xyz2xy', 'xyz2aa', 'xyz2tp', 'xyz2uv',
                                 'uv2xyz']),
              help="transformation: "
                   "'xyz2xy': cartesian direction vector to equiangular. "
                   "'xyz2aa': cartesian direction vector to alt/azimuth. "
                   "'xyz2tp': cartesian to spherical (normalized). "
                   "'xyz2uv': cartesian to shirley-chiu square. "
                   "'uv2xyz': shirley-chiu square to certesian. ")
@click.option("-outf", default=None,
              help="if none, return to stdout, else save as text file")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def transform(ctx, d=None, flip=False, reshape=None, cols=None,
              op='xyz2xy', outf=None, **kwargs):
    """coordinate transformations"""
    if d is None:
        click.echo("-d is required", err=True)
        raise click.Abort()
    if reshape is not None:
        if len(reshape) != 2:
            click.echo("must reshape to 2d", err=True)
            raise click.Abort()
        d = d.reshape(*reshape)
    if flip:
        d = d.T
    if cols is not None:
        d = d[:, cols]
    opfunc = {'xyz2xy': translate.xyz2xy, 'xyz2aa': translate.xyz2aa,
              'xyz2tp': translate.xyz2tp, 'xyz2uv': translate.xyz2uv,
              'uv2xyz': translate.uv2xyz}
    if op[0:3] == 'xyz':
        d = translate.norm(d[:, 0:3])
        if d.shape[1] != 3:
            click.echo(f"for xyz transform input must have 3 elem not "
                       f"{d.shape[1]}", err=True)
            raise click.Abort()
    if op[0:2] == 'uv':
        if d.shape[1] != 2:
            click.echo(f"for uv transform input must have 2 elem not "
                       f"{d.shape[1]}", err=True)
            raise click.Abort()
    out = opfunc[op](d)
    if outf is None:
        for o in out:
            print(*o)
    else:
        np.savetxt(outf, out)


@main.command()
@click.option("-imgs", callback=clk.are_files,
              help="hdr image files, must be angular fisheye projection,"
                   "if no view in header, assumes 180 degree")
@click.option("-metrics", callback=clk.split_str, default="illum dgp ugp",
              help='metrics to compute, choices: ["illum", '
                   '"avglum", "gcr", "ugp", "dgp", "tasklum", "backlum", '
                   '"dgp_t1", "log_gc", "dgp_t2", "ugr", "threshold", "pwsl2", '
                   '"view_area", "backlum_true", "srcillum", "srcarea", '
                   '"maxlum", "gss"]')
@click.option("--parallel/--no-parallel", default=True,
              help="use available cores")
@click.option("-basename", default="img_metrics",
              help="LightResult object is written to basename.npz.")
@click.option("--npz/--no-npz", default=True,
              help="write LightResult object to .npz, use 'raytraverse pull'"
                   "or LightResult('basename.npz') to access results")
@click.option("--peakn/--no-peakn", default=True,
              help="corrrect aliasing and/or filtering artifacts for direct sun"
                   " by assigning up to expected energy to peakarea")
@click.option("-peaka", default=6.7967e-05,
              help="expected peak area over which peak energy is distributed")
@click.option("-peakt", default=1.0e5,
              help="include down to this threshold in possible peak, note that"
                   "once expected peak energy is satisfied remaining pixels are"
                   "maintained, so it is safe-ish to keep this value low")
@click.option("-peakr", default=4.0,
              help="for peaks that do not meet expected area (such as partial"
                   " suns, to determines the ratio of what counts as part of"
                   " the source (max/peakr)")
@click.option("-threshold", default=2000.,
              help="same as the evalglare -b option. if factor is larger than "
                   "100, it is used as constant threshold in cd/m2, else this "
                   "factor is multiplied by the average task luminance. task "
                   "position is center of image with a 30 degree field of view")
@click.option("-scale", default=179.,
              help="scale factor applied to pixel values to convert to cd/m^2")
@click.option("--blursun/--no-blursun", default=False,
              help="applies human PSF to peak glare source (only if peekn=True")
@click.option("--lowlight/--no-lowlight", default=False,
              help="use lowlight correction for dgp")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def imgmetric(ctx, imgs=None, metrics=None, parallel=True,
              basename="img_metrics", npz=True, peakn=False,
              peaka=6.7967e-05, peakt=1e5, peakr=4.0, threshold=2000.,
              scale=179., blursun=False, lowlight=False, **kwargs):
    """calculate metrics for hdr images, similar to evalglare but without
    glare source grouping, equivalent to -r 0 in evalglare. This ensures that
    all glare source positions are  weighted by the metrics to which they are
    applied. Additional peak normalization reduces the deviation between images
    processed in different ways, for example pfilt with -r, rpict drawsource(),
    or an undersampled vwrays | rtrace run where the pixels give a coarse
    estimate of the actual sun area."""
    if parallel:
        cap = None
    else:
        cap = 1
    try:
        gss = metrics.pop(metrics.index("gss"))
    except ValueError:
        gss = False
    if len(metrics) > 0:
        results = pool_call(imagetools.imgmetric, list(zip(imgs)), metrics,
                            cap=cap, desc="processing images", peakn=peakn,
                            peaka=peaka, peakt=peakt, peakr=peakr,
                            threshold=threshold, scale=scale, blursun=blursun,
                            lowlight=lowlight)
        results = np.asarray(results)
    else:
        results = None
    if gss:
        gresult = np.asarray(hvsgsm.gss_compute(imgs, age=35,
                                                pigmentation=0.142))[:, None]
        metrics.append("gss")
        if results is not None:
            results = np.hstack((results, gresult))
        else:
            results = gresult
    imgaxis = ResultAxis(imgs, "image")
    metricaxis = ResultAxis(metrics, "metric")
    lr = LightResult(results, imgaxis, metricaxis)
    if npz:
        lr.write(f"{basename}.npz")
    ctx.obj['lightresult'] = lr


@main.command()
@click.option("-img", callback=clk.are_files,
              help="hdr image files, uv anr angular")
@click.option("--uv2ang/--ang2uv", default=False,
              help="direction of transform")
@click.option("--useview/--no-useview", default=True,
              help="use view direction for transform ang2uv to match standard"
                   "projection")
@click.option("-rotate", default=None, type=float,
              help="degrees to rotate img (overrides projection, "
                   "assumes angular fisheye input)")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def project(ctx, img=None, uv2ang=False, useview=True, rotate=None, **kwargs):
    """project images between angular and shirley-chiu square coordinates"""
    if rotate is not None:
        func = imagetools.hdr_rotate
    elif uv2ang:
        func = imagetools.hdr_uv2ang
    else:
        func = imagetools.hdr_ang2uv
    results = pool_call(func, img, expandarg=False, useview=useview, rotate=rotate)


# @main.command()
# @click.option('-out', default="imglf", type=click.Path(file_okay=False),
#               help="output directory")
# @click.option("-imga", callback=clk.are_files,
#               help="hdr image files, primary view direction, must be angular "
#                    "fisheye projection, view header required")
# @click.option("-imgb", callback=clk.are_files,
#               help="hdr image files, opposite view direction, must be angular "
#                    "fisheye projection, assumed to be same as imga with -vd"
#                    " reversed")
# @clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
# def img2lf(ctx, imga=None, imgb=None, out="imglf", **kwargs):
#     """read and compress angular fisheye images into lightpoints/lightplane"""
#     if imga is None:
#         click.echo("-imga is required", err=True)
#         raise click.Abort
#     if imgb is None:
#         imgb = [None] * len(imga)
#     elif len(imgb) < len(imga):
#         imgb = imgb + [None] * (len(imga) - len(imgb))
#     scene = ImageScene(out)
#     srcs = [f"img{i:03d}" for i in range(len(imga))]
#     results = pool_call(imagetools.img2lf, list(zip(imga, imgb, srcs)), scn=scene)


def _dview(x, **kwargs):
    lpt = api.load_lp(x)
    lpt.direct_view(**kwargs)
    return lpt.lum.shape[0]


@main.command()
@click.option('-lp', callback=clk.are_files, required=True,
              help="path to lightpoint(s)")
@click.option('-res', default=800)
@click.option('--showsample/--no-showsample', default=False)
@click.option('-srcidx', default=None, type=int)
@click.option('-scalefactor', default=1.0)
@click.option('-interp', type=click.Choice(['fast', 'high','', 'None',
                                            'False']))
@click.option('--omega/--no-omega', default=False)
@click.option('--order/--no-order', default=False)
@click.option('--fisheye/--no-fisheye', default=True)
@click.option('--showweight/--no-showweight', default=True)
@click.option('--rnd/--no-rnd', default=False)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def lp2img(ctx, lp=None, res=800, showsample=False, showweight=True, rnd=False,
           srcidx=None, interp=False, omega=False, order=False, scalefactor=1,
           fisheye=True, **kwargs):
    """make hdr directview of lightpoint"""
    rays = pool_call(_dview, lp, expandarg=False, res=res,
                     showsample=showsample, showweight=showweight, rnd=rnd,
                     srcidx=srcidx, interp=interp, omega=omega, order=order,
                     scalefactor=scalefactor, fisheye=fisheye,)
    click.echo("lightpoint \t samples")
    for r, l in zip(rays, lp):
        click.echo(f"{l} \t {r}")


@main.command()
@clk.shared_decs(pull_decs)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def pull(*args, **kwargs):
    return shared_pull(*args, **kwargs)


@main.command()
@click.option('-out', default="merged",
              help="output file to write, will change/append .npz extension")
@click.option('-lr', callback=clk.are_files, required=True,
              help="lightresults to merge, must match along all axes except"
                   " 'axis")
@click.option('-axis', default='sky',
              help="axis along which to merge. in case of duplicate values, output"
                   " uses first occurence from order of -lr, otherwise axes order"
                   " is not changed")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def merge(ctx, out="merged", lr=None, axis='sky', **kwargs):
    if re.match(r'.+\.[\w\s]+$', out):
        out = out.rsplit(".", 1)[0] + ".npz"
    else:
        out += ".npz"
    if lr is None:
        lr = []
    if len(lr) < 2:
        click.echo("merge requires atleast two light results", err=True)
        raise click.Abort
    try:
        lr0 = LightResult(lr[0])
    except KeyError:
        lr0 = ZonalLightResult(lr[0])
    try:
        mlr = [type(lr0)(i) for i in lr[1:]]
    except (KeyError, ValueError):
        click.echo("-lr cannot mix LightResult and ZonalLightResult"
                   " check with raytraverse/raytu pull --info", err=True)
        raise click.Abort
    result = lr0.merge(*mlr, axis=axis)
    result.write(out)


@main.command()
@click.option("-wea", callback=np_load_safe,
              help="path to epw, wea, .npy file or np.array, or .npz file,"
                   "if loc not set attempts to extract location data "
                   "(if needed).")
@click.option("-loc", default=None, callback=clk.split_float,
              help="location data given as 'lat lon mer' with + west of prime "
                   "meridian overrides location data in wea")
@click.option("-minalt", default=2.0,
              help="minimum solar altitude for daylight masking")
@click.option("-mindiff", default=5.0,
              help="minumum diffuse horizontal irradiance for daylight masking")
@click.option("-mindir", default=0.0,
              help="minumum direct normal irradiance for daylight masking")
@click.option("-data", callback=np_load,
              help="data to pad")
@click.option("-cols", callback=clk.split_int,
              help="cols of data to return (default all)")
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def padsky(ctx, wea=None, loc=None, opts=False,
            debug=False, version=None, data=None, cols=None, **kwargs):
    """pad filtered result data according to sky filtering
    """
    if data is None or wea is None:
        click.echo("-wea and -data are required", err=True)
        raise click.Abort
    if loc is not None:
        loc = (loc[0], loc[1], int(loc[2]))
    sd = SkyData(wea, loc=loc, skyres=3, **kwargs)
    if len(data) != sd.daysteps:
        raise ValueError(f"data has {len(data)} rows, but {sd.daysteps} were "
                         f"expected by SkyData")
    if cols is None:
        cols = slice(None)
    d = sd.fill_data(data)[:,cols]
    try:
        s = wea.shape
    except AttributeError:
        wd = skycalc.read_epw(wea)
        d = np.hstack((wd[:, 0:3], d))
        intcol = 2
    else:
        intcol = 0
    for i in d:
        print("\t".join([str(int(j)) for j in i[0:intcol]] +
                        [f"{j}" for j in i[intcol:]]))





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
