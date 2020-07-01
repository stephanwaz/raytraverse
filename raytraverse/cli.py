# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""Console script for raytraverse."""
from io import StringIO
import sys

import numpy as np

from clasp import click
import clasp.click_ext as clk
import raytraverse
from raytraverse import translate, sunpos, io
from raytraverse.sampler import SCBinSampler, SunViewSampler, SunSampler, SunRunner
from raytraverse.integrator import SunViewIntegrator, SkyIntegrator, Integrator
from raytraverse.scene import Scene, SunSetter
from raytraverse.field import SunView
from clipt import mplt

__version__ = raytraverse.__version__


def invoke_scene(ctx):
    clk.invoke_dependency(ctx, 'scene', 'out', Scene)


def invoke_suns(ctx):
    clk.invoke_dependency(ctx, 'suns', 'scene', SunSetter)


@click.group(chain=True, invoke_without_command=True)
@click.argument('out')
@clk.shared_decs(clk.main_decs(__version__))
@click.option('--template/--no-template', is_eager=True,
              callback=clk.printconfigs,
              help="write default options to std out as config")
@click.option('--opts', '-opts', is_flag=True,
              help="check parsed options")
@click.option('--debug', is_flag=True,
              help="show traceback on exceptions")
def main(ctx, out, config, outconfig, **kwargs):
    """docstring"""
    clk.get_config_chained(ctx, config, outconfig, None, None)
    ctx.obj = dict(out=out)


@main.command()
@click.option('-scene', help='space separated list of radiance scene files '
              '(no sky) or octree')
@click.option('-area', help='radiance scene file containing planar geometry of '
              'analysis area')
@click.option('-wea', help='path to epw or wea file, if loc not set attempts to'
              ' extract location data')
@click.option('-ptres', default=2.0)
@click.option('--reload/--no-reload', default=True)
@click.option('--overwrite/--no-overwrite', default=False)
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def scene(ctx, **kwargs):
    """load scene"""
    s = Scene(ctx.obj['out'], **kwargs)
    ctx.obj['scene'] = s


@main.command()
@click.option('-sunres', default=5.0)
@click.option('-srct', default=.01)
@click.option('-maxspec', default=.3)
@click.option('--reload/--no-reload', default=True)
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def suns(ctx, **kwargs):
    """create sun positions"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    skyint = SkyIntegrator(ctx.obj['scene'])
    skyint.write_skydetail(reload=kwargs['reload'])
    s = SunSetter(ctx.obj['scene'], **kwargs)
    click.echo(f'scene has {s.suns.shape[0]} sun positions', err=True)
    skyint.write_sun_pdfs(s.suns, maxspec=kwargs['maxspec'],
                          reload=kwargs['reload'])
    ctx.obj['suns'] = s


@main.command()
@click.option('-minrate', default=.05)
@click.option('-t0', default=.1)
@click.option('-t1', default=0)
@click.option('-idres', default=4)
@click.option('-fdres', default=9)
@click.option('-srcn', default=20)
@click.option('-rcopts', default='-ab 2 -ad 1024 -as 0 -lw 1e-5 -st 0 -ss 16')
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def sky(ctx, **kwargs):
    """run scbinsampler"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    sampler = SCBinSampler(ctx.obj['scene'], **kwargs)
    sampler.run(rcopts=kwargs['rcopts'], executable='rcontrib')
    click.echo("building kd-tree and evaluation data", err=True)
    SkyIntegrator(ctx.obj['scene'])


@main.command()
@click.option('-minrate', default=.005)
@click.option('-maxrate', default=.01)
@click.option('-idres', default=9)
@click.option('-fdres', default=10)
@click.option('-maxspec', default=.3)
@click.option('-wpow', default=.5)
@click.option('-rcopts', default='-ab 1 -ad 1024 -aa 0 -as 0 -lw 1e-5 -st 0 -ss 4')
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def sunrun(ctx, **kwargs):
    """run sunsampler"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    if 'suns' not in ctx.obj:
        invoke_suns(ctx)
    sampler = SunRunner(ctx.obj['scene'], ctx.obj['suns'], **kwargs)
    sampler.run(rcopts=kwargs['rcopts'])


@main.command()
@click.option('--rebuild/--no-rebuild', default=False)
@click.option('-sidx', default=0)
@click.option('--mark/--no-mark', default=False)
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def sunimage(ctx, rebuild=False, sidx=0, mark=False, **kwargs):
    """build integrator and make images"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    if 'suns' not in ctx.obj:
        invoke_suns(ctx)
    ski = SkyIntegrator(ctx.obj['scene'], rebuild=rebuild)
    coefs = np.zeros(400)
    sunuv = translate.xyz2uv(ctx.obj['suns'].suns)
    side = 20
    sunbin = translate.uv2bin(sunuv, side).astype(int)
    print(sunbin[sidx])
    coefs[sunbin[sidx]] = 1
    ski.view(ctx.obj['scene'].pts(), np.array([[0, -1, 0]]),
             maxl=-1, decades=5, mark=mark, viewangle=180, coefs=coefs)
    # svi = SunViewIntegrator(ctx.obj['scene'], rebuild=rebuild)
    sri = Integrator(ctx.obj['scene'], rebuild=rebuild, prefix=f'sunr_{sidx:04d}')
    sri.view(ctx.obj['scene'].pts(), np.array([[0, -1, 0]]),
             maxl=-1, decades=5, mark=mark, viewangle=180)


@main.command()
@click.option('--rebuild/--no-rebuild', default=False)
@click.option('-prefix', default='sky')
@click.option('--mark/--no-mark', default=False)
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def image(ctx, rebuild=False, prefix='sky', mark=False, **kwargs):
    """build integrator and make images"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    if 'suns' not in ctx.obj and prefix is not 'sky':
        invoke_suns(ctx)
    ski = Integrator(ctx.obj['scene'], prefix=prefix, rebuild=rebuild)
    ski.view(ctx.obj['scene'].pts(), np.array([[0, -1, 0]]),
             maxl=0, decades=5, mark=mark, viewangle=180)


@main.command()
@click.option('--rebuild/--no-rebuild', default=False)
@click.option('--mark/--no-mark', default=False)
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def sunview(ctx, rebuild=False, mark=False, **kwargs):
    """build integrator and make images"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    if 'suns' not in ctx.obj:
        invoke_suns(ctx)
    ski = SunView(ctx.obj['scene'], ctx.obj['suns'], rebuild=rebuild)
    np.set_printoptions(2)
    sxyz = translate.aa2xyz(ctx.obj['scene'].skydata[:, 0:2])
    xyz = sxyz[sxyz[:,2] > 0]
    idx, err = ski.query(ctx.obj['scene'].pts(), xyz)
    hour = np.mod(np.arange(8760), 24)
    xy = translate.xyz2xy(xyz, flip=False)
    sxy = translate.xyz2xy(ctx.obj['suns'].suns, flip=False)
    mplt.quick_scatter([xy[:,0], sxy[:,0]], [xy[:,1], sxy[:,1]], lw=0, ms=(2, 5), cs=[err[:, 1], np.zeros(50)])
    # print(sxyz[sxyz[:,2] > 0])

    # # perr, pis, serr, sis = ski.query(ctx.obj['suns'].suns[1:5], ctx.obj['scene'].pts())
    # # for pi in pis:
    # #     for si in sis:
    # #         print(ski.vec[pi][si].shape, ski.lum[pi][si].shape)
    # ski.view_together(ctx.obj['scene'].pts(), ctx.obj['suns'].suns, np.array([[0, -1, 0]]),
    #                   maxl=0, decades=5, mark=mark, viewangle=180, colors='spring')


@main.resultcallback()
@click.pass_context
def printconfig(ctx, returnvalue, **kwargs):
    """callback to save config file"""
    try:
        clk.tmp_clean(ctx)
    except Exception:
        pass
    if kwargs['outconfig']:
        for r in returnvalue:
            clk.print_config(ctx, (r[0], r[1]), kwargs['outconfig'],
                             kwargs['config'], kwargs['configalias'],
                             chain=True)


if __name__ == '__main__':
    main()
