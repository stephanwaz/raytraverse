# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""Console script for raytraverse."""

import numpy as np

from clasp import click
import clasp.click_ext as clk
import raytraverse
from raytraverse import SunViewSampler, SunSetter, SunSampler, Scene, Integrator
from raytraverse import SCBinSampler, SkyIntegrator

__version__ = raytraverse.__version__


def invoke_scene(ctx):
    clk.invoke_dependency(ctx, 'scene', 'out', Scene)


def invoke_suns(ctx):
    clk.invoke_dependency(ctx, 'suns', 'scene', SunSetter)


@click.group(chain=True, invoke_without_command=True)
@click.argument('out')
@clk.shared_decs(clk.main_decs(__version__))
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
    skyint.filter_sky_pdf(s.suns, maxspec=kwargs['maxspec'],
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
@click.option('-rcopts', default='-ab 0')
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def sunview(ctx, **kwargs):
    """run sunviewsampler"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    if 'suns' not in ctx.obj:
        invoke_suns(ctx)
    sampler = SunViewSampler(ctx.obj['scene'], ctx.obj['suns'])
    sampler.run(rcopts=kwargs['rcopts'])


@main.command()
@click.option('-minrate', default=.005)
@click.option('-maxrate', default=.01)
@click.option('-idres', default=10)
@click.option('-fdres', default=12)
@click.option('-maxspec', default=.3)
@click.option('-wpow', default=.5)
@click.option('-rcopts', default='-ab 2 -ad 1024 -as 0 -lw 1e-5 -st 0 -ss 16')
@click.option('--mkpmap/--no-mkpmap', default=False)
@click.option('-apo', default='', callback=clk.split_str)
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def sunrefl(ctx, mkpmap=False, apo=[], **kwargs):
    """run sunsampler"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    if 'suns' not in ctx.obj:
        invoke_suns(ctx)
    sampler = SunSampler(ctx.obj['scene'], ctx.obj['suns'], **kwargs)
    if mkpmap:
        sampler.mkpmap(apo)
    sampler.run(rcopts=kwargs['rcopts'])


@main.command()
@click.option('--rebuild/--no-rebuild', default=False)
@clk.shared_decs(clk.command_decs(__version__, wrap=True))
def image(ctx, rebuild=False, **kwargs):
    """build integrator and make images"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    if 'suns' not in ctx.obj:
        invoke_suns(ctx)
    # ski = SkyIntegrator(ctx.obj['scene'], rebuild=rebuild)
    # svi = SunViewIntegrator(ctx.obj['scene'], rebuild=rebuild)
    sri = Integrator(ctx.obj['scene'], rebuild=rebuild, prefix='sunreflect')
    coefs = np.array([(i, 1) for i in range(ctx.obj['suns'].suns.shape[0])])
    sri.view(ctx.obj['scene'].pts(), np.array([[0, -1, 0]]),
             coefs=coefs, maxl=-2, decades=5, mark=False, viewangle=180)


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
