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


def invoke_dependency(ctx, cmd, objkey, obj):
    kws = ctx.parent.command.commands[cmd].context_settings['default_map']
    for p in ctx.parent.command.commands[cmd].params:
        try:
            kws[p.name] = p.process_value(ctx, kws[p.name])
        except KeyError:
            kws[p.name] = p.get_default(ctx)
    kws.update(reload=True, overwrite=False)
    s = obj(ctx.obj[objkey], **kws)
    ctx.obj[cmd] = s


def invoke_scene(ctx):
    invoke_dependency(ctx, 'scene', 'out', Scene)


def invoke_suns(ctx):
    invoke_dependency(ctx, 'suns', 'scene', SunSetter)


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
@clk.shared_decs(clk.command_decs(__version__))
def scene(ctx, **kwargs):
    """load scene"""
    kwargs['opts'] = kwargs['opts'] or ctx.parent.params['opts']
    kwargs['debug'] = kwargs['debug'] or ctx.parent.params['debug']
    if kwargs['opts']:
        kwargs['opts'] = False
        click.echo('\nscbin options:\n', err=True)
        clk.echo_args(**kwargs)
    else:
        try:
            s = Scene(ctx.obj['out'], **kwargs)
            ctx.obj['scene'] = s
        except click.Abort:
            raise
        except Exception as ex:
            clk.print_except(ex, kwargs['debug'])
            raise click.Abort
    return 'scene', kwargs, ctx


@main.command()
@click.option('-sunres', default=5.0)
@click.option('-srct', default=.01)
@click.option('-maxspec', default=.3)
@click.option('--reload/--no-reload', default=True)
@clk.shared_decs(clk.command_decs(__version__))
def suns(ctx, **kwargs):
    """run scbinsampler"""
    kwargs['opts'] = kwargs['opts'] or ctx.parent.params['opts']
    kwargs['debug'] = kwargs['debug'] or ctx.parent.params['debug']
    if kwargs['opts']:
        kwargs['opts'] = False
        click.echo('\nsuns options:\n', err=True)
        clk.echo_args(**kwargs)
    else:
        try:
            if 'scene' not in ctx.obj:
                invoke_scene(ctx)
            s = SunSetter(ctx.obj['scene'], **kwargs)
            click.echo(f'scene has {s.suns.shape[0]} sun positions', err=True)
            click.echo("rewriting sky_pdf", err=True)
            skyint = SkyIntegrator(ctx.obj['scene'])
            skyint.filter_sky_pdf(s.suns, maxspec=kwargs['maxspec'],
                                  reload=kwargs['reload'])
            ctx.obj['suns'] = s
        except click.Abort:
            raise
        except Exception as ex:
            clk.print_except(ex, kwargs['debug'])
            raise click.Abort
    return 'suns', kwargs, ctx


@main.command()
@click.option('-minrate', default=.05)
@click.option('-t0', default=.1)
@click.option('-t1', default=0)
@click.option('-idres', default=4)
@click.option('-fdres', default=9)
@click.option('-srcn', default=20)
@click.option('-rcopts', default='-ab 2 -ad 1024 -as 0 -lw 1e-5 -st 0 -ss 16')
@clk.shared_decs(clk.command_decs(__version__))
def sky(ctx, **kwargs):
    """run scbinsampler"""
    kwargs['opts'] = kwargs['opts'] or ctx.parent.params['opts']
    kwargs['debug'] = kwargs['debug'] or ctx.parent.params['debug']
    if kwargs['opts']:
        kwargs['opts'] = False
        click.echo('\nsky options:\n', err=True)
        clk.echo_args(**kwargs)
    else:
        try:
            if 'scene' not in ctx.obj:
                invoke_scene(ctx)
            sampler = SCBinSampler(ctx.obj['scene'], **kwargs)
            sampler.run(rcopts=kwargs['rcopts'], executable='rcontrib')
            click.echo("building kd-tree and evaluation data", err=True)
            SkyIntegrator(ctx.obj['scene'])
        except click.Abort:
            raise
        except Exception as ex:
            clk.print_except(ex, kwargs['debug'])
            raise click.Abort
    return 'sky', kwargs, ctx


@main.command()
@click.option('-rcopts', default='-ab 0')
@clk.shared_decs(clk.command_decs(__version__))
def sunview(ctx, **kwargs):
    """run sunviewsampler"""
    kwargs['opts'] = kwargs['opts'] or ctx.parent.params['opts']
    kwargs['debug'] = kwargs['debug'] or ctx.parent.params['debug']
    if kwargs['opts']:
        kwargs['opts'] = False
        click.echo('\nsunview options:\n', err=True)
        clk.echo_args(**kwargs)
    else:
        try:
            if 'scene' not in ctx.obj:
                invoke_scene(ctx)
            if 'suns' not in ctx.obj:
                invoke_suns(ctx)
            sampler = SunViewSampler(ctx.obj['scene'], ctx.obj['suns'])
            sampler.run(rcopts=kwargs['rcopts'])
        except click.Abort:
            raise
        except Exception as ex:
            clk.print_except(ex, kwargs['debug'])
            raise click.Abort
    return 'sunview', kwargs, ctx


@main.command()
@click.option('-minrate', default=.005)
@click.option('-maxrate', default=.01)
@click.option('-idres', default=10)
@click.option('-fdres', default=12)
@click.option('-maxspec', default=.3)
@click.option('-wpow', default=.5)
@click.option('-rcopts', default='-ab 2 -ad 1024 -as 0 -lw 1e-5 -st 0 -ss 16')
@clk.shared_decs(clk.command_decs(__version__))
def sunrefl(ctx, **kwargs):
    """run sunsampler"""
    kwargs['opts'] = kwargs['opts'] or ctx.parent.params['opts']
    kwargs['debug'] = kwargs['debug'] or ctx.parent.params['debug']
    if kwargs['opts']:
        kwargs['opts'] = False
        click.echo('\nsunrefl options:\n', err=True)
        clk.echo_args(**kwargs)
    else:
        try:
            if 'scene' not in ctx.obj:
                invoke_scene(ctx)
            if 'suns' not in ctx.obj:
                invoke_suns(ctx)
            sampler = SunSampler(ctx.obj['scene'], ctx.obj['suns'], **kwargs)
            sampler.run(rcopts=kwargs['rcopts'])
        except click.Abort:
            raise
        except Exception as ex:
            clk.print_except(ex, kwargs['debug'])
            raise click.Abort
    return 'sunrefl', kwargs, ctx


@main.command()
@click.argument('prefix', type=click.Choice(['sky', 'sunreflect', 'sunview']))
@click.option('--rebuild/--no-rebuild', default=False)
@clk.shared_decs(clk.command_decs(__version__))
def image(ctx, prefix, **kwargs):
    """build integrator and make images"""
    kwargs['opts'] = kwargs['opts'] or ctx.parent.params['opts']
    kwargs['debug'] = kwargs['debug'] or ctx.parent.params['debug']
    if kwargs['opts']:
        kwargs['opts'] = False
        click.echo('\nimage options:\n', err=True)
        clk.echo_args(**kwargs)
    else:
        try:
            if 'scene' not in ctx.obj:
                invoke_scene(ctx)
            if 'suns' not in ctx.obj:
                invoke_suns(ctx)
            itg = Integrator(ctx.obj['scene'], prefix=prefix,
                             rebuild=kwargs['rebuild'])
            if prefix is 'sunreflect':
                coefs = np.array([(i, 1)
                                  for i in range(len(ctx.obj['suns'].suns))])
            itg.view(ctx.obj['scene'].pts(), np.array([[0, -1, 0]]),
                     coefs=coefs, maxl=-1, decades=5, mark=False, viewangle=180)
        except click.Abort:
            raise
        except Exception as ex:
            clk.print_except(ex, kwargs['debug'])
            raise click.Abort
    return 'image', kwargs, ctx


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
