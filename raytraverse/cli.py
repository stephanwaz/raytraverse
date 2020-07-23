# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""Console script for raytraverse."""
import os

import numpy as np

from clasp import click
import clasp.click_ext as clk
import raytraverse
from raytraverse.integrator import Integrator
from raytraverse.sampler import SCBinSampler, SunSampler
from raytraverse.scene import Scene, SunSetter
from raytraverse.lightfield import SCBinField, SunField


def invoke_scene(ctx):
    clk.invoke_dependency(ctx, 'scene', 'out', Scene)


def invoke_suns(ctx):
    clk.invoke_dependency(ctx, 'suns', 'scene', SunSetter)


@click.group(chain=True, invoke_without_command=True)
@click.argument('out')
@clk.shared_decs(clk.main_decs(raytraverse.__version__))
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
@click.option('-skyres', default=10.0)
@click.option('-ptres', default=2.0)
@click.option('-maxspec', default=0.3)
@click.option('--reload/--no-reload', default=True)
@click.option('--overwrite/--no-overwrite', default=False)
@click.option('--info/--no-info', default=False, help='print info on scene')
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def scene(ctx, **kwargs):
    """load scene"""
    s = Scene(ctx.obj['out'], **kwargs)
    ctx.obj['scene'] = s
    if kwargs['info']:
        print(f'\nScene {s.outdir}:')
        print('='*60 + '\n')
        print('Scene Geometry:')
        os.system(f'getinfo {s.scene}')
        print('Analysis Area:')
        print('-'*60)
        print(f'extents:\n{s.area.bbox}')
        print(f'resolution: {s.area.sf/s.ptshape}')
        print(f'number of points: {s.ptshape[0]*s.ptshape[1]}')
        try:
            suncount = len(open(f'{s.outdir}/sun_modlist.txt').readlines())
            print(f'scene has {suncount} suns')
        except FileNotFoundError:
            print('sun setter not initialized')
        print('\nSimulation Data:')
        print('-'*60)
        try:
            if os.path.isfile(f'{s.outdir}/sky_vals.out'):
                print('Sky simulation data exists!')
                scheme = np.load(f'{s.outdir}/sky_scheme.npy').astype(int)
                print(f'Sampling scheme:\n{scheme}')
            else:
                print('no simulation data exists!')
                ctx.exit()
            if os.path.isfile(f'{s.outdir}/sunview_vals.out'):
                print('Sun view simulation data exists!')
                print('Sun sampling file size: ' +
                      str(os.path.getsize(f'{s.outdir}/sunview_vals.out')) +
                      ' bytes')
            else:
                print('no sun view data')
        except FileNotFoundError as e:
            print(f'Bad simulation data, file should exist: {e}')
            ctx.exit()
        print('\n Lightfield Data:')
        print('-'*60)
        print('Has sky lightfield data:',
              os.path.isfile(f'{s.outdir}/sky_kd_data.pickle'))
        print('Has sunview lightfield data:',
              os.path.isfile(f'{s.outdir}/sunview_kd_data.pickle'))


@main.command()
@click.option('-minrate', default=.05)
@click.option('-t0', default=.1)
@click.option('-t1', default=0)
@click.option('-idres', default=4)
@click.option('-fdres', default=9)
@click.option('-rcopts', default='-ab 2 -ad 1024 -as 0 -lw 1e-5 -st 0 -ss 16')
@click.option('--plotp/--no-plotp', default=False)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sky(ctx, **kwargs):
    """run scbinsampler"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    sampler = SCBinSampler(ctx.obj['scene'], **kwargs)
    sampler.run(rcopts=kwargs['rcopts'], executable='rcontrib')
    sk = SCBinField(ctx.obj['scene'], rebuild=True)
    sk.direct_view()


@main.command()
@click.option('-srct', default=.01)
@click.option('--reload/--no-reload', default=True)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def suns(ctx, **kwargs):
    """create sun positions"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    s = SunSetter(ctx.obj['scene'], **kwargs)
    s.write_sun_pdfs()
    s.direct_view()
    click.echo(f'scene has {s.suns.shape[0]} sun positions', err=True)
    ctx.obj['suns'] = s


@main.command()
@click.option('-minrate', default=.005)
@click.option('-maxrate', default=.01)
@click.option('-idres', default=9)
@click.option('-fdres', default=10)
@click.option('-maxspec', default=.3)
@click.option('-wpow', default=.5)
@click.option('-rcopts',
              default='-ab 1 -ad 1024 -aa 0 -as 0 -lw 1e-5 -st 0 -ss 4')
@click.option('--view/--no-view', default=True)
@click.option('--ambient/--no-ambient', default=True)
@click.option('--reflection/--no-reflection', default=True)
@click.option('-apo', callback=clk.split_str)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sunrun(ctx, **kwargs):
    """run sunsampler"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    if 'suns' not in ctx.obj:
        invoke_suns(ctx)
    scn = ctx.obj['scene']
    sns = ctx.obj['suns']
    sampler = SunSampler(scn, sns, **kwargs)
    sampler.run(**kwargs)
    su = SunField(scn, sns, rebuild=True)
    su.direct_view()
    su.view.direct_view()


@main.command()
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def integrate(ctx, **kwargs):
    """build integrator and make images"""
    if 'scene' not in ctx.obj:
        invoke_scene(ctx)
    if 'suns' not in ctx.obj:
        invoke_suns(ctx)
    scn = ctx.obj['scene']
    sns = ctx.obj['suns']
    su = SunField(scn, sns)
    sk = SCBinField(scn)
    itg = Integrator(sk, su, stol=5)
    smtx, grnd, sun, si = itg.get_sky_mtx()
    # subset = np.array([1, 14, 32, 35])
    subset = np.array([1, 14])
    subset = np.arange(0, 103, 17)
    print(len(subset))
    # subset = np.arange(1000)
    # # itg.skyfield.direct_view()
    # # itg.sunfield.direct_view()
    # # itg.sunfield.view.direct_view()
    # print(scn.skydata[itg.dayhours][subset])
    itg.hdr([(5, 5, 1.25)], (0, -1, 0), smtx[subset], sun[subset], si[subset], interp=4, res=400)
    # itg.hdr([(5, 5, 1.25)], sun[14, 0:3], smtx[subset], sun[subset], si[subset], interp=1, res=800, vname='view2')
    # itg.hdr([(5, 5, 1.25)], [(0, 0, 1)], [1], [[0,]], [False,], interp=1, res=800)
    # illum = itg.illum([(5, 5, 1.25)], [(-1, 0, 0), sun[14, 0:3]], smtx[subset], sun[subset], si[subset])
    # print(illum)
    # print(illum.shape)


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
