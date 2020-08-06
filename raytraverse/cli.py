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
from raytraverse import metric
from raytraverse.integrator import Integrator
from raytraverse.sampler import SCBinSampler, SunSampler
from raytraverse.scene import Scene, SunSetter, SunSetterLoc, SunSetterPositions
from raytraverse.lightfield import SCBinField, SunField, SunViewField


def np_load(ctx, param, s):
    """read np array from command line

    trys np.load (numpy binary), then np.loadtxt (space seperated txt file)
    then split row by spaces and columns by commas.
    """
    if os.path.exists(s):
        try:
            return np.load(s)
        except ValueError:
            return np.loadtxt(s)
    else:
        return np.array([[float(i) for i in j.split(',')] for j in s.split()])



@click.group(chain=True, invoke_without_command=True)
@click.argument('out')
@clk.shared_decs(clk.main_decs(raytraverse.__version__, writeconfig=False))
@click.option('--template/--no-template', is_eager=True,
              callback=clk.printconfigs,
              help="write default options to std out as config")
@click.option('--opts', '-opts', is_flag=True,
              help="check parsed options")
@click.option('--debug', is_flag=True,
              help="show traceback on exceptions")
def main(ctx, out, config, outconfig=None, **kwargs):
    """docstring"""
    clk.get_config_chained(ctx, config, outconfig, None, None)
    ctx.obj = dict(out=out)


@main.command()
@click.option('-scene', help='space separated list of radiance scene files '
              '(no sky) or octree')
@click.option('-area', help='radiance scene file containing planar geometry of '
              'analysis area')
@click.option('-skyres', default=10.0)
@click.option('-ptres', default=2.0)
@click.option('-maxspec', default=0.3)
@click.option('--reload/--no-reload', default=True)
@click.option('--overwrite/--no-overwrite', default=False)
@click.option('--info/--no-info', default=False, help='print info on scene')
@click.option('--points/--no-points', default=False,
              help='print point locations')
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def scene(ctx, **kwargs):
    """load scene"""
    s = Scene(ctx.obj['out'], **kwargs)
    ctx.obj['scene'] = s
    if kwargs['points']:
        for pt in s.pts():
            print('{}\t{}\t{}'.format(*pt))
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
        print(f'rotation: {s.ptro}')
        print(f'sky sampling resolution: {s.skyres}')


@main.command()
@click.option('-accuracy', default=1)
@click.option('-idres', default=4)
@click.option('-fdres', default=9)
@click.option('-rcopts',
              default='-ab 7 -ad 60000 -as 30000 -lw 1e-7 -st 0 -ss 16')
@click.option('--plotp/--no-plotp', default=False)
@click.option('--plotdview/--no-plotdview', default=False)
@click.option('--run/--no-run', default=True)
@click.option('--rmraw/--no-rmraw', default=True)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sky(ctx, plotdview=False, run=True, rmraw=True, **kwargs):
    """run scbinsampler"""
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene)
    sampler = SCBinSampler(ctx.obj['scene'], **kwargs)
    if run:
        sampler.run(rcopts=kwargs['rcopts'], executable='rcontrib')
    sk = SCBinField(ctx.obj['scene'], rebuild=run, rmraw=rmraw)
    if plotdview:
        sk.direct_view()


@main.command()
@click.option('-srct', default=.01)
@click.option('-skyro', default=0.0)
@click.option('-loc', callback=clk.split_float)
@click.option('-wea')
@click.option('--reload/--no-reload', default=True)
@click.option('--usepositions/--no-usepositions', default=False)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def suns(ctx, loc=None, wea=None, usepositions=False, **kwargs):
    """create sun positions"""
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene)
    if usepositions:
        if wea is None:
            raise ValueError('option -wea is required when use positions is '
                             'True')
        s = SunSetterPositions(ctx.obj['scene'], wea, **kwargs)
    elif loc is not None:
        s = SunSetterLoc(ctx.obj['scene'], loc, **kwargs)
    elif wea is not None:
        loc = raytraverse.skycalc.get_loc_epw(wea)
        s = SunSetterLoc(ctx.obj['scene'], loc, **kwargs)
    else:
        s = SunSetter(ctx.obj['scene'], **kwargs)
    s.direct_view()
    ctx.obj['suns'] = s


@main.command()
@click.option('-accuracy', default=1)
@click.option('-idres', default=4)
@click.option('-fdres', default=10)
@click.option('-speclevel', default=9)
@click.option('-rcopts',
              default='-ab 6 -ad 3000 -as 1500 -st 0 -ss 16 -aa .1')
@click.option('--view/--no-view', default=True)
@click.option('--ambcache/--no-ambcache', default=True)
@click.option('--keepamb/--no-keepamb', default=False)
@click.option('--reflection/--no-reflection', default=True)
@click.option('--plotp/--no-plotp', default=False)
@click.option('--plotdview/--no-plotdview', default=False)
@click.option('--run/--no-run', default=True)
@click.option('--rmraw/--no-rmraw', default=False)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def sunrun(ctx, plotdview=False, run=True, rmraw=False, **kwargs):
    """run sunsampler"""
    if 'suns' not in ctx.obj:
        clk.invoke_dependency(ctx, suns)
    scn = ctx.obj['scene']
    sns = ctx.obj['suns']
    sampler = SunSampler(scn, sns, **kwargs)
    if run:
        sampler.run(**kwargs)
    if kwargs['view']:
        sv = SunViewField(scn, sns, rebuild=run, rmraw=rmraw)
        if plotdview:
            sv.direct_view()
    if kwargs['reflection']:
        su = SunField(scn, sns, rebuild=run, rmraw=rmraw)
        if plotdview:
            su.direct_view(res=200)


@main.command()
@click.option('-loc', callback=clk.split_float)
@click.option('-wea')
@click.option('-pts', default='0,0,0', callback=np_load)
@click.option('-vdirs', default='0,-1,0', callback=np_load)
@click.option('-res', default=800)
@click.option('-interp', default=12)
@click.option('-vname', default='view')
@click.option('--skyonly/--no-skyonly', default=False)
@click.option('--hdr/--no-hdr', default=True)
@click.option('--illum/--no-illum', default=True)
@click.option('--cr/--no-cr', default=True)
@click.option('--rebuildsky/--no-rebuildsky', default=False)
@click.option('--rebuildsun/--no-rebuildsun', default=False)
@clk.shared_decs(clk.command_decs(raytraverse.__version__, wrap=True))
def integrate(ctx, pts=None, vdirs=None, skyonly=False, hdr=True,
              illum=True, cr=True, res=800, interp=12, vname='view',
              rebuildsky=False, rebuildsun=False, **kwargs):
    """build integrator and make images"""
    if 'scene' not in ctx.obj:
        clk.invoke_dependency(ctx, scene)
    scn = ctx.obj['scene']
    if not skyonly:
        if 'suns' not in ctx.obj:
            clk.invoke_dependency(ctx, suns)
        sns = ctx.obj['suns']
        su = SunField(scn, sns, rebuild=rebuildsun)
    else:
        su = None
    sk = SCBinField(scn, rebuild=rebuildsky)
    itg = Integrator(sk, su, **kwargs)
    skymtx = itg.get_sky_mtx()
    if hdr:
        for i, vd in enumerate(vdirs):
            vn = f'{vname}{i:02d}'
            itg.hdr(pts, vd, *skymtx, interp=interp, res=res, vname=vn)
    if illum:
        mf = (metric.illum, metric.sqlum)
        metrics = itg.metric(pts, vdirs, *skymtx, scale=179, metricfuncs=mf)
        # print("view\tpoint\tsky\t" + "\t".join([f.__name__ for f in mf]))
        for v, views in enumerate(metrics):
            for p, pts in enumerate(views):
                for s, skies in enumerate(pts):
                    print(f"{v}\t{p}\t{s}\t" + "\t".join([f"{i}" for i in skies]))
    if cr:
        itg.contrastratio(pts, vdirs, *skymtx, vname=vn)


@main.resultcallback()
@click.pass_context
def printconfig(ctx, returnvalue, **kwargs):
    """callback to save config file"""
    try:
        if ctx.command.commands['scene'].context_settings['default_map']['info']:
            s = ctx.obj['scene']
            print('\nCallback Scene Info:')
            print('='*60 + '\n')
            try:
                suncount = len(open(f'{s.outdir}/sun_modlist.txt').readlines())
                print(f'scene has {suncount} suns')
            except FileNotFoundError:
                print('sun setter not initialized')
            print('\n Lightfield Data:')
            print('-'*60)
            print('Has sky lightfield data:',
                  os.path.isfile(f'{s.outdir}/sky_kd_data.pickle'))
            print('Has sunview lightfield data:',
                  os.path.isfile(f'{s.outdir}/sunview_kd_data.pickle'))
    except KeyError:
        pass
    try:
        clk.tmp_clean(ctx)
    except Exception:
        pass


if __name__ == '__main__':
    main()
