#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse.sampler import scbinscal, scxyzcal
from raytraverse import Scene, Sampler, translate
import numpy as np
import clasp.script_tools as cst

# from clipt import mplt


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/test/', data + '/test')
    shutil.copy('raytraverse/genskyvec.pl', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    yield data + '/test'
    os.chdir(cpath)


@pytest.fixture
def scene(tmpdir):
    return Scene('test.oct', 'plane.rad', 'results',
                 wea='geneva.epw', reload=True)


def test_cals(tmpdir):
    f = open('d2b.cal', 'w')
    f.write(scbinscal)
    f.close()
    f = open('b2d.cal', 'w')
    f.write(scxyzcal)
    f.close()
    bins = np.arange(100)
    result = cst.pipeline(["cnt 100 | rcalc -f rayinit.cal -f b2d.cal -e 'side=10;bin=$1"
              ";$1=$1;$2=U;$3=V;$4=Dx;$5=Dy;$6=Dz' | rcalc -f rayinit.cal -f "
              "d2b.cal -e 'side=10;Dx=$4;Dy=$5;Dz=$6;$1=bin'"])
    r = [float(i) for i in result.split()]
    assert np.allclose(r, bins)


def test_genskyvec(tmpdir):
    result = cst.pipeline(["gendaylit -ang 45 45 -W 900 100 | ./genskyvec.pl"
                           " -m 10 -sc -c 1 1 1 | getinfo - | rcalc "
                           "-e '$1=recno-2;cond=$1-1000'"])
    bins = np.array(result.split()).astype(int)
    si = np.stack(np.unravel_index(bins, (10, 10)))
    uv = si.T/10 + .5/10
    xyz = translate.uv2xyz(uv)
    tp = translate.xyz2tp(xyz)
    tp[:, 1] = 3*np.pi/2 - tp[:, 1]
    print(tp - np.array([np.pi/4, -np.pi/4]))


def test_init(tmpdir, scene):
    res = np.array([[3,    5,   32,   16,   20,   20],
                    [6,   10,   64,   32,   20,   20],
                    [12,   20,  128,   64,   20,   20],
                    [24,   40,  256,  128,   20,   20],
                    [24,   40,  512,  256,   20,   20],
                    [24,   40, 1024,  512,   20,   20]])
    sampler = Sampler(scene, ptres=.5)
    f = open(f'{sampler.scene.outdir}/scbins.cal')
    assert f.read() == scbinscal
    f.close()
    assert np.alltrue(res == sampler.levels)


def test_mkpmap(tmpdir, scene):
    sampler = Sampler(scene)
    sampler.mkpmap('glz sglz', nphotons=1e4)
    r = sampler.mkpmap('glz sglz', nphotons=1e4, overwrite=True)
    with pytest.raises(ChildProcessError):
        sampler.mkpmap('glz sglz', nphotons=1e4)
    assert os.path.isfile('results/sky.gpm')


def test_sky_sample(tmpdir, scene):
    sampler = Sampler(scene)
    lum = sampler.sky_sample(np.array([[5, 5, 1.25, 0, -1, 0]]))
    assert lum.size == 400 and np.sum(lum) > .05/179


def test_sample_idx(tmpdir, scene):
    sampler = Sampler(scene)
    si, vecs = sampler.sample_idx(np.arange(7680), upsample=False)
    assert vecs.shape[0] == 7680
    l1 = np.prod(sampler.levels[1, 0:4])
    sampler.idx = 2
    si, vecs = sampler.sample_idx(np.random.choice(np.arange(l1), 10))
    assert vecs.shape[0] == 10*np.prod(np.prod(sampler.levels[2, 0:4]))/np.prod(l1)
#
