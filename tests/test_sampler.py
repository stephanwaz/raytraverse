#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse.translate import scbinscal, scxyzcal
from raytraverse.scene import Scene
from raytraverse.sampler import SCBinSampler
from raytraverse import translate
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
    return Scene('results', 'test.oct', 'plane.rad', wea='geneva.epw')


def test_cals(tmpdir):
    f = open('d2b.cal', 'w')
    f.write(scbinscal)
    f.close()
    f = open('b2d.cal', 'w')
    f.write(scxyzcal)
    f.close()
    bins = np.arange(100)
    result = cst.pipeline(
        ["cnt 100 | rcalc -f rayinit.cal -f b2d.cal -e 'side=10;bin=$1"
         ";$1=$1;$2=U;$3=V;$4=Dx;$5=Dy;$6=Dz' | rcalc -f rayinit.cal -f "
         "d2b.cal -e 'side=10;Dx=$4;Dy=$5;Dz=$6;$1=bin'"])
    r = [float(i) for i in result.split()]
    assert np.allclose(r, bins)


def test_init(tmpdir, scene):
    res = np.array([[32, 16, ],
                    [64, 32, ],
                    [128, 64, ],
                    [256, 128, ],
                    [512, 256, ],
                    [1024, 512, ]])
    sampler = SCBinSampler(scene)
    assert np.alltrue(res == sampler.levels)


def test_sky_sample(tmpdir, scene, capfd):
    sampler = SCBinSampler(scene)
    vecf = sampler.dump_vecs(np.array([[5, 5, 1.25, 0, -1, 0]]))
    with capfd.disabled():
        lum = sampler.sample(vecf)
    assert lum > .05/179
