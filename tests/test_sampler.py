#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse import Scene, Sampler
import numpy as np

# from clipt import mplt


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/test/', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    yield data + '/test'
    os.chdir(cpath)


@pytest.fixture
def scene(tmpdir):
    return Scene('test.oct', 'plane.rad', 'results',
                 wea='geneva.epw', reload=True)


def test_init(tmpdir, scene):
    res = np.array([[3,    5,   32,   16,   20,   20],
                    [6,   10,   64,   32,   20,   20],
                    [12,   20,  128,   64,   20,   20],
                    [24,   40,  256,  128,   20,   20],
                    [24,   40,  512,  256,   20,   20],
                    [24,   40, 1024,  512,   20,   20]])
    sampler = Sampler(scene, ptres=.5)
    assert np.alltrue(res == sampler.levels)


def test_mkpmap(tmpdir, scene):
    sampler = Sampler(scene)
    sampler.mkpmap('glz sglz', nphotons=1e4)
    with pytest.raises(ChildProcessError):
        sampler.mkpmap('glz sglz', nphotons=1e4)
    assert os.path.isfile('results/sky.gpm')


