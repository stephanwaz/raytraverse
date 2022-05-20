#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
import numpy as np

from raytraverse.renderer import Rtrace, Rcontrib
from raytraverse.scene import Scene
from raytraverse.sampler import Sensor, ISamplerArea, ISamplerSuns, SamplerSuns
from raytraverse.mapper import PlanMapper, SkyMapper


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/samplers/', data + '/test')
    cpath = os.getcwd()
    # use temp
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/samplers'
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_rtrace_sensor(tmpdir):
    scene = Scene('skysample', "box.rad", frozen=False)
    rtrace = Rtrace("-I+ -ab 1 -aa 0 -ad 10000 -lw 1e-6 -u+", scene=scene.scene)
    rtrace.load_source("sky.rad")
    sensor = Sensor(rtrace, ((0, -1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)))
    pts = ((1.5, 1.5, 1.5), (1.5, 1.5, 1.5), (1.5, 1.5, 1.5))
    r = sensor(pts)
    assert np.all(r[:, 0] == 0)
    assert np.allclose(r[:, 1], 1.74, atol=1e-2)
    assert np.allclose(r[:, 2:], 0.35, atol=1e-2)
    rtrace.reset()


def test_rcontrib_sensor(tmpdir):
    scene = Scene('skysample', "box.rad", frozen=False)
    rcontrib = Rcontrib("-I+ -ab 1 -aa 0 -ad 10000 -lw 1e-6 -u+",
                        scene=scene.scene, skyres=2)
    sensor = Sensor(rcontrib, ((0, -1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)))
    pts = ((1.5, 1.5, 1.5), (1.5, 1.5, 1.5), (1.5, 1.5, 1.5))
    r = sensor(pts)
    r = np.sum(r, axis=2)
    assert np.all(r[:, 0] == 0)
    assert np.allclose(r[:, 1], 1.74, atol=1e-2)
    assert np.allclose(r[:, 2:], 0.35, atol=1e-2)
    rcontrib.reset()


def test_isamplerarea(tmpdir):
    sm = PlanMapper('plane.rad', ptres=.25)
    scene = Scene('skysample', "box.rad", frozen=False)
    # return color
    rtrace = Rtrace("-I+ -ab 1 -aa 0 -ad 10000 -lw 1e-6 -u+ -ov",
                    scene=scene.scene)
    rtrace.load_source("sky.rad")
    sensor = Sensor(rtrace, ((0, -1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)))
    sampler = ISamplerArea(scene, sensor)
    sampler.run(sm)
    rtrace.reset()


def test_isamplerarearc(tmpdir):
    sm = PlanMapper('plane.rad', ptres=.25)
    scene = Scene('skysample', "box.rad", frozen=False)
    rcontrib = Rcontrib("-I+ -ab 1 -aa 0 -ad 10000 -lw 1e-6 -u+",
                        scene=scene.scene)
    sensor = Sensor(rcontrib, ((0, -1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)))
    sampler = ISamplerArea(scene, sensor, weightfunc=np.sum, stype='sky')
    sampler.run(sm)
    rcontrib.reset()


def test_isunsampler(tmpdir):
    sm = SkyMapper((-30, 0, 0), sunres=9)
    scene = Scene('skysample', "box.rad", frozen=False)
    rtrace = Rtrace("-I+ -ab 0", scene=scene.scene)
    pm = PlanMapper('plane.rad', ptres=.25)
    sensor = Sensor(rtrace, ((0, -1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)))
    sampler = ISamplerSuns(scene, sensor, nlev=2)
    sampler.run(sm, pm)


# def test_sunsampler(tmpdir):
#     sm = SkyMapper((-30, 0, 0), sunres=8)
#     scene = Scene('ssample', "box.rad", frozen=False)
#     rtrace = Rtrace("-I+ -ab 0", scene=scene.scene)
#     pm = PlanMapper('plane.rad', ptres=1)
#     sampler = SamplerSuns(scene, rtrace, nlev=2, areakwargs=dict(nlev=1))
#     sampler.run(sm, pm, recover=False)
