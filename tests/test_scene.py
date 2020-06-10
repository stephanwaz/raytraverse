#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse import Scene
import numpy as np




@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/test/', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    yield data + '/test'
    os.chdir(cpath)


def test_scene(tmpdir):
    scene = Scene('test.oct', 'plane.rad', 'results')
    assert scene.scene == 'results/scene.oct'
    scene2 = Scene('MATERIAL/*.mat RAD/*.rad', 'plane.rad', 'results', overwrite=True)
    assert scene2.scene == 'results/scene.oct'
    with pytest.raises(ChildProcessError):
        Scene('RAD/*.rad', 'plane.rad', 'results', overwrite=True)
    with pytest.raises(FileExistsError):
        Scene('test.oct', 'plane.rad', 'results')
    with pytest.raises(ValueError):
        Scene('test.oct', 'plane.rad', 'results2', weaformat='dfd')


def test_skydat(tmpdir):
    loc = (46.25, -6.13, -15)
    scene = Scene('test.oct', 'plane.rad', 'results', wea='geneva.epw', overwrite=True)
    assert scene.skydata.shape == (8760, 4)
    scene2 = Scene('test.oct', 'plane.rad', 'results2', wea='results/skydat.txt',
                  overwrite=True, weaformat='angle')
    assert np.allclose(scene.skydata, scene2.skydata)
    scene3 = Scene('test.oct', 'plane.rad', 'results', wea='geneva_nohead.wea',
                  overwrite=True, loc=loc)
    assert np.allclose(scene3.skydata, scene2.skydata)


def test_solarbounds(tmpdir):
    loc = (46.25, -6.13, -15)
    scene = Scene('test.oct', 'plane.rad', 'results', loc=loc, overwrite=True)
    assert np.all(np.logical_not(scene.in_solarbounds(np.array([[.5,.5], [1.5,.2], [.5,0]]))))


def test_area(tmpdir):
    scene = Scene('test.oct', 'plane.rad', 'results', overwrite=True)
    # print(scene.area.bbox)
    # print(np.prod(scene.area.bbox[1, 0:2] - scene.area.bbox[0, 0:2]))
    assert scene.in_area(np.array([[.5, .5]]))
    assert not scene.in_area(np.array([[1.05, 1]]))
    scene = Scene('test.oct', 'plane.rad', 'results', overwrite=True, ptro=3)
    assert scene.in_area(np.array([[.5, .5]]))
    grid_u, grid_v = np.meshgrid(np.arange(.0005, 1, .001), np.arange(.0005, 1, .001))
    uv = np.vstack((grid_u.flatten(), grid_v.flatten())).T
    ia = scene.in_area(uv)
    area = np.prod(scene.area.bbox[1, 0:2] - scene.area.bbox[0, 0:2])*np.sum(ia)/uv.shape[0]
    assert np.isclose(area, 171, atol=.05)
    assert not scene.in_area(np.array([[1, 1]]))


def test_reload(tmpdir):
    scene = Scene('test.oct', 'plane.rad', 'results', reload=True)
    assert scene.skydata is None
    scene2 = Scene('test.oct', 'plane.rad', 'results2', reload=True)
    os.system('ls results/*')
    assert scene2.skydata.shape == (8760,4)
