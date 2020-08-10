#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse.scene import Scene, SkyInfo
from raytraverse import quickplot
import numpy as np
from clipt import mplt




@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/test/', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    yield data + '/test'
    os.chdir(cpath)


def test_scene(tmpdir):
    scene = Scene('results', 'test.oct', 'plane.rad')
    assert scene.scene == 'results/scene.oct'
    scene2 = Scene('results', 'MATERIAL/*.mat RAD/*.rad', 'plane.rad', overwrite=True, reload=False)
    assert scene2.scene == 'results/scene.oct'
    with pytest.raises(ChildProcessError):
        s = Scene('results', 'RAD/*.rad', 'plane.rad', overwrite=True)
        os.system(f'getinfo {s.scene}')
    with pytest.raises(FileExistsError):
        Scene('results', 'test.oct', 'plane.rad', reload=False)


def test_solarbounds(tmpdir):
    loc = (46.25, -6.13, -15)
    scene = SkyInfo(loc)
    assert np.all(np.logical_not(scene.in_solarbounds(np.array([[.5,.5], [1.5,1], [.5,0]]))))


def test_area(tmpdir):
    scene = Scene('results', 'test.oct', 'plane.rad', overwrite=True)
    # print(scene.area.bbox)
    # print(np.prod(scene.area.bbox[1, 0:2] - scene.area.bbox[0, 0:2]))
    assert scene.area.in_area(np.array([[.5, .5]]))
    assert not scene.area.in_area(np.array([[1.05, 1]]))
    scene = Scene('results', 'test.oct', 'plane.rad', overwrite=True, ptro=3)
    assert scene.area.in_area(np.array([[.5, .5]]))
    grid_u, grid_v = np.meshgrid(np.arange(.0005, 1, .001), np.arange(.0005, 1, .001))
    uv = np.vstack((grid_u.flatten(), grid_v.flatten())).T
    ia = scene.area.in_area(uv)
    area = np.prod(scene.area.bbox[1, 0:2] - scene.area.bbox[0, 0:2])*np.sum(ia)/uv.shape[0]
    assert np.isclose(area, 171, atol=.05)
    assert not scene.area.in_area(np.array([[1, 1]]))

