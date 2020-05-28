#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse import Scene
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
    # ax, fig = mplt.plot_setup()
    # j, d = scene.solarbounds
    # cmap = mplt.get_colors('viridis')
    # mplt.plot_scatter(fig, ax, [j[:,0], d[:,0]], [j[:,1], d[:,1]], [], cmap)
    # mplt.ticks(ax)
    # fig.set_size_inches(10,10)
    # mplt.plt.tight_layout()
    # mplt.plt.show()
    assert np.all(np.logical_not(scene.in_solarbounds(np.array([[.5,.5], [1.5,.2], [.5,0]]))))


def test_area(tmpdir):
    scene = Scene('test.oct', 'plane.rad', 'results', overwrite=True)
    assert scene.in_area(np.array([[.5, .5]]))
    assert not scene.in_area(np.array([[.85, .5]]))
