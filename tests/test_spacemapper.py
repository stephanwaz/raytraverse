#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse.mapper import SpaceMapper
import numpy as np

# from clipt import mplt


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/spacemapper/', data + '/test')
    cpath = os.getcwd()
    # use temp
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/spacemapper'
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_spacemapper(tmpdir):
    sm = SpaceMapper("rad", area='plane.rad', ptres=4, reload=False)
    assert np.all(np.equal(sm.pt2uv(sm.bbox), [[0, 0], [1, 1]]))
    sm.add_grid()
    points = np.loadtxt('rad/points.dat')
    sm2 = SpaceMapper("pts", points=points, ptres=4, reload=False)
    assert np.allclose(sm.points, sm2.points)
    sm2.ptres = 2
    sm2.add_grid()
    sm3 = SpaceMapper("pts", ptres=2)
    assert np.allclose(sm2.points, sm3.points)
    sm4 = SpaceMapper("pts2", mask=False, reload=False, rotation=43, points=np.array([[0, 0, 0], [1, 0, 0]]))
    sm4.add_points([[0, 4, 0], [0, 8, 0]])
    assert np.allclose(sm4.pt2uv(sm4.uv2pt([[0, 0], [1, 1]])), [[0, 0], [1, 1]])


