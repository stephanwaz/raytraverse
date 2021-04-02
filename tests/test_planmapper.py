#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse.mapper import PlanMapper, PointSet
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


def test_planmapper(tmpdir):
    sm = PlanMapper('plane.rad', ptres=4)
    bbox = np.hstack([sm.bbox, [[0], [0]]])
    assert np.all(np.equal(sm.xyz2uv(bbox), [[0, 0], [1, 1]]))

    ps = PointSet(sm, fill=True, jitter=False)
    # np.savetxt("grid.txt", ps.points)
    psa = PointSet(sm, points='grid.txt')
    assert np.allclose(ps.points, psa.points)

    sm2 = PlanMapper('plane43.rad', ptres=4, rotation=43)
    ps2 = PointSet(sm2, fill=True, jitter=False)
    assert np.allclose(ps.points, sm2.world2view(ps2.points), atol=1e-3)
