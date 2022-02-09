#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.craytraverse"""
import os
import shutil

from raytraverse import api
from raytraverse.evaluate import MetricSet
from raytraverse.integrator import IntegratorDS, Integrator
from raytraverse.lightfield import ResultAxis
from raytraverse.mapper import ViewMapper
from raytraverse.sky import SkyData
from raytraverse.scene import Scene
import numpy as np
import pytest


import clasp.script_tools as cst


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/integrator/', data + '/test')
    cpath = os.getcwd()
    # use temp
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/integrator'
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_api(tmpdir):
    with pytest.raises(FileNotFoundError):
        scn, pm, skd = api.auto_reload("null", "plane.rad", ptres=1.2)
    with pytest.raises(FileNotFoundError):
        scn, pm, skd = api.auto_reload("result", "plane.rad", ptres=1.2)
    scn2 = Scene("result")
    skd2 = SkyData("geneva.epw")
    skd2.write(scene=scn2)
    scn, pm, skd = api.auto_reload("result", "plane.rad", ptres=1.2)
    assert np.alltrue(skd.smtx == skd2.smtx)


def check_itg(itg, skd):
    points = np.loadtxt("plane_0.6.pts")
    vm = [[0, 0, 1], [0, -1, 0]]
    # dxyz = np.asarray(vm).reshape(-1, 3)
    # vms = [ViewMapper(d, 180) for d in dxyz]
    vm, vms, metrics, sumsafe = itg._check_params(vm)
    tidxs, skydatas, dsns, vecs = itg._group_query(skd, points)
    oshape = (len(skd.maskindices), len(points), len(vms), len(metrics))

    sinfo, dinfo = itg._sinfo(True, vecs, tidxs, oshape[0:2])
    # compose axes: (skyaxis, ptaxis, viewaxis, metricaxis)
    axes = (ResultAxis(skd.maskindices, f"sky"),
            ResultAxis(points, "point"),
            ResultAxis([v.dxyz for v in vms], "view"),
            ResultAxis(list(metrics) + dinfo, "metric"))
    qtup, qidx, tup_isort = itg._sort_run_data(tidxs)

    _, cnts = np.unique(tidxs.T, axis=0, return_counts=True)
    r1 = np.arange(tidxs.shape[1])
    r2 = []
    for qt, qi, c in zip(qtup, qidx, cnts):
        mask = np.all(tidxs.T == qt, -1)
        assert np.sum(mask) == c
        r2 += list(r1[mask])
    assert np.allclose(r1, np.array(r2)[tup_isort])


def test_3comp(tmpdir):
    scn, pm, skd = api.auto_reload("result", "plane.rad", ptres=1.2)
    itg = api.get_integrator(scn, pm, simtype="3comp")
    assert type(itg) == IntegratorDS
    check_itg(itg, skd)


def test_2comp(tmpdir):
    scn, pm, skd = api.auto_reload("result", "plane.rad", ptres=1.2)
    itg = api.get_integrator(scn, pm, simtype="2comp")
    assert type(itg) == Integrator
    check_itg(itg, skd)

