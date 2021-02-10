#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil
from matplotlib.testing.compare import compare_images
import pytest
import numpy as np

from raytraverse.sky import Suns, SunsLoc, SunsPos, SkyData
from raytraverse import translate


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/sky/', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    yield data + '/test'
    os.chdir(cpath)


def test_suns(tmpdir):
    sunsa = SunsLoc("testsky", (46.25, -6.13, -15))
    sunsa.direct_view()
    assert compare_images("testsky_ref.png", "testsky_suns.png", .01) is None
    sunsb = SunsPos("testsky2", "testsky/suns.dat", reload=False)
    assert np.allclose(sunsa.suns, sunsb.suns)
    suns = SunsPos("testsky3", "geneva.wea", reload=False, skyro=30)
    assert suns.suns.shape[0] == 129
    suns = Suns("testsky4", reload=False)
    assert suns.suns.shape == (suns.sunres**2, 3)
    suns = SunsLoc("testsky5", (46.25, -6.13, -15), reload=False, skyro=30)
    assert suns.suns.shape[0] == 127


def test_suncheck(tmpdir):
    suns = Suns("testsky4", reload=False)
    s1 = suns.suns
    suns = Suns("testsky4", reload=False)
    sbins, err = suns.proxy_src(s1)
    assert np.average(err) < 5
    sbins, err = suns.proxy_src(s1, tol=1)
    hasmatch = sbins < suns.sunres**2
    assert np.average(err[hasmatch]) < 1
    assert 52 > np.sum(hasmatch) > 20


def test_skydata(tmpdir):
    loc = (46.25, -6.13, -15)
    suns = SunsLoc("testsky", loc)
    skydat = SkyData("geneva.wea", suns, loc=loc)
    hasmatch = skydat.sunproxy[:, 1] < suns.sunres**2
    assert np.sum(hasmatch) == skydat.sunproxy[:, 1].size
    hdr = "LOCATION= lat: {} lon: {} tz: {} ro: 0.0".format(*loc)
    assert hdr == skydat.header()
    wsun = skydat.smtx_patch_sun()
    assert wsun.shape == skydat.smtx.shape
    d = np.max(wsun - skydat.smtx, 1)
    assert np.allclose(d, skydat.sun[:, 4])
    # print(skydat.sun)
    # print(skydat.sunproxy.shape, np.percentile(skydat.sunproxy, (0, 100), 0))
    # print(skydat.smtx.shape, skydat.sunproxy.shape, skydat.suns.suns.shape)

