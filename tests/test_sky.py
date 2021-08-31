#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil
import pytest
import numpy as np

from raytraverse.sky import SkyData
from raytraverse.mapper import SkyMapper
from raytraverse import translate, io


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    subf = 'tests/sky/'
    shutil.copytree(subf, data + '/test')
    cpath = os.getcwd()
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/' + subf
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_suns(tmpdir):
    sm = SkyMapper((46.25, -6.13, -15), sunres=22)
    suns = sm.solar_grid(False, 1)
    sm.plot(suns, "test_sky_suns.hdr", res=512, grow=0)
    ref = io.hdr2array("ref_sky_suns.hdr")
    test = io.hdr2array("test_sky_suns.hdr")
    assert np.allclose(ref, test, atol=.0001)
    sm = SkyMapper((46.25, -6.13, -15), sunres=22)
    suns = sm.solar_grid(True, 1)
    np.savetxt('test_suns.txt', suns)
    sm = SkyMapper('test_suns.txt', sunres=22)
    suns2 = sm.solar_grid(False, 1)
    assert np.allclose(suns, suns2, atol=.0001)
    sm = SkyMapper(suns, sunres=22)
    suns2 = sm.solar_grid(False, 1)
    assert np.allclose(suns, suns2, atol=.0001)
    sm = SkyMapper("geneva.wea", sunres=22)
    suns2 = sm.solar_grid(False, 3)
    assert suns2.shape[0] == 509
    img, vecs, mask, mask2, header = sm.init_img()
    img[mask] = sm.in_solarbounds(vecs[mask], level=3)
    io.array2hdr(img, "test_geneva_mask.hdr")
    ref = io.hdr2array("ref_geneva_mask.hdr")
    test = io.hdr2array("test_geneva_mask.hdr")
    assert np.allclose(ref, test, atol=.0001)


def test_sunrotation(tmpdir):
    sm = SkyMapper((46.25, -6.13, -15), skyro=127, sunres=22)
    suns = sm.solar_grid(False, 1)
    sm.plot(suns, "test2_sky_suns.hdr", res=512, grow=0)
    ref = io.hdr2array("ref2_sky_suns.hdr")
    test = io.hdr2array("test2_sky_suns.hdr")
    assert np.allclose(ref, test, atol=.0001)
    sm = SkyMapper((46.25, -6.13, -15), skyro=127, sunres=22)
    suns = sm.solar_grid(True, 1)
    np.savetxt('test2_suns.txt', suns)
    sm = SkyMapper('test2_suns.txt', sunres=22)
    suns2 = sm.solar_grid(False, 1)
    assert np.allclose(suns, suns2, atol=.0001)
    sm = SkyMapper(translate.rotate_elem(suns, -127), skyro=127, sunres=22)
    suns2 = sm.solar_grid(False, 1)
    assert np.allclose(suns, suns2, atol=.0001)
    sm = SkyMapper("geneva.wea", skyro=127, sunres=22)
    suns2 = sm.solar_grid(False, 3)
    assert suns2.shape[0] == 521
    img, vecs, mask, mask2, header = sm.init_img()
    img[mask] = sm.in_solarbounds(vecs[mask], level=3)
    io.array2hdr(img, "test2_geneva_mask.hdr")
    ref = io.hdr2array("ref2_geneva_mask.hdr")
    test = io.hdr2array("test2_geneva_mask.hdr")
    assert np.allclose(ref, test, atol=.0001)


def test_skydata(tmpdir):
    loc = (46.25, -6.13, -15)
    skydat = SkyData("geneva.wea", loc=loc)
    hdr = "LOCATION= lat: {} lon: {} tz: {} ro: 0.0".format(*loc)
    assert hdr == skydat.header()
    wsun = skydat.smtx_patch_sun()
    assert wsun.shape == skydat.smtx.shape
    d = np.max(wsun - skydat.smtx, 1)
    assert np.allclose(d, skydat.sun[:, 4])
    # print(skydat.sun)
    # print(skydat.sunproxy.shape, np.percentile(skydat.sunproxy, (0, 100), 0))
    # print(skydat.smtx.shape, skydat.sunproxy.shape, skydat.suns.suns.shape)


# def test_skydata_mask(tmpdir):
#     loc = (46.25, -6.13, -15)
#     skydat = SkyData("geneva.wea", loc=loc)
