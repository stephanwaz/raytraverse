#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.sunpos"""
import os
import shutil

import pytest
from raytraverse import sunpos, translate
import numpy as np


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/test/', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    yield data + '/test'
    os.chdir(cpath)


@pytest.fixture
def epw(tmpdir):
    return sunpos.read_epw('geneva.epw')


def test_row_2_datetime64(epw):
    times = sunpos.row_2_datetime64(epw[:, 0:3])
    assert times.size == 8760


def test_sunpos_utc(epw, tmpdir):
    lat, lon, mer = sunpos.get_loc_epw('geneva.epw')
    times = sunpos.row_2_datetime64(epw[:, 0:3])
    dt = sunpos.datetime64_2_datetime(times, mer)
    alt, az = sunpos.sunpos_utc(dt, lat, lon)
    aa = sunpos.sunpos_degrees(times, lat, lon, mer)
    assert np.allclose(alt.degrees, aa[:,0])
    assert np.allclose(az.degrees, aa[:,1] + 180)


def test_sunpos_ro(epw, tmpdir):
    lat, lon, mer = sunpos.get_loc_epw('geneva.epw')
    times = sunpos.row_2_datetime64(epw[:, 0:3])
    ro = -33
    aa0 = sunpos.sunpos_degrees(times, lat, lon, mer)
    aa1 = sunpos.sunpos_degrees(times, lat, lon, mer, ro=ro)
    aa2 = sunpos.sunpos_radians(times, lat, lon, mer, ro=ro*np.pi/180)
    xyz = sunpos.sunpos_xyz(times, lat, lon, mer, ro=ro)
    aa3 = np.pi/2 - translate.xyz2tp(xyz) - np.pi
    assert np.allclose(translate.tpnorm(aa2), translate.tpnorm(aa3))
    assert np.allclose(aa1[:, 1], aa0[:, 1] - ro)
