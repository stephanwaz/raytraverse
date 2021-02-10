#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.sunpos"""
import itertools
import os
import shutil
import shlex
from subprocess import Popen, PIPE

import pytest
from raytraverse import translate
from raytraverse.sky import skycalc
import clasp.script_tools as cst
import numpy as np


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/test/', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    yield data + '/test'
    os.chdir(cpath)
#
#
@pytest.fixture
def epw(tmpdir):
    return skycalc.read_epw('geneva.epw')


def test_row_2_datetime64(epw):
    print(epw.shape)
    times = skycalc.row_2_datetime64(epw[:, 0:3])
    assert times.size == 8760


def test_sunpos_utc(epw, tmpdir):
    lat, lon, mer = skycalc.get_loc_epw('geneva.epw')
    times = skycalc.row_2_datetime64(epw[0::27, 0:3])
    dt = skycalc.datetime64_2_datetime(times, mer)
    alt, az = skycalc.sunpos_utc(dt, lat, lon)
    aa = skycalc.sunpos_degrees(times, lat, lon, mer)
    assert np.allclose(alt.degrees, aa[:, 0])
    assert np.allclose(az.degrees, aa[:, 1] + 180)


def test_sunpos_ro(epw, tmpdir):
    lat, lon, mer = skycalc.get_loc_epw('geneva.epw')
    times = skycalc.row_2_datetime64(epw[0::27, 0:3])
    ro = -33
    aa0 = skycalc.sunpos_degrees(times, lat, lon, mer)
    aa1 = skycalc.sunpos_degrees(times, lat, lon, mer, ro=ro)
    aa2 = skycalc.sunpos_radians(times, lat, lon, mer, ro=ro*np.pi/180)
    xyz = skycalc.sunpos_xyz(times, lat, lon, mer, ro=ro)
    aa3 = np.pi/2 - translate.xyz2tp(xyz) - np.pi
    assert np.allclose(translate.tpnorm(aa2), translate.tpnorm(aa3))
    assert np.allclose(aa1[:, 1], aa0[:, 1] - ro)


coms = ['gendaylit -ang  025.7720 -097.5262 -W  0305.00  0169.00',
        'gendaylit -ang  015.4508  028.2398 -W  0006.00  0110.00',
        'gendaylit -ang  046.3428  066.1638 -W  0001.00  0173.00',
        'gendaylit -ang  023.1744 -096.6224 -W  0361.00  0143.00',
        'gendaylit -ang  037.5105  058.9071 -W  0772.00  0106.00',
        'gendaylit -ang  035.0428 -061.0204 -W  0009.00  0240.00',
        'gendaylit -ang  061.2242 -005.8188 -W  0636.00  0268.00',
        'gendaylit -ang  031.9013 -083.9051 -W  0108.00  0266.00',
        'gendaylit -ang  025.3829 -004.8149 -W  0071.00  0217.00',
        'gendaylit -ang  057.7150  054.6960 -W  0013.00  0379.00',
        'gendaylit -ang  028.2644  094.8719 -W  0164.00  0207.00',
        'gendaylit -ang  029.4139 -077.6655 -W  0006.00  0165.00',
        'gendaylit -ang  005.4847  068.0852 -W  0000.00  0011.00',
        'gendaylit -ang  017.1461 -033.6273 -W  0000.00  0068.00',
        'gendaylit -ang  036.9844 -042.8306 -W  0099.00  0265.00',
        'gendaylit -ang  002.4051 -095.1627 -W  0000.00  0013.00',
        'gendaylit -ang  027.5590 -039.2650 -W  0012.00  0196.00',
        'gendaylit -ang  009.9383 -055.4619 -W  0133.00  0052.00',
        'gendaylit -ang  055.6883  025.0152 -W  0005.00  0268.00',
        'gendaylit -ang  006.4689 -098.7054 -W  0000.00  0010.00',
        'gendaylit -ang  049.6190 -025.5670 -W  0648.00  0213.00',
        'gendaylit -ang  053.0355 -052.8380 -W  0856.00  0122.00',
        'gendaylit -ang  041.2106  040.6076 -W  0710.00  0155.00',
        'gendaylit -ang  025.7741 -097.3680 -W  0458.00  0143.00',
        'gendaylit -ang  021.1122 -013.4338 -W  0000.00  0087.00',
        'gendaylit -ang  059.0493 -029.5303 -W  0884.00  0128.00',
        'gendaylit -ang  049.1749 -001.1427 -W  0838.00  0127.00',
        'gendaylit -ang  003.9904 -114.3718 -W  0000.00  0008.00',
        'gendaylit -ang  067.0600 -004.8696 -W  0802.00  0177.00',
        'gendaylit -ang  056.6835 -028.1568 -W  0228.00  0392.00',
        'gendaylit -ang  028.4933  028.3746 -W  0000.00  0115.00',
        'gendaylit -ang  065.5123 -006.2799 -W  0805.00  0186.00',
        'gendaylit -ang  035.1922  078.2664 -W  0591.00  0154.00',
        'gendaylit -ang  003.1383  075.0774 -W  0000.00  0021.00',
        'gendaylit -ang  014.4518 -105.0055 -W  0000.00  0067.00',
        'gendaylit -ang  005.9489  092.7117 -W  0022.00  0031.00',
        'gendaylit -ang  022.1230  017.4593 -W  0000.00  0113.00',
        'gendaylit -ang  054.1227 -002.1801 -W  0017.00  0368.00',
        'gendaylit -ang  012.7378 -057.8102 -W  0000.00  0071.00',
        'gendaylit -ang  004.8257 -115.2646 -W  0000.00  0015.00',
        'gendaylit -ang  001.9823  074.2390 -W  0000.00  0009.00',
        'gendaylit -ang  040.6201 -004.9735 -W  0820.00  0108.00',
        'gendaylit -ang  053.7655  051.8910 -W  0009.00  0311.00',
        'gendaylit -ang  007.8679  115.6470 -W  0000.00  0020.00',
        'gendaylit -ang  041.2106  040.6076 -W  0710.00  0155.00']
dirdif = np.array([[float(i) for i in j.split()[-2:]] for j in coms])


@pytest.fixture
def check():
    at = []
    suni = []
    for i in coms:
        a = cst.pipeline([i, ])
        try:
            suni.append(float(a.split('solar')[-2].strip().split()[-1]))
        except IndexError:
            suni.append(0)
        at.append([float(j) for j in a.strip().split()[-10:]])
    return np.array(at), np.array(suni)


def test_perez(check):
    sxyz = check[0][:, -3:]
    result, suni = skycalc.perez(sxyz, dirdif)
    irerr = (result - check[0])/check[0]
    np.set_printoptions(4, suppress=True)
    print()
    print('Coefficients:')
    print('avg abs relative error:\n', np.average(np.abs(irerr), 0)[:-3])
    print('avg relative error:\n', np.average(irerr, 0)[:-3])
    print('max abs relative error:\n', np.max(np.abs(irerr), 0)[:-3])
    serr = (suni[check[1] > 0] - check[1][check[1] > 0])/check[1][check[1] > 0]
    print('sun error (min, med, max):\n', np.percentile(np.abs(serr), (0, 50, 100)))
    assert np.allclose(result, check[0], atol=0.001, rtol=.001)
    assert np.allclose(suni[check[1] > 0], check[1][check[1] > 0], atol=0.001, rtol=.001)


def call_generic(commands, n=1):
    pops = []
    stdin = None
    for c in commands:
        pops.append(Popen(shlex.split(c), stdin=stdin, stdout=PIPE))
        stdin = pops[-1].stdout
    a = stdin.read()
    return np.fromstring(a, sep=' ').reshape(-1, n)


def test_sky_mtx(check):
    sxyz = check[0][:, -3:]
    side = 20
    lum, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, side)
    smtx = np.load("smtx.npy")
    suni = sun[:, -1]
    assert np.allclose(lum[0, 0:5], smtx[0, 0:5], atol=0.001, rtol=.001)
    assert np.allclose(suni[check[1] > 0], check[1][check[1] > 0], atol=0.001,
                       rtol=.001)


def test_read_epw(epw):
    aepw = skycalc.read_epw_full("geneva.epw", [1, 2, 3, 14, 15])
    assert np.allclose(aepw, epw)


def test_perez_water(check):
    lat, lon, mer = skycalc.get_loc_epw('geneva.epw')
    epw = skycalc.read_epw_full("geneva.epw", [1, 2, 3, 7, 14, 15])
    times = skycalc.row_2_datetime64(epw[:, 0:3])
    dirdif2 = epw[:, 4:]
    tdp = epw[:, 3]
    sxyz = skycalc.sunpos_xyz(times, lat, lon, mer)
    # sxyz = check[0][:, -3:]
    r1, s1 = skycalc.perez(sxyz, dirdif2)
    r2, s2 = skycalc.perez(sxyz, dirdif2, td=tdp)
    avg = (np.hstack((r1[:, 0:2], s1[:, None])) +
           np.hstack((r2[:, 0:2], s2[:, None])))/2
    rerr = np.nan_to_num(np.hstack((r1[:, 0:2] - r2[:, 0:2], (s1 - s2)[:, None]))/avg)
    np.set_printoptions(4, suppress=True)
    nonzero = rerr.shape[0] - np.sum(rerr == 0, 0)
    assert np.allclose(np.sum(rerr, 0)/nonzero, [0.004, -0.0008, -0.0186], rtol=.01, atol=.01)
    assert np.allclose(np.sum(np.abs(rerr), 0)/nonzero, [0.01, 0.0033, 0.0315], rtol=.01, atol=.01)


def test_generate_wea(epw):
    ts = np.array(list(itertools.product([3], [12], np.arange(0, 24, .25))))
    ref = epw[np.all((epw[:, 0] == 3, epw[:, 1] == 12), 0)]
    out = skycalc.generate_wea(ts, 'geneva.epw', interp='cubic')
    assert np.allclose(np.sum(ref[:, 3:], 0), np.sum(out[:, 3:], 0)/4, atol=1)
