#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.translate"""

import pytest
from raytraverse import translate
from raytraverse.mapper import ViewMapper
import numpy as np
from clipt import mplt

import clasp.script_tools as cst


@pytest.fixture
def thetas():
    return fibset(4000)


def fibset(n=1000):
    dphi = np.pi*(3-5**.5)
    phi = 0
    dz = 2/n
    z = 1 - dz/2
    thetas = np.zeros((n,2))
    for j in range(n):
        thetas[j][0] = np.arcsin(z)
        thetas[j][1] = np.mod(phi, 2*np.pi)
        z = z - dz
        phi = phi + dphi
    return thetas


def test_norm(thetas):
    xyz = translate.tp2xyz(thetas)
    assert np.allclose(translate.norm(xyz), xyz)
    rxyz = np.random.random((1000, 3))
    ln = rxyz/np.linalg.norm(rxyz, axis=-1)[..., None]
    tn = translate.norm(rxyz)
    assert np.allclose(ln, tn)


def test_uv2xy():
    uv = (np.stack(np.mgrid[0:100, 0:100], 2).reshape(-1, 2) + .5)/100
    xy1 = translate.uv2xy(uv)
    r1 = np.linalg.norm(xy1, axis=1)
    z = 1 - r1*r1
    spterm = np.sqrt(2 - r1*r1)
    xyz1 = np.hstack((xy1 * spterm[:, None], z[:, None]))
    uv1 = translate.xyz2uv(xyz1, flipu=False)
    assert np.allclose(uv1, uv, atol=1e-7)


def test_xyz2uv(thetas):
    xyz = translate.tp2xyz(thetas)
    uv = translate.xyz2uv(xyz, normalize=True)
    xyz2 = translate.uv2xyz(uv)
    for a, b in zip(xyz, xyz2):
        assert np.allclose(a, b)


def test_tpnorm():
    a = np.linspace(-np.pi/2, np.pi/2, 100000)
    b = np.linspace(-np.pi, np.pi, 100000)
    thetas = translate.tpnorm(np.vstack((a,b)).T)
    assert np.allclose(np.max(thetas, 0), np.array([np.pi, 2*np.pi]), atol=1e-4)
    assert np.allclose(np.min(thetas, 0), np.array([0, 0]), atol=1e-4)


def test_tp2xyz(thetas):
    thetas = translate.tpnorm(thetas)
    xyz = translate.tp2xyz(thetas)
    theta2 = translate.xyz2tp(xyz)
    assert np.allclose(thetas, theta2)


def test_chord():
    x = np.linspace(0, 1, 200)
    theta = x*np.pi
    c = translate.theta2chord(theta)
    theta2 = translate.chord2theta(c)
    assert np.allclose(theta, theta2)


def test_rmtx_world2std():
    for z in (-1, 0, 1):
        for a, b in zip([1, 1, 0, -1, -1, -1, 0, 1], [0, 1, 1, 1, 0, -1, -1, -1]):
            np.set_printoptions(3, suppress=True)
            v = translate.norm1([a, b, z])
            ymtx, pmtx = translate.rmtx_yp(v)
            t = v.reshape(3, -1)
            t2 = (pmtx@(ymtx@t))
            t3 = (ymtx.T@(pmtx.T@t2)).T[0]
            assert np.allclose((0,0,1), t2.T[0])
            assert np.allclose(v, t3)
        v = translate.norm1([0, 0, z])
        if z == 0:
            with pytest.raises(ValueError):
                ymtx, pmtx = translate.rmtx_yp(v)
        else:
            ymtx, pmtx = translate.rmtx_yp(v)
            t = v.reshape(3, -1)
            assert np.allclose((0, 0, 1), (pmtx@(ymtx@t)).T[0])


def test_bin2uv():
    bins = np.arange(200)
    ij = np.stack(np.unravel_index(bins, (20, 10))).T
    uv = translate.bin2uv(bins, 10)
    ij2 = translate.uv2ij(uv, 10)
    assert np.allclose(ij, ij2)
    bins2 = translate.uv2bin(uv, 10)
    bins3 = translate.uv2bin(uv + .999/10, 10)
    assert np.allclose(bins, bins2)
    assert np.allclose(bins3, bins2)


def test_view2xyz():
    dxyz = translate.norm1((0, 1, 0.))
    va = 180
    res = 200
    v = ViewMapper(dxyz=dxyz, viewangle=va)
    vwrays = 'vwrays -vta -vv {} -vh {} -vd {} {} {} -vu 0 0 1 -x {} -y {}'.format(va, va, *dxyz, res, res)
    vrays = np.fromstring(cst.pipeline([vwrays]), sep=' ').reshape(res, res, 6)[-1::-1, -1::-1, 3:]
    xyz = v.pixelrays(res)
    xyz = np.swapaxes(xyz, 0, 1)
    assert np.allclose(xyz, vrays, rtol=.01, atol=.01)
