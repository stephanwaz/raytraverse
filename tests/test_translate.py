#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.translate"""

import pytest
from raytraverse import translate
import numpy as np


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


def test_arrray2uv():
    ar = np.zeros((1, 1, 10, 5, 10, 10))
    uv = translate.array2uv(ar, 2, 3)
    ari = translate.uv2ij(uv, (10, 5))
    uv2 = translate.bin2uv(np.arange(50), (10,5))
    assert np.allclose(uv, uv2)
    assert np.allclose(np.arange(50), translate.uv2bin(uv, (10, 5)))
    assert np.allclose(uv, ari/5 + .1)


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
