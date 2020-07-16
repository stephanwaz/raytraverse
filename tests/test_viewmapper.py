#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse import translate, io
from raytraverse.mapper import ViewMapper
import numpy as np


# from clipt import mplt

def test_viewmapper():
    vm = ViewMapper(viewangle=90)
    t = np.array(((0.25, .25), (.75, .75)))
    assert np.allclose(vm.bbox, t)
    vm = ViewMapper()
    t = np.array(((0, 0), (2, 1)))
    assert np.allclose(vm.bbox, t)


def test_uv2xyz():
    vm = ViewMapper((.45, .45, -.1), viewangle=44)
    r = vm.uv2xyz([[.5, .5], ])
    r2 = vm.uv2xyz([[0, 0], [1, 1]])
    assert np.allclose(r, vm.dxyz)


def test_xyz2uv():
    grid_u, grid_v = np.mgrid[0:2:2/2048, 0:1:1/1024]
    uv = np.vstack((grid_u.flatten(), grid_v.flatten())).T + 1/2048
    xyz = translate.uv2xyz(uv, axes=(0, 2, 1))
    vm = ViewMapper( (.13,-.435,1), viewangle=90)
    inside = np.sum(np.prod(np.logical_and(vm.xyz2uv(xyz) > 0, vm.xyz2uv(xyz) < 1), 1))/xyz.shape[0]
    assert np.isclose(inside, 1/8, atol=1e-4)


def test_radians():
    vm = ViewMapper(viewangle=180)
    vec = translate.norm([[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
                          [-1, 0, 0], [-1, -1, 0], [0, -1, 0], [1, -1, 0]])
    deg = np.array([90, 45, 0, 45, 90, 135, 180, 135])
    rad = deg * np.pi/180
    ans = vm.radians(vec)
    ans2 = vm.degrees(vec)
    assert np.allclose(ans, rad)
    assert np.allclose(ans2, deg)


def test_omega():
    res = 800
    va = 180
    vm = ViewMapper(viewangle=va)
    pxy = (np.stack(np.mgrid[0:res, 0:res]).T + .5)
    xyz, mask = vm.pixel2ray(pxy, res)
    omega = vm.pixel2omega(pxy, res)
    exp = np.pi*2*(1-np.cos(va*np.pi/360))
    ans = np.sum(omega[mask])
    assert np.isclose(ans, exp, rtol=1e-4)
