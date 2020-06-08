#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse import ViewMapper, translate
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
    assert np.allclose(translate.rotate(r2, (.45, .45, -.1), (0,0,1), (0,1,0)), translate.uv2xyz(vm.bbox))


def test_xyz2uv():
    grid_u, grid_v = np.mgrid[0:2:2/2048, 0:1:1/1024]
    uv = np.vstack((grid_u.flatten(), grid_v.flatten())).T + 1/2048
    xyz = translate.uv2xyz(uv, axes=(0, 2, 1))
    vm = ViewMapper( (.13,-.435,1), viewangle=90)
    inside = np.sum(np.prod(np.logical_and(vm.xyz2uv(xyz) > 0, vm.xyz2uv(xyz) < 1), 1))/xyz.shape[0]
    assert np.isclose(inside, 1/8, atol=1e-4)

