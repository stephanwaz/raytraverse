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
    vm = ViewMapper()
    t = np.array(((0, 0), (2, 1)))
    assert np.allclose(vm.bbox, t)
    vm = ViewMapper((0, 1))
    t = np.array(((-0.5, 0), (1.5, 1)))
    assert np.allclose(vm.bbox, t)
    vm = ViewMapper((0, 1), 90)
    t = np.array(((.25, 0), (.75, 1)))
    assert np.allclose(vm.bbox, t)


def test_uv2view():
    vm = ViewMapper((1, 0), 90)
    r = vm.uv2view([[.5, .5], ])
    r2 = vm.uv2view([[0, 0], [1, 1]])
    assert np.allclose(r, vm.dxyz)
    assert np.allclose(r2, translate.uv2xyz(vm.bbox, axes=(0, 2, 1)))


def test_view2uv():
    vm = ViewMapper((1, 0), 90)
    r = vm.view2uv(np.array([[1, 0, 0], ]))
    assert np.allclose(r, [[.5, .5], ])


def test_bbox():
    vm = ViewMapper((1, 0), 90, 90)
    dtp = translate.tpnorm(translate.xyz2tp(vm.dxyz[None, :]))
    print(dtp)
    vh = vm._vh/2*np.pi/180
    vv = vm._vv/2*np.pi/180
    print(vh, vv)
    print(dtp + np.array([[vv, vh]]),
dtp + np.array([[-vv, vh]]),
dtp + np.array([[-vv, -vh]]),
dtp + np.array([[vv, -vh]]),
dtp + np.array([[0, -vh]]),
dtp + np.array([[0, vh]]),
dtp + np.array([[-vv, 0]]),
dtp + np.array([[vv, 0]]))
    corners = np.vstack((translate.tp2xyz(dtp + np.array([[vv, vh]])),
                         translate.tp2xyz(dtp + np.array([[-vv, vh]])),
                         translate.tp2xyz(dtp + np.array([[-vv, -vh]])),
                         translate.tp2xyz(dtp + np.array([[vv, -vh]])),
                         translate.tp2xyz(dtp + np.array([[0, -vh]])),
                         translate.tp2xyz(dtp + np.array([[0, vh]])),
                         translate.tp2xyz(dtp + np.array([[-vv, 0]])),
                         translate.tp2xyz(dtp + np.array([[vv, 0]]))))
    print(corners)
    cuv = translate.xyz2uv(corners)
    print(cuv)
    np.set_printoptions(precision=3, suppress=True)
    print(np.max(cuv, 0), np.min(cuv, 0))
