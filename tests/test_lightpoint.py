#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil
import pytest
import numpy as np

from raytraverse.mapper import ViewMapper
from raytraverse.lightpoint import LightPointKD
from raytraverse import translate, io
from raytraverse.scene import BaseScene
from raytraverse.evaluate import MetricSet


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/lightpoint/', data + '/test')
    cpath = os.getcwd()
    # use temp
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/lightpoint'
    os.chdir(path)
    yield path
    os.chdir(cpath)


def make_checker(res, offset=0.5):
    bn = np.arange(res*res)
    uv = translate.bin2uv(bn, res, offset=offset)
    vm = ViewMapper(viewangle=180)
    xyz = vm.uv2xyz(uv)
    vals = np.mod(np.indices((res, res)).sum(axis=0), 2).ravel()
    vals = np.stack((vals, 1-vals)).T
    scene = BaseScene(None)
    return LightPointKD(scene, xyz, vals, write=False, vm=vm, srcn=2)


def test_img(tmpdir):
    checker = make_checker(10)
    outf = "checkers.hdr"
    img, vec, mask, _, hdr = checker.vm.init_img()
    checker.add_to_img(img, vec[mask], mask, [1, 0], interp=False)
    io.array2hdr(img, outf)
    ref = io.hdr2array("checkers_ref.hdr")
    assert np.allclose(img, ref, atol=.005)


def test_metric():
    checker = make_checker(100)
    energy = ['illum', 'avglum', 'dgp_t1', 'avgraylum']
    energy_part = ['tasklum', 'backlum', 'srcillum', 'backlum_true']
    constants = ['threshold']
    tree = ['view_area', 'density', 'reldensity']
    contrast = ['gcr', 'ugp', 'ugr', 'dgp', 'log_gc', 'dgp_t2', 'ugr', 'pwsl2']
    allmets = energy + energy_part + contrast + tree + constants
    assert len(allmets) == len(MetricSet.allmetrics)
    a1 = MetricSet(checker.vm, *checker.get_applied_rays(2000/179))
    a2 = MetricSet(checker.vm, *checker.get_applied_rays([3000/179, 1000/179]))
    a3 = MetricSet(checker.vm, *checker.get_applied_rays([1000/179, 3000/179]))
    a4 = MetricSet(checker.vm, *checker.get_applied_rays(0))
    assert np.alltrue(np.isclose(a4(energy + energy_part), 0))
    assert np.alltrue(np.isnan(a4(contrast[:-1])))
    assert np.allclose(a2(allmets), a3(allmets))
    assert np.allclose(a2(energy + constants + tree), a1(energy + constants + tree))
    assert np.isclose(np.pi * 2, a1.view_area)
    assert np.isclose(a2.gcr, 1.25)
    assert np.isclose(a2.illum, np.pi * 2000, rtol=.01)
    assert np.isclose(a1.dgp, a2.dgp - a2.dgp_t2)


