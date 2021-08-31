#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil
import pytest
import numpy as np

from raytraverse.mapper import ViewMapper
from raytraverse.utility import imagetools
from raytraverse.lightpoint import LightPointKD, CompressedPointKD
from raytraverse import translate, io
from raytraverse.scene import BaseScene, ImageScene
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


def make_checker(res, offset=0.5, src='checker'):
    bn = np.arange(res*res)
    uv = translate.bin2uv(bn, res, offset=offset)
    vm = ViewMapper(viewangle=180)
    xyz = vm.uv2xyz(uv)
    vals = np.mod(np.indices((res, res)).sum(axis=0), 2).ravel()
    vals = np.stack((vals, 1-vals)).T
    scene = BaseScene(None)
    return LightPointKD(scene, xyz, vals, write=False, vm=vm, srcn=2, src=src)


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
    energy_part = ['tasklum', 'backlum', 'srcillum', 'backlum_true', "srcarea", "maxlum"]
    constants = ['threshold']
    tree = ['view_area', 'density']
    contrast = ['gcr', 'ugp', 'ugr', 'dgp', 'log_gc', 'dgp_t2', 'pwsl2']
    allmets = energy + energy_part + contrast + tree + constants
    a1 = MetricSet(*checker.evaluate(2000/179), checker.vm)
    a2 = MetricSet(*checker.evaluate([3000/179, 1000/179]), checker.vm)
    a3 = MetricSet(*checker.evaluate([1000/179, 3000/179]), checker.vm)
    a4 = MetricSet(*checker.evaluate(0), checker.vm)
    assert np.alltrue(np.isclose(a4(energy + energy_part), 0))
    assert np.alltrue(np.isnan(a4(contrast[1:-1])))
    assert np.allclose(a2(allmets), a3(allmets))
    assert np.allclose(a2(energy + constants + tree), a1(energy + constants + tree))
    assert np.isclose(np.pi * 2, a1.view_area)
    assert np.isclose(a2.gcr, 1.25)
    assert np.isclose(a2.illum, np.pi * 2000, rtol=.01)
    assert np.isclose(a1.dgp, a2.dgp - a2.dgp_t2)


def test_add_sources():
    checker1 = make_checker(15, src='c1')
    checker2 = make_checker(45, src='c2')
    checker3 = checker1.add(checker2, write=False)
    a1 = MetricSet(*checker1.evaluate([3000/179, 0]), checker1.vm)
    a3 = MetricSet(*checker3.evaluate([3000/179, 0, 0, 0]), checker3.vm)
    assert np.isclose(a1.illum, a3.illum, rtol=.1)
    assert np.isclose(a1.avglum, a3.avglum, rtol=.01)


def test_compress(tmpdir):
    scene = ImageScene('snakesample')
    vm = ViewMapper(viewangle=180)
    lf = LightPointKD(scene, src='image', vm=vm)
    m1 = MetricSet(*lf.evaluate(1, vm), vm, scale=1000, threshold=500)
    soga1 = np.sum(m1.sources[1])
    slum1 = np.average(m1.sources[2], weights=m1.sources[1])
    boga1 = np.sum(m1.background[1])
    blum1 = np.average(m1.background[2], weights=m1.background[1])
    met1 = (soga1, slum1, boga1, blum1, m1.illum, m1.srcillum, m1.gcr)
    lf2 = CompressedPointKD(lf, dist=.2, lerr=.001, plotc=True)

    lf2.direct_view(512, showsample=False, interp=False, fisheye=True)
    vol = imagetools.hdr2vol("snakesample_image_compressed_000000.hdr")
    mi = MetricSet(*vol, vm, scale=1000, threshold=500)
    assert np.isclose(mi.illum, m1.illum, atol=1e-4, rtol=.001)

    m2 = MetricSet(*lf2.evaluate(1, vm), vm, scale=1000, threshold=500)
    soga2 = np.sum(m2.sources[1])
    slum2 = np.average(m2.sources[2], weights=m2.sources[1])
    boga2 = np.sum(m2.background[1])
    blum2 = np.average(m2.background[2], weights=m2.background[1])
    met2 = (soga2, slum2, boga2, blum2, m2.illum, m2.srcillum, m2.gcr)

    assert np.allclose(met1, met2, atol=1e-4, rtol=.01)
