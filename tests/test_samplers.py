#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil
import sys

import pytest
import numpy as np

from raytraverse import io, translate
from raytraverse.evaluate import MetricSet
from raytraverse.renderer import Rtrace, Rcontrib
from raytraverse.scene import Scene
from raytraverse.sampler import SkySampler, SunSampler, SunViewSampler
from raytraverse.mapper import ViewMapper


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/samplers/', data + '/test')
    cpath = os.getcwd()
    # use temp
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/samplers'
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_skysample(tmpdir):

    def img_illum(lf0, vm0, sbin):
        outf = lf0.direct_view(res=512, srcidx=sbin, showsample=False,
                               scalefactor=1)
        hdr = io.hdr2array(outf)
        vecs = vm0.pixelrays(512)
        pxs = vm0.pixels(512).reshape(-1, 2)
        omega = vm0.pixel2omega(pxs, 512)
        cost = vm0.ctheta(vecs)

        coefs = np.zeros(325)
        coefs[sbin] = 1
        illumm0 = MetricSet(vm, *lf.get_applied_rays(coefs, vm), ["illum"])()[0]

        return np.sum(hdr.flatten()*omega*cost)*179, illumm0

    scene = Scene('skysample', "box.rad", frozen=False)
    rcontrib = Rcontrib('-ab 1 -ad 10000 -c 1 -lw 1e-5', scene.scene)
    sampler = SkySampler(scene, rcontrib, fdres=7)
    vm = ViewMapper((0, 1, 0), viewangle=180)
    lf = sampler.run((1.5, 1.5, 1.5), 0, vm)
    illum, illumm = img_illum(lf, vm, 176)
    assert np.isclose(illum, illumm, atol=.2, rtol=.1)
    illum2, illumm2 = img_illum(lf, vm, 158)
    assert np.isclose(illum2, illumm2, atol=.2, rtol=.1)
    assert np.isclose(np.average([illum, illumm, illum2, illumm2]), 2.88, atol=.1, rtol=.3)
    illum, illumm = img_illum(lf, vm, 174)
    assert np.isclose(illum, illumm, atol=.2, rtol=.1)
    assert np.isclose(np.average([illum, illumm]), 0.169, atol=.03, rtol=.2)
    fmetric = MetricSet(vm, *lf.get_applied_rays(np.ones(325), vm), ["illum", "density"])()
    assert np.isclose(fmetric[0], 352.7, atol=1, rtol=.01)
    assert np.isclose(fmetric[1], 612, atol=30)
    Rcontrib.reset()
    Rcontrib._pyinstance = None


def test_sunsample(tmpdir):

    def img_illum(lf0, vm0):
        outf = lf0.direct_view(res=512, showsample=False, scalefactor=285.32)
        hdr = io.hdr2array(outf)
        vecs = vm0.pixelrays(512)
        pxs = vm0.pixels(512).reshape(-1, 2)
        omega = vm0.pixel2omega(pxs, 512)
        cost = vm0.ctheta(vecs)
        illumm0 = MetricSet(vm, *lf.get_applied_rays(285.32, vm), ["illum"])()[0]

        return np.sum(hdr.flatten()*omega*cost)*179, illumm0

    scene = Scene('skysample', "box.rad", frozen=False)
    sun = translate.skybin2xyz([174], 18)[0]
    rtrace = Rtrace(scene=scene.scene, direct=True)
    sampler = SunSampler(scene, rtrace, sun, 174)
    vm = ViewMapper((0, 1, 0), viewangle=180)
    lf = sampler.run((1.5, 1.5, 1.5), 0, vm)
    illum, illumm = img_illum(lf, vm)
    assert np.isclose(illum, illumm, atol=.01, rtol=.05)
    assert np.isclose(np.average([illum, illumm]), 0.169, atol=.01, rtol=.01)
    sampler.engine.reset()


def test_sunviewsample(tmpdir):
    scene = Scene('skysample', "box.rad", frozen=False)
    sun = translate.skybin2xyz([176], 18)[0]
    rtrace = Rtrace(scene=scene.scene, direct=True)
    sampler = SunViewSampler(scene, rtrace, sun, 176)
    lf = sampler.run((1.5, 1.5, 1.79), 0, plotp=False)
    lf2 = sampler.run((1.5, 1.5, 1.5), 1, plotp=False)
    assert np.allclose([lf.lum, lf2.lum], 1.0)
    assert lf.vec[2] < lf2.vec[2]
    assert np.isclose(lf2.omega, 2*np.pi*(1 - np.cos(0.533*np.pi/360)))
    assert np.isclose(lf.omega/lf2.omega, 0.566650390625, atol=.0001 )
    for res in [5, 10, 50]:
        outf2 = lf.direct_view(res=res)
        sun2 = io.hdr2array(outf2)
        vm = ViewMapper(viewangle=.666)
        pxy = vm.pixels(sun2.shape[0])
        omega = vm.pixel2omega(pxy, sun2.shape[0])
        omegat = np.einsum("i,i", omega.ravel(), sun2.ravel())
        assert np.isclose(omegat, lf.omega, atol=.01, rtol=.01)
    for res in [5, 10, 50]:
        outf2 = lf2.direct_view(res=res)
        sun2 = io.hdr2array(outf2)
        vm = ViewMapper(viewangle=.666)
        pxy = vm.pixels(sun2.shape[0])
        omega = vm.pixel2omega(pxy, sun2.shape[0])
        omegat = np.einsum("i,i", omega.ravel(), sun2.ravel())
        assert np.isclose(omegat, lf2.omega, atol=.01, rtol=.01)
    Rtrace.reset()
    Rtrace._pyinstance = None
