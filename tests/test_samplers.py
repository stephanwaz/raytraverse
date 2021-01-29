#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
import numpy as np

from raytraverse import io, translate
from raytraverse.integrator import MetricSet
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
    path = cpath + '/tests/samplers'
    os.chdir(path)
    yield path
    os.chdir(cpath)


# def test_skysample(tmpdir, capsys):
#
#     def img_illum(lf0, vm0, sbin):
#         outf = lf0.direct_view(res=512, srcidx=sbin, showsample=False,
#                                scalefactor=1)
#         hdr = io.hdr2array(outf)
#         vecs = vm0.pixelrays(512)
#         pxs = vm0.pixels(512).reshape(-1, 2)
#         omega = vm0.pixel2omega(pxs, 512)
#         cost = vm0.ctheta(vecs)
#
#         coefs = np.zeros(325)
#         coefs[sbin] = 1
#         illumm0 = MetricSet(vm, *lf.get_applied_rays(vm, coefs), ["illum"])()[0]
#
#         return np.sum(hdr.flatten()*omega*cost)*179, illumm0
#
#     scene = Scene('skysample', "box.rad", frozen=False)
#     sampler = SkySampler(scene, engine_args='-ab 1 -ad 10000 -c 1 -lw 1e-5')
#     vm = ViewMapper((0, 1, 0), viewangle=180)
#     with capsys.disabled():
#         lf = sampler.run((1.5, 1.5, 1.5), 0, vm)
#     illum, illumm = img_illum(lf, vm, 176)
#     assert np.isclose(illum, illumm, atol=.2, rtol=.1)
#     illum2, illumm2 = img_illum(lf, vm, 158)
#     assert np.isclose(illum2, illumm2, atol=.2, rtol=.1)
#     assert np.isclose(np.average([illum, illumm, illum2, illumm2]), 2.88, atol=.1, rtol=.2)
#     illum, illumm = img_illum(lf, vm, 174)
#     assert np.isclose(illum, illumm, atol=.2, rtol=.1)
#     assert np.isclose(np.average([illum, illumm]), 0.169, atol=.03, rtol=.2)
#     fmetric = MetricSet(vm, *lf.get_applied_rays(vm, np.ones(325)), ["illum", "density"])()
#     assert np.isclose(fmetric[0], 352.7, atol=1, rtol=.01)
#     assert np.isclose(fmetric[1], 2367, atol=30)
#
#
# def test_sunsample(tmpdir, capsys):
#
#     def img_illum(lf0, vm0):
#         outf = lf0.direct_view(res=512, showsample=False, scalefactor=285.32)
#         hdr = io.hdr2array(outf)
#         vecs = vm0.pixelrays(512)
#         pxs = vm0.pixels(512).reshape(-1, 2)
#         omega = vm0.pixel2omega(pxs, 512)
#         cost = vm0.ctheta(vecs)
#         illumm0 = MetricSet(vm, *lf.get_applied_rays(vm, 285.32), ["illum"])()[0]
#
#         return np.sum(hdr.flatten()*omega*cost)*179, illumm0
#
#     scene = Scene('skysample', "box.rad", frozen=False)
#     sun = translate.skybin2xyz([174], 18)[0]
#     sampler = SunSampler(scene, sun, 174, engine_args='-ab 0')
#     vm = ViewMapper((0, 1, 0), viewangle=180)
#     with capsys.disabled():
#         lf = sampler.run((1.5, 1.5, 1.5), 0, vm)
#     illum, illumm = img_illum(lf, vm)
#     assert np.isclose(illum, illumm, atol=.01, rtol=.05)
#     assert np.isclose(np.average([illum, illumm]), 0.169, atol=.01, rtol=.01)


def test_sunviewsample(tmpdir, capsys):
    scene = Scene('skysample', "box.rad", frozen=False)
    sun = translate.skybin2xyz([176], 18)[0]
    sampler = SunViewSampler(scene, sun, 176)
    with capsys.disabled():
        lf = sampler.run((1.5, 1.5, 1.79), 0, plotp=True)
        img = np.zeros((1024, 1024))
        vm = ViewMapper(sun, 2)
        vecs = vm.pixelrays(1024)
        mask = vm.in_view(vecs)
        print(vecs.shape)
        lf.add_to_img(img, vecs[mask], mask, vm=vm)
        io.array2hdr(img, "sunview.hdr")
