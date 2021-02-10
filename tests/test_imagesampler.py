#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
import numpy as np

from raytraverse import io
from raytraverse.evaluate import MetricSet
from raytraverse.scene import ImageScene
from raytraverse.sampler import DeterministicImageSampler, ImageSampler
from raytraverse.mapper import ViewMapper


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/imagesampler/', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    # os.chdir(cpath + '/tests/imagesampler')
    yield data + '/test'
    # yield cpath + '/tests/imagesampler'
    os.chdir(cpath)


def test_sample(tmpdir):
    scene = ImageScene('imgsample', "oct21_detail_glz_EW_desk.hdr")
    sampler = DeterministicImageSampler(scene, idres=5, fdres=10, accuracy=4.0)
    sampler2 = ImageSampler(scene, idres=5, fdres=10, accuracy=4.0)
    vm = ViewMapper(viewangle=180)
    lf = sampler.run((0, 0, 0), 0, vm)
    lf2 = sampler2.run((0, 0, 0), 0, vm)
    ref = sampler.engine.scene
    lf.direct_view(ref.shape[0], showsample=False, interp=True)
    fmetric = MetricSet(vm, *lf.get_applied_rays(1, vm), ["illum", "density"])()
    fmetric2 = MetricSet(vm, *lf2.get_applied_rays(1, vm), ["illum", "density"])()
    assert np.abs(fmetric[0] - 28200) < 10
    assert np.abs(fmetric[1] - 2920) < 20
    assert np.abs(fmetric[0] - fmetric2[0])/fmetric[0] < .05
    assert np.abs(fmetric[1] - fmetric2[1])/fmetric[1] < .1
    test = io.hdr2array("imgsample_image_000000.hdr")
    filt = np.logical_and(test > 0, ref > 0)
    mad = np.average(np.abs(ref[filt] - test[filt]))
    msd = np.average(test[filt] - ref[filt])
    assert mad < 6 and np.abs(msd) < 2
    lf.direct_view(ref.shape[0], showsample=True, interp=False, omega=True)
    test = io.hdr2array("imgsample_image_000000_omega.hdr")
    assert np.abs(np.sum(test) - 7280) < 3
    assert np.isclose(np.sum(lf.omega), np.pi * 2)

