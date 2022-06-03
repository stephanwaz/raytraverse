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
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/imagesampler'
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_sample(tmpdir):
    scene = ImageScene('imgsample', "oct21_detail_glz_EW_desk.hdr")
    sampler = DeterministicImageSampler(scene, idres=32, nlev=6, accuracy=4.0)
    sampler2 = ImageSampler(scene, idres=32, nlev=6, accuracy=4.0)
    vm = ViewMapper(viewangle=180, jitterrate=0)
    lf = sampler.run((0, 0, 0), 0, vm)
    lf2 = sampler2.run((0, 0, 0), 0, vm)
    ref = sampler.engine.scene
    lf.direct_view(ref.shape[0], showsample=False, interp=False)
    lf.direct_view(ref.shape[0], showsample=False, interp=False, omega=True)
    fmetric = MetricSet(*lf.evaluate(1, vm), vm, ["illum", "density"])()
    fmetric2 = MetricSet(*lf2.evaluate(1, vm), vm, ["illum", "density"])()
    assert np.abs(fmetric[0] - 28144) < 40
    assert np.abs(fmetric[1] - 2480) < 20
    assert np.abs(fmetric[0] - fmetric2[0])/fmetric[0] < .05
    assert np.abs(fmetric[1] - fmetric2[1])/fmetric[1] < .1
    test = io.hdr2array("imgsample_image_000000.hdr")
    filt = np.logical_and(test > 0, ref > 0)
    mad = np.average(np.abs(ref[filt] - test[filt]))
    msd = np.average(test[filt] - ref[filt])
    assert mad < 6 and np.abs(msd) < 2
    lf.direct_view(ref.shape[0], showsample=True, interp=False, omega=True)
    test = io.hdr2array("imgsample_image_000000_omega.hdr")
    assert np.abs(np.sum(test) - 6866) < 3
    assert np.isclose(np.sum(lf.omega), np.pi * 2)

