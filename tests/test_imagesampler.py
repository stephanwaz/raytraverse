#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil
import re

import pytest
import numpy as np

from raytraverse import io
from raytraverse.lightfield import StaticField
from raytraverse.scene import ImageScene
from raytraverse.sampler import ImageSampler
from datetime import datetime


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/test/', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    # os.chdir(cpath + '/tests/test')
    yield data + '/test'
    # yield cpath + '/tests/test'
    os.chdir(cpath)


def test_sample(tmpdir):
    scene = ImageScene('imgsample', "oct21_detail_glz_EW_desk.hdr")
    sampler = ImageSampler(scene, idres=5, fdres=10, accuracy=1.0)
    ref = sampler.engine.scene
    sampler.run()
    lf = StaticField(scene, prefix="image")
    lf.direct_view(ref.shape[0], showsample=False, interp=1)
    del scene
    f = open('imgsample/log.txt', 'r')
    log = f.read()
    f.close()
    test = io.hdr2array("imgsample_image_0000.hdr")
    filt = np.logical_and(test > 0, ref > 0)
    mad = np.average(np.abs(ref[filt] - test[filt]))
    rmse = np.sqrt(np.average(np.square(ref[filt] - test[filt])))
    sampling = int(log.split("total sampling:")[-1].strip().split()[1])
    log = log.splitlines(keepends=False)
    ts = (log[0].split('\t')[0], log[-1].split('\t')[0])
    start = datetime.strptime(ts[0], "%d-%b-%Y %H:%M:%S")
    end = datetime.strptime(ts[1], "%d-%b-%Y %H:%M:%S")
    print()
    np.set_printoptions(3, suppress=True)
    print(np.average(ref[filt]), np.average(test[filt]))
    print(np.percentile(ref[filt], (0, 50, 100)))
    print(np.percentile(test[filt], (0, 50, 100)))
    print(rmse, mad, sampling, end - start)
    assert mad < 70
