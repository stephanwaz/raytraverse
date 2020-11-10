#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.integrate.retina"""
import os

import pytest
from raytraverse import translate, io
from raytraverse.integrator import retina
import numpy as np


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    cpath = os.getcwd()
    os.chdir(data)
    yield data
    os.chdir(cpath)


def test_rgcf_density(tmpdir):
    grid = 999
    uv = np.reshape((np.mgrid[0:grid, 0:grid]+.5) / grid, (2, -1)).T
    xy = translate.uv2xy(uv)
    density = retina.rgcf_density(xy)
    ref = np.array([4.75929761e+00, 5.34634456e+00,
                    6.09887818e+00, 7.10039102e+00, 8.49759373e+00,
                    1.05803866e+01, 1.40277281e+01, 2.08498700e+01,
                    4.08354445e+01, 3.30310525e+04])
    io.uvarray2hdr(density.reshape(grid, grid), "density.hdr")
    test = np.quantile(density, np.linspace(.1, 1, 10))
    assert np.allclose(ref, test, atol=.01, rtol=.01)


def test_rgc_density(tmpdir):
    grid = 999
    uv = np.reshape((np.mgrid[0:grid, 0:grid]+.5) / grid, (2, -1)).T
    xy = translate.uv2xy(uv)
    density = retina.rgc_density(xy)
    ref = np.array([4.4559419, 6.53164379, 9.30496826,
                    12.76227423, 17.14212168, 21.7143411,
                    28.35079283, 43.31721288, 89.69326626,
                    2190.76230207])
    io.uvarray2hdr(density.reshape(grid, grid), "density2.hdr")
    test = np.quantile(density, np.linspace(.1, 1, 10))
    assert np.allclose(ref, test, rtol=.01)
