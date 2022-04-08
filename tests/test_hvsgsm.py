#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.evaluate.hvsgsm"""
import os
import shutil

import pytest
import numpy as np

from raytraverse.evaluate.hvsgsm import GSS
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


def test_init(tmpdir):
    with pytest.raises(FileNotFoundError):
        GSS("nofile")
    g1 = GSS(ViewMapper((.5, .5, .5)))
    assert np.allclose(g1.vm.dxyz, (0.57735027, 0.57735027, 0.57735027))
    g1 = GSS()
    assert np.allclose(g1.vm.dxyz, (0, 1, 0))
    g2 = GSS("v1.vf")
    assert np.allclose(g2.vm.dxyz, (-1, 0, 0))
    with pytest.raises(ValueError):
        g2.compute()


def test_lumset(tmpdir):
    gss = GSS("oct21_detail_glz_EW_desk.hdr", age=35, pigmentation=0.142)
    assert np.allclose(gss.vm.dxyz, (-1, 0, 0))
    # bs, cs = gss.psf_coef(2)
    # a = "z = log((x * π/180)^2 * "
    # for b, c in zip(bs, cs):
    #     print(f"b: {b:.05f}, c: {c:.05f}")
    #     print("z = log(10^(x*π/180))^2 * c * b/(2 * π * (sin^2(10^(x*π/180)) + b^2*cos^2(10^(x*π/180)))^1.5))")
    # a = a[:-2] + ")"
    # print(a)

