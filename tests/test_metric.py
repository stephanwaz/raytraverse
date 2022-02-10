#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.metric"""
import os

import pytest
from raytraverse import io
from raytraverse.mapper import ViewMapper
from raytraverse.evaluate import MetricSet, PositionIndex
import numpy as np


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    cpath = os.getcwd()
    os.chdir(data)
    yield data
    os.chdir(cpath)


def test_get_pos_idx(tmpdir):
    exp = [[16.0, 16.0, 16.0, 8.076347019578511, 6.856663497343681, 8.968471163183743, 16.0, 16.0, 16.0, 16.0],
           [16.0, 9.477810403835639, 5.965841356086234, 4.2579984530618225, 3.548712152029083, 4.522287557454674, 8.626821872361402, 16.0, 16.0, 16.0],
           [16.0, 6.087587840512674, 3.743634068668167, 2.5905982171679907, 2.1066586423000793, 2.6158672551048934, 4.88801223548715, 11.250158366243358, 16.0, 16.0],
           [8.242109188672083, 4.5378327204826645, 2.729101697082527, 1.8268060895950733, 1.4364596439443964, 1.7403135007849175, 3.28638637322815, 7.818635653939338, 16.0, 16.0],
           [7.163371648976106, 3.918286200607336, 2.3229020762248958, 1.5151625926531762, 1.1368003104524078, 1.3622203177528458, 2.8170336179218136, 6.980479737328229, 16.0, 16.0],
           [7.163371648976106, 3.918286200607336, 2.3229020762248958, 1.5151625926531764, 1.1368003104524078, 1.3622203177528456, 2.817033617921815, 6.980479737328229, 16.0, 16.0],
           [8.242109188672083, 4.5378327204826645, 2.729101697082527, 1.8268060895950733, 1.4364596439443964, 1.7403135007849182, 3.286386373228152, 7.818635653939338, 16.0, 16.0],
           [16.0, 6.087587840512674, 3.743634068668167, 2.5905982171679907, 2.1066586423000793, 2.615867255104895, 4.888012235487156, 11.250158366243358, 16.0, 16.0],
           [16.0, 9.477810403835639, 5.965841356086234, 4.2579984530618225, 3.548712152029083, 4.522287557454676, 8.626821872361402, 16.0, 16.0, 16.0],
           [16.0, 16.0, 16.0, 8.076347019578508, 6.8566634973436775, 8.968471163183748, 16.0, 16.0, 16.0, 16.0]]
    vm = ViewMapper(viewangle=180)
    res = 10
    img = vm.pixelrays(res)
    fimg = img.reshape(-1, 3)
    posfinder = PositionIndex()
    posidx = posfinder.positions(vm, fimg).reshape(res, res)
    assert np.allclose(posidx, exp)


def test_position():
    vm = ViewMapper((.5, .5, -1), viewangle=180)
    res = 1000
    img = vm.pixelrays(res)
    fimg = img.reshape(-1, 3)
    cos = vm.ctheta(fimg)
    pc = cos.reshape(res, res)
    io.array2hdr(pc, "position_cos.hdr")
    posfinder = PositionIndex()
    posidx = posfinder.positions(vm, fimg).reshape(res, res)
    pg = 1/np.square(posidx).reshape(res, res)
    io.array2hdr(pg, "position_guth.hdr")
    posfinder = PositionIndex(guth=False)
    posidx = posfinder.positions(vm, fimg).reshape(res, res)
    pk = 1/posidx.reshape(res, res)
    io.array2hdr(pk, "position_kim.hdr")
    position_kim = io.hdr2array("position_kim.hdr")
    position_guth = io.hdr2array("position_guth.hdr")
    position_cos = io.hdr2array("position_cos.hdr")
    assert np.allclose(position_kim, pk, atol=4.8, rtol=.1)
    assert np.allclose(position_guth, pg, atol=.5, rtol=.1)
    assert np.allclose(position_cos, pc, atol=.6, rtol=.1)
