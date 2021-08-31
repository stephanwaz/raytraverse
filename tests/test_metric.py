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
    exp = [[16., 16., 11.08225407, 8.05160291, 6.85207743, 8.96847116, 16., 16.,
            16., 16.],
           [16., 9.45072581, 5.8238892, 4.1874664, 3.54008629, 4.52228756,
            8.62682187, 16., 16., 16.],
           [11.08225407, 5.8238892, 3.54008629, 2.50789922, 2.09708451,
            2.61586726, 4.88801224, 11.25015837, 16., 16.],
           [8.05160291, 4.1874664, 2.50789922, 1.74084506, 1.42630958,
            1.7403135, 3.28638637, 7.81863565, 16., 16.],
           [6.85207743, 3.54008629, 2.09708451, 1.42630958, 1.12383868,
            1.36222032, 2.81703362, 6.98047974, 16., 16.],
           [6.85207743, 3.54008629, 2.09708451, 1.42630958, 1.12383868,
            1.36222032, 2.81703362, 6.98047974, 16., 16.],
           [8.05160291, 4.1874664, 2.50789922, 1.74084506, 1.42630958,
            1.7403135, 3.28638637, 7.81863565, 16., 16.],
           [11.08225407, 5.8238892, 3.54008629, 2.50789922, 2.09708451,
            2.61586726, 4.88801224, 11.25015837, 16., 16.],
           [16., 9.45072581, 5.8238892, 4.1874664, 3.54008629, 4.52228756,
            8.62682187, 16., 16., 16.],
           [16., 16., 11.08225407, 8.05160291, 6.85207743, 8.96847116, 16., 16.,
            16., 16.]]
    vm = ViewMapper(viewangle=180)
    res = 10
    img = vm.pixelrays(res)
    fimg = img.reshape(-1, 3)
    posfinder = PositionIndex()
    posidx = posfinder.positions(vm, fimg).reshape(res, res)
    assert np.allclose(posidx, exp)


def test_position():
    vm = ViewMapper(viewangle=180)
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
