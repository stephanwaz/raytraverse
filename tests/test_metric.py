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
    exp = [[2.425470197687602, 2.705124422836401, 3.250879646438889,
            4.510683974902716, 4.6, 8.968471163183748, 16.0, 16.0, 16.0, 16.0],
           [2.705124422836401, 2.696633390419641, 2.7728963523408092,
            3.03767302360837, 3.310302356612268, 4.522287557454674,
            8.626821872361402, 16.0, 16.0, 16.0],
           [3.250879646438889, 2.7728963523408083, 2.3897578466576777,
            2.2356889432064517, 2.2019576000714194, 2.6158672551048947,
            4.888012235487149, 11.250158366243358, 16.0, 16.0],
           [4.510683974902716, 3.0376730236083707, 2.2356889432064517,
            1.8247960248510424, 1.4234251448256428, 1.7403135007849175,
            3.2863863732281517, 7.818635653939338, 16.0, 16.0],
           [4.6, 3.310302356612268, 2.2019576000714203, 1.423425144825643,
            1.1784359625867407, 1.362220317752846, 2.8170336179218136,
            6.980479737328231, 16.0, 16.0],
           [4.6, 3.310302356612268, 2.2019576000714203, 1.423425144825643,
            1.1784359625867407, 1.3622203177528456, 2.817033617921814,
            6.980479737328229, 16.0, 16.0],
           [4.510683974902716, 3.0376730236083707, 2.235688943206452,
            1.8247960248510426, 1.423425144825643, 1.7403135007849182,
            3.2863863732281517, 7.818635653939338, 16.0, 16.0],
           [3.250879646438889, 2.7728963523408092, 2.3897578466576777,
            2.235688943206453, 2.2019576000714207, 2.615867255104895,
            4.888012235487156, 11.250158366243358, 16.0, 16.0],
           [2.705124422836401, 2.696633390419641, 2.7728963523408092,
            3.037673023608373, 3.310302356612268, 4.522287557454676,
            8.626821872361402, 16.0, 16.0, 16.0],
           [2.425470197687602, 2.705124422836402, 3.2508796464388903,
            4.5106839749027134, 4.6, 8.968471163183748, 16.0, 16.0, 16.0, 16.0],
           ]
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

