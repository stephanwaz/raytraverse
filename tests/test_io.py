#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.io"""
import os
import shutil

import pytest
from raytraverse import io
import numpy as np


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    cpath = os.getcwd()
    os.chdir(data)
    yield data
    os.chdir(cpath)


def test_array2img(tmpdir):
    b, a = np.mgrid[0:600, 0:400]
    ar = a*b
    io.array2hdr(ar, 'mgrid.hdr')
    io.array2hdr(a, 'mgrida.hdr')
    io.array2hdr(b, 'mgridb.hdr')
    a2 = io.hdr2array('mgrida.hdr')
    b2 = io.hdr2array('mgridb.hdr')
    ar2 = io.hdr2array('mgrid.hdr')
    assert np.allclose(a.T, a2, atol=.25, rtol=.03)
    assert np.allclose(b.T, b2, atol=.25, rtol=.03)
    assert np.allclose(ar2, ar2, atol=.25, rtol=.03)
