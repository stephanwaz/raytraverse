#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.io"""
import os
import re

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
    assert np.allclose(ar.T, ar2, atol=.25, rtol=.03)


def test_setproc():
    nproc = io.get_nproc(8)
    assert nproc == 8
    nproc = io.get_nproc()
    assert nproc == os.cpu_count()
    io.set_nproc(7)
    assert io.get_nproc() == 7
    io.unset_nproc()
    assert io.get_nproc() == os.cpu_count()
    io.unset_nproc()
    with pytest.raises(ValueError):
        io.set_nproc("7")
    assert io.get_nproc() == os.cpu_count()


def test_version_header():
    header = io.version_header()
    check = r'CAPDATE= .* UTC\nSOFTWARE= RAYTRAVERSE .* lastmod .* // RADIANCE .*'
    assert re.match(check, "\n".join(header))


def test_npbytefile(tmpdir):
    a = np.arange(999).reshape(-1, 9)
    io.np2bytefile(a, "test_npbytefile")
    c = io.bytefile2np(open("test_npbytefile", 'rb'), (-1, 9))
    assert np.all(a == c)
