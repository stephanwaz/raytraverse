#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.io"""
import os
import shutil
import sys

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


def test_capture(tmpdir):
    inp = ("cnt 100 | rcalc -f rayinit.cal -f b2d.cal -e 'side=10;bin=$1"
           ";$1=$1;$2=U;$3=V;$4=Dx;$5=Dy;$6=Dz' | rcalc -f rayinit.cal -f "
           "d2b.cal -e 'side=10;Dx=$4;Dy=$5;Dz=$6;$1=bin'")
    f = open("test.txt", 'w')
    with io.CaptureStdOut(outf=f) as cap:
        pass
    with io.CaptureStdOut(outf=f) as cap:
        print(inp)
    f.close()
    f = open('test.txt').read()
    assert inp + "\n" == f
    assert cap.stdout == inp + "\n"
    inp = np.arange(10)
    f = open("test.txt", 'wb')
    with io.CaptureStdOut(b=True, outf=f) as cap:
        sys.stdout.buffer.write(io.np2bytes(inp))
    f.close()
    f = open('test.txt', 'rb').read()
    assert np.allclose(inp, io.bytes2np(f, (10,)))
    assert np.allclose(inp, io.bytes2np(cap.stdout, (10,)))
    with pytest.raises(AttributeError):
        with io.CaptureStdOut(b=True, outf="test.txt"):
            pass
