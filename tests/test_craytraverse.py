#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.craytraverse"""
import os
import shutil
import sys

from raytraverse import renderer
from raytraverse.sampler import draw
import numpy as np
from scipy import stats
import pytest


import clasp.script_tools as cst


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/craytraverse/', data + '/test')
    cpath = os.getcwd()
    # use temp
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/craytraverse'
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_from_pdf():
    rv = stats.norm()
    nsamp = 500000
    t = .01
    x = np.linspace(0, 5, nsamp)
    pdf = rv.pdf(x)
    exp = np.sum(pdf > t)
    # c2 = draw.from_pdf2(pdf, t)
    c = draw.from_pdf(pdf, t)
    # print(np.sum(pdf > t), c.size, c2.size)
    # hist = np.histogram(x[c], 50)
    # mplt.quick_scatter([hist[1][1:], [0, 0]], [hist[0], [0, 7000]])
    # print(exp, c.size, c2.size)
    assert np.isclose(c.size, exp)


def test_empty_reset(tmpdir):
    rt = renderer.Rtrace("", "sky.oct")
    args = ('-I+ -ab 2 -ad 60000 -as 30000 -lw 1e-7')
    rc = renderer.Rcontrib(args, "scene.oct")
    rc.reset()
    rc.reset()
    rt.reset()
    rt.reset()
    assert True


def test_rcontrib_call(capfd, tmpdir):
    args = ('-V+ -I+ -ab 2 -ad 60000 -as 30000 -lw 1e-7 -e side:6'
            ' -f scbins.cal -b bin -bn 36 -m skyglow ')
    cargs = f"rcontrib -n 5 -h- {args}  sky.oct"
    check = cst.pipeline([cargs], inp='rays2.txt',
                         forceinpfile=True)
    check = np.fromstring(check, sep=' ').reshape(-1, 36, 3)
    check = np.einsum('ikj,j->ik', check, [47.435/179, 119.93/179, 11.635/179])
    r = renderer.Rcontrib(rayargs=None, skyres=30, ground=False)
    r.set_args('-I+ -ab 2 -ad 600 -as 300 -c 100 -lw 1e-5')
    r.load_scene("sky.oct")
    vecs = np.loadtxt('rays2.txt')
    # try:
    #     with capfd.disabled():
    #         ans, b = r.call(vecs, 'rays2.txt')
    # except AttributeError:
    test = r(vecs)
    # test = np.fromstring(ans, sep=' ').reshape(-1, 36)
    assert np.allclose(check, test, atol=.03)
    renderer.Rcontrib.reset()


def test_rtrace_call(tmpdir):
    args = "-ab 1 -ar 600 -ad 2000 -aa .2 -as 1500 -I+"
    cargs = f"rtrace -h {args} -n 4 sky.oct"
    check = cst.pipeline([cargs], inp='rays.txt',
                         forceinpfile=True)
    check = np.fromstring(check, sep=' ').reshape(-1, 3)
    check2 = np.einsum('ij,j->i', check, [47.435/179, 119.93/179, 11.635/179])
    # first load
    r = renderer.Rtrace(args, "sky.oct", default_args=False)
    print(r.instance)
    vecs = np.loadtxt('rays.txt')
    ans = r(vecs)
    assert np.allclose(check, ans, atol=.03)

    # change output
    r.update_ospec('ZL')
    ans = r(vecs)
    assert np.allclose(check2, ans[:, 0], atol=.03)


    #
    # reload and change output to float
    r.reset()
    args2 = args + ' -oZ'
    r = renderer.Rtrace(args, "sky.oct", default_args=True)
    test3 = r(vecs).ravel()
    assert np.allclose(check2, test3, atol=.03)
    #
    # reload new scene
    r.load_scene("sky.oct")
    test3 = r(vecs).ravel()
    assert np.allclose(check2, test3, atol=.03)
    #
    # change args
    r.set_args("-ab 0 -oZ -I")
    test3 = r(vecs).ravel()
    assert np.allclose([0, 0, 0, 0, np.pi*2], test3, atol=.03)
    #
    # change back
    r.set_args(args2)
    test3 = r(vecs).ravel()
    assert np.allclose(check2, test3, atol=.03)
    #
    # load sources
    r.set_args("-ab 0 -oZ")
    r.load_scene("scene.oct")
    r.load_source("sun.rad")
    test = r(vecs).ravel()
    r.load_source("sun2.rad")
    test2 = r(vecs).ravel()
    assert np.allclose(test * 2, test2, atol=.03)
    cargs = f"rtrace -h -ab 0 -n 4 sun.oct"
    check = cst.pipeline([cargs], inp='rays.txt',
                         forceinpfile=True)
    check = np.fromstring(check, sep=' ').reshape(-1, 3)
    check2 = np.einsum('ij,j->i', check, [47.435/179, 119.93/179, 11.635/179])
    assert np.allclose(test, check2, atol=.03)


    r.set_args(args)
    test2 = r(vecs).ravel()
    r.load_source("sky.rad")
    test3 = r(vecs).ravel()
    r.load_source("sun2.rad", 0)
    test4 = r(vecs).ravel()
    r.load_source("sun.rad")
    test5 = r(vecs).ravel()
    assert np.allclose(test2 + test3, test4, atol=.03)
    assert np.allclose(test2, test5, atol=.03)
    renderer.Rtrace.reset()


    # print(r.header)


