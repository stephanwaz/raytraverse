#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.craytraverse"""
import os
import shutil
import sys

from raytraverse import renderer
from raytraverse.sampler import draw
from raytraverse.formatter import RadianceFormatter
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

# breaks test/example foor soome reason (must be in c-code!!!)
# def test_rcontrib_call(capfd, tmpdir):
#     args = ('-V+ -I+ -ab 2 -ad 60000 -as 30000 -lw 1e-7 -e side:6'
#             ' -f scbins.cal -b bin -bn 36 -m skyglow ')
#     cargs = f"rcontrib -n 5 -h- {args}  sky.oct"
#     check = cst.pipeline([cargs], inp='rays2.txt',
#                          forceinpfile=True)
#     check = np.fromstring(check, sep=' ').reshape(-1, 36, 3)
#     check = np.einsum('ikj,j->ik', check, [47.435/179, 119.93/179, 11.635/179])
#     r = renderer.Rcontrib(rayargs='-I+ -ab 2 -ad 600 -as 300 -c 100 -lw 1e-5',
#                           skyres=30, ground=False, scene="sky.oct")
#     # r.set_args('-I+ -ab 2 -ad 600 -as 300 -c 100 -lw 1e-5')
#     # r.load_scene("sky.oct")
#     vecs = np.loadtxt('rays2.txt')
#     # try:
#     #     with capfd.disabled():
#     #         ans, b = r.call(vecs, 'rays2.txt')
#     # except AttributeError:
#     test = r(vecs)
#     # test = np.fromstring(ans, sep=' ').reshape(-1, 36)
#     assert np.allclose(check, test, atol=.03)
#     r.reset()


def test_rtrace_call(tmpdir):
    args = "-ab 1 -ar 600 -ad 2000 -aa .2 -as 1500 -I+"
    cargs = f"rtrace -h {args} -n 4 sky.oct"
    check = cst.pipeline([cargs], inp='rays.txt',
                         forceinpfile=True)
    check = np.fromstring(check, sep=' ').reshape(-1, 3)
    check2 = np.einsum('ij,j->i', check, [47.435/179, 119.93/179, 11.635/179])
    # first load
    r = renderer.Rtrace(args, "sky.oct", default_args=False)
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
    # test3 = r(vecs).ravel()
    # assert np.allclose(check2, test3, atol=.03)
    #
    # reload new scene
    r.load_scene("sky.oct")
    test3 = r(vecs).ravel()
    assert np.allclose(check2, test3, atol=.03)
    # #
    # # change args
    r.set_args("-lr 0 -oZ -I")
    test3 = r(vecs).ravel()
    assert np.allclose([0, 0, 0, 0, np.pi*2], test3, atol=.03)
    # #
    # # change back
    r.set_args(args2)
    test3 = r(vecs).ravel()
    assert np.allclose(check2, test3, atol=.03)
    # #
    # load sources
    r.set_args("-lr 0")
    r.update_ospec("Z")
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
    #
    #
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
    r.reset()


def test_ambient_reset(tmpdir):

    formatter = RadianceFormatter()
    args = "-u- -ab 1 -ar 1000 -ad 4000 -aa .05 -as 2000 -I+"
    r = renderer.Rtrace(args, "scene.oct", default_args=False)
    vecs = np.loadtxt('rays2.txt')

    def load_sun(sun, val, af=None):
        srcdef = f'tmp_sun.rad'
        f = open(srcdef, 'w')
        f.write(formatter.get_sundef(sun, (val, val, val)))
        f.close()
        r.load_source(srcdef, ambfile=af)
        os.remove(srcdef)

    load_sun((0, -.5, 1), 1000000, "temp.amb")
    r(vecs)
    a1 = r(vecs).ravel()
    load_sun((0, -.5, 1), 1000000, "temp2.amb")
    r(vecs)
    a2 = r(vecs).ravel()

    load_sun((-.5, -.5, 1), 2000000, "temp3.amb")
    r(vecs)
    a3 = r(vecs).ravel()

    load_sun((0, -.5, 1), 1000000, "temp.amb")
    a1a = r(vecs).ravel()
    load_sun((0, -.5, 1), 1000000, "temp2.amb")
    a2a = r(vecs).ravel()
    load_sun((0, -.5, 1), 1000000, "temp4.amb")
    r(vecs)
    a4 = r(vecs).ravel()
    r.set_args(r.defaultargs + " -I+ -ab 1 -c 30")
    aa0 = r(np.repeat(vecs[0:1], 1000, 0)).ravel()
    d = stats.norm(loc=np.mean(aa0), scale=np.std(aa0))
    ks = stats.kstest(aa0, d.cdf)
    # bm = np.random.default_rng().normal(loc=np.mean(aa0),
    #                                     scale=np.std(aa0),
    #                                     size=10000)
    # ks2 = stats.kstest(bm, d.cdf)
    assert ks[0] < .08
    r.set_args(args)
    load_sun((-.5, -.5, 1), 2000000, "temp3.amb")
    a3a = r(vecs).ravel()
    assert(np.allclose(a1, a1a))
    assert (np.allclose(a2, a2a))
    assert (np.allclose(a3, a3a))
    assert 1e-8 < np.sum(np.abs(a2 - a4)) < .5
    load_sun((-.5, -.5, 1), 2000000)
    noamb = r(vecs).ravel()
    assert 1e-8 < np.sum(np.abs(noamb - a3)) < 1
    r.reset()


def test_ambient_nostore(tmpdir):

    formatter = RadianceFormatter()
    args = "-u- -ab 1 -ar 1000 -ad 4000 -aa .05 -as 2000 -I+"
    r = renderer.Rtrace(args, "scene.oct", default_args=False, nproc=1)
    vecs = np.loadtxt('rays2.txt')

    def load_sun(sun, val):
        srcdef = f'tmp_sun.rad'
        f = open(srcdef, 'w')
        f.write(formatter.get_sundef(sun, (val, val, val)))
        f.close()
        r.load_source(srcdef)
        os.remove(srcdef)

    load_sun((0, -.5, 1), 1000000)

    # ambient values are not shared
    r(vecs)
    an2 = r(vecs).ravel()
    an3 = r(vecs).ravel()
    assert 1e-8 < np.sum(np.abs(an2 - an3)) < .5

    # ambient values are shared (vectors repeated, 1 process)
    an4 = r(np.repeat(vecs, 2, 0)).reshape(-1, 2).T
    assert (np.allclose(an4[0], an4[1]))
    r.reset()
