#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.craytraverse"""
import importlib
import sys

from raytraverse import renderer, draw
import numpy as np
from scipy import stats
import pytest


import clasp.script_tools as cst


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


def test_empty_reset():
    rt = renderer.Rtrace()
    rt.reset()
    rt.reset()
    rt.reset_instance()
    rt = renderer.Rcontrib()
    rt.reset()
    rt.reset()
    rt.reset_instance()
    assert True


@pytest.mark.skipif(renderer.rtrace.cRtrace.version == "PyVirtual",
                    reason="no c-extension modules present")
def test_rtrace_call(capfd):
    args = "-ab 1 -ar 600 -ad 2000 -aa .2 -as 1500 -I"
    cargs = f"rtrace -h {args} -n 4 tests/test/test_run/sky.oct"
    check = cst.pipeline([cargs], inp='tests/test/rays.txt',
                         forceinpfile=True)
    check = np.fromstring(check, sep=' ').reshape(-1, 3)
    check2 = np.einsum('ij,j->i', check, [47.435/179, 119.93/179, 11.635/179])
    r = renderer.Rtrace(args, "tests/test/test_run/sky.oct", iot='aa')
    print('call')
    try:
        with capfd.disabled():
            ans = r.call('tests/test/rays.txt')
    except AttributeError:
        ans = r.call('tests/test/rays.txt')
    print('call done')
    test = np.fromstring(ans, sep=' ').reshape(-1, 3)
    r.update_ospec('ZL', 'a')
    try:
        with capfd.disabled():
            ans = r.call('tests/test/rays.txt')
    except AttributeError:
        ans = r.call('tests/test/rays.txt')
    test2 = np.fromstring(ans, sep=' ').reshape(-1, 2)
    r.reset()
    args2 = args + ' -oZ'
    r.initialize(args2, "tests/test/test_run/sky.oct", iot='af')
    try:
        with capfd.disabled():
            ans = r.call('tests/test/rays.txt')
    except AttributeError:
        ans = r.call('tests/test/rays.txt')
    # test3 = np.fromstring(ans, sep=' ').reshape(-1, 3)
    test3 = np.frombuffer(ans, '<f')
    assert np.allclose(check, test, atol=.03)
    assert np.allclose(check2, test2[:, 0], atol=.03)
    assert np.allclose(check2, test3, atol=.03)
    # print(r.header)


@pytest.mark.skipif(renderer.rcontrib.cRcontrib.version == "PyVirtual",
                    reason="no c-extension modules present")
def test_rcontrib_call(capfd):
    args = ('-V+ -I+ -ab 2 -ad 60000 -as 30000 -lw 1e-7 -e side:6'
            ' -f tests/test/scbins.cal -b bin -bn 36 -m skyglow ')
    cargs = f"rcontrib -n 5 -h- {args}  tests/test/test_run/sky.oct"
    check = cst.pipeline([cargs], inp='tests/test/rays2.txt',
                         forceinpfile=True)
    check = np.fromstring(check, sep=' ').reshape(-1, 36, 3)
    check = np.einsum('ikj,j->ik', check, [47.435/179, 119.93/179, 11.635/179])
    r = renderer.Rcontrib('-Z+' + args, 'tests/test/test_run/sky.oct', iot='aa')
    try:
        with capfd.disabled():
            ans = r.call('tests/test/rays.txt')
    except AttributeError:
        ans = r.call('tests/test/rays.txt')
    test = np.fromstring(ans, sep=' ').reshape(-1, 36)
    assert np.allclose(check, test, atol=.03)


if __name__ == "__main__":
    # test_rcontrib_call()
    rc = renderer.Rcontrib()
    # rt = renderer.Rtrace()
    # print(rt.instance)
    test_rtrace_call(None)
    rt = renderer.Rtrace()
    print(rt.header)
    rt.reset()
    test_rtrace_call(None)
    rt.reset_instance()
    test_rtrace_call(None)
    rt = renderer.Rtrace()
    rt.reset_instance()
    # print(rt.instance)
    print('rt done')

    # print(rc.instance)
    print('rc 1')
    test_rcontrib_call(None)
    print(rc.header)
    rc.reset()
    # print(rc.instance)
    print('rc 2')
    test_rcontrib_call(None)
    # rc.reset_instance()
    print('rc 3')
    test_rcontrib_call(None)
    rc = renderer.Rcontrib()
    # print(rc.instance)
    print('rc 4')
    test_rcontrib_call(None)
    test_rtrace_call(None)

