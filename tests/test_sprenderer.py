#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.craytraverse"""
import importlib
import sys

from raytraverse import renderer
import numpy as np

import clasp.script_tools as cst


def test_rtrace_call(capfd):
    args = "-ab 1 -ar 600 -ad 2000 -aa .2 -as 1500 -I"
    cargs = f"rtrace -h {args} -n 4 tests/test/test_run/sky.oct"
    check = cst.pipeline([cargs], inp='tests/test/rays.txt',
                         forceinpfile=True)
    check = np.fromstring(check, sep=' ').reshape(-1, 3)
    check2 = np.einsum('ij,j->i', check, [47.435/179, 119.93/179, 11.635/179])
    r = renderer.SPRtrace(args, "tests/test/test_run/sky.oct", iot='af')
    try:
        with capfd.disabled():
            ans = r.call('tests/test/rays.txt')
    except AttributeError:
        ans = r.call('tests/test/rays.txt')
    test = np.frombuffer(ans, '<f')
    r.initialize(args, "tests/test/test_run/sky.oct", iot='af')
    try:
        with capfd.disabled():
            ans = r.call('tests/test/rays.txt')
    except AttributeError:
        ans = r.call('tests/test/rays.txt')
    # test3 = np.fromstring(ans, sep=' ').reshape(-1, 3)
    test3 = np.frombuffer(ans, '<f')
    assert np.allclose(check2, test, atol=.03)
    assert np.allclose(check2, test3, atol=.03)
    # print(r.header)


def test_rcontrib_call(capfd):
    args = ('-V+ -I+ -ab 2 -ad 60000 -as 30000 -lw 1e-7 -e side:6'
            ' -f tests/test/scbins.cal -b bin -bn 36 -m skyglow ')
    cargs = f"rcontrib -n 5 -h- {args}  tests/test/test_run/sky.oct"
    check = cst.pipeline([cargs], inp='tests/test/rays2.txt',
                         forceinpfile=True)
    check = np.fromstring(check, sep=' ').reshape(-1, 36, 3)
    check = np.einsum('ikj,j->ik', check, [47.435/179, 119.93/179, 11.635/179])
    r = renderer.SPRcontrib(args, 'tests/test/test_run/sky.oct', iot='af')
    try:
        with capfd.disabled():
            ans = r.call('tests/test/rays.txt')
    except AttributeError:
        ans = r.call('tests/test/rays.txt')
    test = np.frombuffer(ans, '<f').reshape(-1, 36)
    assert np.allclose(check, test, atol=.03)

