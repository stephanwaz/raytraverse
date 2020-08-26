#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.craytraverse"""
import importlib

from raytraverse import craytraverse, io, renderer
import numpy as np
import pytest

import clasp.script_tools as cst


def test_rtrace_call():
    args = ("rtrace -n 4 -ab 1 -ar 600 -ad 2000 -aa .2 -as 1500 -I -h "
            "tests/test/test_run/sky.oct").split()
    check = cst.pipeline([' '.join(args)], inp='tests/test/rays.txt',
                         forceinpfile=True)
    check = np.fromstring(check, sep=' ').reshape(-1, 3)
    r = renderer.Rtrace(args)
    ans = r.call('tests/test/rays.txt')
    test = np.fromstring(ans, sep=' ').reshape(-1, 3)
    r.update_ospec('ZL', 'a')
    ans = r.call('tests/test/rays.txt')
    test2 = np.fromstring(ans, sep=' ').reshape(-1, 2)
    r.reset(args)
    ans = r.call('tests/test/rays.txt')
    test3 = np.fromstring(ans, sep=' ').reshape(-1, 3)
    print('check', check)
    print('test', test)
    print('test3', test3)
    assert np.allclose(check, test, atol=.03)
    assert np.allclose(check[:, 1], test2[:, 0], atol=.03)
    assert np.allclose(test, test3, atol=.03)
    # r.reset_instance()
    print(test3)


def test_rcontrib_call():
    args = ('rcontrib -V+ -I+ -ab 2 -ad 60000 -as 30000 -h -lw 1e-7 -n 1 -e side:6'
            ' -f tests/test/scbins.cal -b bin -bn 36 -m skyglow '
            'tests/test/test_run/sky.oct').split()
    print(renderer.Rcontrib.initialized)
    r = renderer.Rcontrib(args)
    # print(r._pyinstance)
    # print(renderer.Rcontrib.initialized)
    # print(r.instance)
    ans = r.call('tests/test/rays.txt')
    test4 = np.fromstring(ans, sep=' ').reshape(-1, 36, 3)
    print(test4[-1, -8])
    ans = r.call('tests/test/rays.txt')
    # ans = r.call('tests/test/rays.txt')
    # ans = r.call('tests/test/rays.txt')
    test4 = np.fromstring(ans, sep=' ').reshape(-1, 36, 3)
    print(test4[-1, -8])
    # r.reset_instance()


if __name__ == "__main__":
    rc = renderer.Rcontrib()
    rt = renderer.Rtrace()
    print(rt.instance)
    test_rtrace_call()
    rt.reset()
    test_rtrace_call()
    rt.reset_instance()
    test_rtrace_call()
    rt = renderer.Rtrace()
    print(rt.instance)

    print(rc.instance)
    test_rcontrib_call()
    rc.reset()
    print(rc.instance)
    test_rcontrib_call()
    rc.reset_instance()
    test_rcontrib_call()
    rc = renderer.Rcontrib()
    print(rc.instance)

    # test_rcontrib_call()

