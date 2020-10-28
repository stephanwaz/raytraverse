#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.draw"""

import pytest
from raytraverse import draw
import numpy as np
from scipy import stats


# def test_version():
#     # with craytraverse.ostream_redirect():
#     #     craytraverse.rtrace("rtrace -defaults".split())
#     print(craytraverse.version())
#
#     # print(craytraverse.__doc__)
#     # print(craytraverse.rtdefaults())


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


def test_get_detail():
    ans = np.array([3, 2.5, 1.5, 0., 0., 0., 0., 0., 2.5, 5, 2., 1.5, 0., 0.,
                    0., 0., 1.5, 2.,5, 2., 1.5, 0., 0., 0., 0., 1.5, 2., 5, 2.,
                    1.5, 0., 0., 0., 0., 1.5, 2., 5,  2., 1.5, 0., 0., 0., 0.,
                    1.5, 2., 5, 2., 1.5, 0., 0., 0., 0., 1.5, 2., 5, 2.5, 0.,
                    0., 0., 0., 0., 1.5, 2.5, 3])
    ar = np.eye(8)
    d = draw.get_detail(ar*.5, (0, 1))*2
    assert np.allclose(ans, d)
