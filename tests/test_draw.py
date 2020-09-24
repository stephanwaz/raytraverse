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


