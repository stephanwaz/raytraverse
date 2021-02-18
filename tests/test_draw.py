#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.draw"""

import pytest
from raytraverse.sampler import draw, Sampler
import numpy as np
from scipy import stats


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
    ar = np.eye(8)
    ans = np.eye(8)
    ans[1:-1, 1:-1] *= 2
    ans[0:-1, 1:] += ar[1:,1:]
    ans[1:, 0:-1] += ar[1:, 1:]
    ans[(0, 1, -2, -1), (1,0, -1, -2)] += .5
    d = draw.get_detail(ar, *Sampler.filters['wav3']).reshape(8, 8)
    assert np.allclose(ans, d)
