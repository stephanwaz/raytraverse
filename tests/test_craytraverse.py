#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.craytraverse"""
from raytraverse.sampler import draw
import numpy as np
from scipy import stats


def test_from_pdf():
    rv = stats.norm()
    nsamp = 500000
    t = .01
    x = np.linspace(0, 5, nsamp)
    pdf = rv.pdf(x)
    exp = np.sum(pdf > t)
    c = draw.from_pdf(pdf, t)
    assert np.isclose(c.size, exp)
