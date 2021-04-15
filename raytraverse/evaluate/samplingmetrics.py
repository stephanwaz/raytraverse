# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
import functools

from raytraverse import translate
from raytraverse.evaluate.basemetricset import BaseMetricSet


class SamplingMetrics(BaseMetricSet):
    """default metricset for areasampler
    """

    #: available metrics (and the default return set)
    defaultmetrics = ['avglum', 'loggcr', 'xpeak', 'ypeak']

    allmetrics = defaultmetrics

    def __init__(self, vec, omega, lum, vm, scale=1., peakthreshold=0.0,
                 lmin=0, **kwargs):
        kwargs.update(peakthreshold=peakthreshold)
        lum = np.maximum(lum, lmin)
        super().__init__(vec, omega, lum, vm, scale=scale, **kwargs)

    @property
    @functools.lru_cache(1)
    def peakvec(self):
        """average vector (with magnitude) for peak rays"""
        mxlum = np.max(self.lum)
        t = max(mxlum * .9, self.kwargs["peakthreshold"])
        nmax = self.lum > t
        vec = np.einsum('ij,i,i->j', self.vec[nmax], self.lum[nmax],
                        self.omega[nmax])
        return translate.norm1(vec)

    @property
    @functools.lru_cache(1)
    def xpeak(self):
        """x-component of avgvec as positive number (in range 0-2)"""
        x = self.peakvec[0] + 1
        if np.isnan(x):
            return 1.0
        else:
            return x

    @property
    @functools.lru_cache(1)
    def ypeak(self):
        """y-component of avgvec as positive number (in range 0-2)"""
        x = self.peakvec[1] + 1
        if np.isnan(x):
            return 1.0
        else:
            return x

    @property
    @functools.lru_cache(1)
    def loggcr(self):
        """log of global contrast ratio"""
        return np.log(self.gcr)
