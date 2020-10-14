# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
import functools

from raytraverse import translate
from raytraverse.integrator.positionindex import PositionIndex


class MetricSet(object):

    def __init__(self, vm, vec, omega, lum, metricset=None, scale=179.,
                 threshold=2000., guth=True, tradius=30.0):
        if metricset is None or len(metricset) == 0:
            metricset = MetricSet.allmetrics
        self.vm = vm
        self._vec = translate.norm(vec)
        self._lum = lum
        self._omega = omega
        self.scale = scale
        self._threshold = threshold
        self.guth = guth
        self.tradius = tradius
        self.metrics = metricset
        for m in self.metrics:
            if m not in MetricSet.allmetrics:
                raise AttributeError(f"'{m}' is not defined in MetricSet")

    def __call__(self):
        """
        Returns
        -------
        result: np.array
            list of computed metrics

        """
        return np.array([getattr(self, m) for m in self.metrics])

    @property
    def vec(self):
        return self._vec

    @property
    def lum(self):
        return self._lum

    @property
    def omega(self):
        return self._omega

    # -------------------metric dependencies (return array)--------------------

    @property
    @functools.lru_cache(1)
    def src_mask(self):
        """boolean mask for filtering source/background rays"""
        return self.lum * self.scale > self.threshold

    @property
    @functools.lru_cache(1)
    def task_mask(self):
        return self.vm.degrees(self.vec) < self.tradius

    @property
    @functools.lru_cache(1)
    def sources(self):
        """vec, omega, lum of rays above threshold"""
        m = self.src_mask
        vec = self.vec[m]
        lum = self.lum[m]
        oga = self.omega[m]
        return vec, oga, lum

    @property
    @functools.lru_cache(1)
    def background(self):
        """vec, omega, lum of rays below threshold"""
        m = np.logical_not(self.src_mask)
        vec = self.vec[m]
        lum = self.lum[m]
        oga = self.omega[m]
        return vec, oga, lum

    @property
    @functools.lru_cache(1)
    def source_pos_idx(self):
        svec, _, _ = self.sources
        return PositionIndex(self.guth).positions(self.vm, svec)

    # -----------------metric functions (return single value)-----------------

    allmetrics = ["illum", "avglum", "lum2", "ugr", "dgp", "dgp_t1", "dgp_t2",
                  "tasklum", "backlum", "threshold", "pwsl2", "view_area"]

    @property
    @functools.lru_cache(1)
    def view_area(self):
        """total solid angle of vectors"""
        return np.sum(self.omega)

    @property
    @functools.lru_cache(1)
    def threshold(self):
        """threshold for glaresource/background similar behavior to evalglare
        '-b' paramenter"""
        if self._threshold > 100:
            return self._threshold
        else:
            return self.tasklum * self._threshold

    @property
    @functools.lru_cache(1)
    def pwsl2(self):
        """position weighted source luminance squared, used by dgp, ugr, etc
        sum(Ls^2*omega/Ps^2)"""
        _, soga, slum = self.sources
        return np.sum(np.square(slum)*soga*self.scale**2 /
                      np.square(self.source_pos_idx))

    @property
    @functools.lru_cache(1)
    def backlum(self):
        """average background luminance"""
        bvec, boga, blum = self.background
        return np.einsum('i,i->', blum, boga)*self.scale/np.sum(boga)

    @property
    @functools.lru_cache(1)
    def tasklum(self):
        """average task luminance"""
        lum = self.lum[self.task_mask]
        oga = self.omega[self.task_mask]
        return np.einsum('i,i->', lum, oga)*self.scale/np.sum(oga)

    @property
    @functools.lru_cache(1)
    def illum(self):
        """illuminance"""
        return np.einsum('i,i,i->', self.vm.ctheta(self.vec), self.lum,
                         self.omega)*self.scale

    @property
    @functools.lru_cache(1)
    def avglum(self):
        """average luminance"""
        return (np.einsum('i,i->', self.lum, self.omega) *
                self.scale/self.view_area)

    @property
    @functools.lru_cache(1)
    def lum2(self):
        """a unitless measure of relative contrast defined as the average of
        the squared luminances divided by the average luminance squared"""
        a2lum = (np.einsum('i,i,i->', self.lum, self.lum, self.omega) *
                 self.scale**2/self.view_area)
        return a2lum/self.avglum**2

    @property
    @functools.lru_cache(1)
    def dgp(self):
        return np.minimum(self.dgp_t1 + self.dgp_t2 + 0.16, 1.0)

    @property
    @functools.lru_cache(1)
    def dgp_t1(self):
        return 5.87 * 10**-5 * self.illum

    @property
    @functools.lru_cache(1)
    def dgp_t2(self):
        return 9.18 * 10**-2 * np.log10(1 + self.pwsl2 / self.illum**1.87)

    @property
    @functools.lru_cache(1)
    def ugr(self):
        return np.maximum(0, 8 * np.log10(0.25 * self.pwsl2 / self.backlum))
