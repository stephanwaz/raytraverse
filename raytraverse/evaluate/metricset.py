# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
import functools

from raytraverse.evaluate.basemetricset import BaseMetricSet


class MetricSet(BaseMetricSet):
    """object for calculating metrics based on a view direction, and rays
    consisting on direction, solid angle and luminance information

    by encapsulating these calculations within a class, metrics with redundant
    calculations can take advantage of cached results, for example dgp does
    not need to recalculate illuminance when it has been directly requested.
    all metrics can be accessed as properties (and are calculated just in time)
    or the object can be called (no arguments) to return a np.array of all
    metrics defined in "metricset"

    Parameters
    ----------
    vm: raytraverse.mapper.ViewMapper
        the view direction
    vec: np.array
        (N, 3) directions of all rays in view
    omega: np.array
        (N,) solid angle of all rays in view
    lum: np.array
        (N,) luminance of all rays in view (multiplied by "scale")
    metricset: list, optional
        keys of metrics to return, same as property names
    scale: float, optional
        scalefactor for luminance
    threshold: float, optional
        threshold for glaresource/background similar behavior to evalglare '-b'
        paramenter. if greater than 100 used as a fixed luminance threshold.
        otherwise used as a factor times the task luminance (defined by
        'tradius')
    guth: bool, optional
        if True, use Guth for the upper field of view and iwata for the lower
        if False, use Kim
    tradius: float, optional
        radius in degrees for task luminance calculation
    kwargs:
        additional arguments that may be required by additional properties
    """

    #: available metrics (and the default return set)
    defaultmetrics = BaseMetricSet.defaultmetrics + ["ugp", "dgp"]

    allmetrics = BaseMetricSet.allmetrics + ["ugp", "dgp", "tasklum", "backlum",
                                             "dgp_t1", "log_gc", "dgp_t2",
                                             "ugr", "threshold", "pwsl2",
                                             "view_area", "backlum_true",
                                             "srcillum", "srcarea", "maxlum"]

    def __init__(self, vec, omega, lum, vm, metricset=None, scale=179.,
                 threshold=2000., guth=True, tradius=30.0,
                 omega_as_view_area=False, lowlight=False, **kwargs):
        super().__init__(vec, omega, lum, vm, metricset=metricset, scale=scale,
                         omega_as_view_area=omega_as_view_area, guth=guth, **kwargs)
        self._lowlight = lowlight
        self._threshold = threshold
        self.tradius = tradius

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
        return self.pos_idx[self.src_mask]

    # -----------------metric functions (return single value)-----------------

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
    def srcillum(self):
        """source illuminance"""
        svec, soga, slum = self.sources
        return np.einsum('i,i,i->', self.vm.ctheta(svec), slum,
                         soga) * self.scale

    @property
    @functools.lru_cache(1)
    def srcarea(self):
        """total source area"""
        _, soga, _ = self.sources
        return np.sum(soga)

    @property
    @functools.lru_cache(1)
    def maxlum(self):
        """peak luminance"""
        if self.lum.size > 0:
            return np.max(self.lum)*self.scale
        else:
            return 0.0

    @property
    @functools.lru_cache(1)
    def backlum(self):
        """average background luminance CIE estimate
        (official for some metrics)"""
        return (self.illum - self.srcillum) / np.pi

    @property
    @functools.lru_cache(1)
    def backlum_true(self):
        """average background luminance mathematical"""
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
    def dgp(self):
        ll = 1
        if self._lowlight and self.illum < 500:
            ll = np.exp(0.024*self.illum - 4)/(1 + np.exp(0.024*self.illum - 4))
        return np.minimum(ll*(self.dgp_t1 + self.dgp_t2 + 0.16), 1.0)

    @property
    @functools.lru_cache(1)
    def dgp_t1(self):
        return 5.87 * 10**-5 * self.illum

    @property
    @functools.lru_cache(1)
    def log_gc(self):
        return np.log10(1 + self.pwsl2/self.illum**1.87)

    @property
    @functools.lru_cache(1)
    def dgp_t2(self):
        return 9.18 * 10**-2 * self.log_gc

    @property
    @functools.lru_cache(1)
    def ugr(self):
        with np.errstate(divide='ignore'):
            ug = np.maximum(0, 8 * np.log10(0.25 * self.pwsl2 / self.backlum))
        return ug

    @property
    @functools.lru_cache(1)
    def ugp(self):
        """http://dx.doi.org/10.1016/j.buildenv.2016.08.005"""
        return (1 + 2/7 * 10**(-(self.ugr + 5)/40))**-10
