# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import sys
import numpy as np
import functools

from raytraverse import translate
from raytraverse.evaluate.positionindex import PositionIndex


class MetricSet(object):
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
    defaultmetrics = ["illum", "avglum", "gcr", "ugp", "dgp"]

    allmetrics = defaultmetrics + ["tasklum", "backlum", "dgp_t1", "log_gc",
                                   "dgp_t2", "ugr", "threshold", "pwsl2",
                                   "view_area", "density", "reldensity",
                                   "lumcenter", "avgraylum", "backlum_true",
                                   "srcillum"]

    def __init__(self, vm, vec, omega, lum, metricset=None, scale=179.,
                 threshold=2000., guth=True, tradius=30.0, **kwargs):
        if metricset is not None:
            self.check_metrics(metricset, True)
            self.defaultmetrics = metricset
        self.vm = vm
        self.view_area = vm.area
        v = translate.norm(vec)
        mask = self.vm.in_view(v)
        self._vec = v[mask]
        self._lum = lum[mask]
        self.omega = omega[mask]
        self.scale = scale
        self._threshold = threshold
        self.guth = guth
        self.tradius = tradius
        self.kwargs = kwargs

    def __call__(self, metrics=None):
        """
        Returns
        -------
        result: np.array
            list of computed metrics

        """
        if metrics is None:
            metrics = self.defaultmetrics
        else:
            self.check_metrics(metrics, True)
        return np.array([getattr(self, m) for m in metrics])

    @staticmethod
    def check_metrics(metrics, raise_error=False):
        """returns list of valid metric names from argument
        if raise_error is True, raises an Atrribute Error"""
        good = [m for m in metrics if m in MetricSet.allmetrics]
        if raise_error and len(good) != len(metrics):
            bad = [m for m in metrics if not m in MetricSet.allmetrics]
            raise AttributeError(f"'{bad}' are not defined in "
                                 f"MetricSet: {MetricSet.allmetrics}")
        return good

    @property
    def vec(self):
        return self._vec

    @property
    def lum(self):
        return self._lum

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, og):
        """correct omega of rays at edge of view to normalize view size"""
        self._omega = np.copy(og)
        # square appoximation of ray area
        ray_side = np.sqrt(self._omega)
        excess = np.sum(og) - self.view_area
        if abs(excess) > .1:
            print(f"Warning, large discrepancy between sum(omega) and view "
                  f"area: {excess}", file=sys.stderr)
        while True:
            # get ray squares that overlap edge of view
            onedge = self.radians > (self.vm.viewangle * np.pi / 360 - ray_side)
            edgetotal = np.sum(og[onedge])
            adjust = 1 - excess/edgetotal
            # if this fails increase search radius to ensure enough rays to
            # absorb the adjustment
            if adjust < 0:
                print("Warning, intitial search radius failed for omega "
                      "adjustment", file=sys.stderr)
                ray_side *= 1.1
            else:
                break
        self._omega[onedge] = og[onedge] * adjust

    # -------------------metric dependencies (return array)--------------------

    @property
    @functools.lru_cache(1)
    def ctheta(self):
        """cos angle between ray and view"""
        # """cos angle between ray and view
        #         with linear interpolation across omega"""
        # radius = np.sqrt(self._omega/np.pi)
        # return .5*(np.cos(self.radians + radius) +
        #            np.cos(self.radians - radius))
        return self.vm.ctheta(self.vec)

    @property
    @functools.lru_cache(1)
    def radians(self):
        """cos angle between ray and view"""
        # return np.arccos(self.vm.ctheta(self.vec))
        rad = np.arccos(self.ctheta)
        return rad

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
        """average background luminance"""
        svec, soga, slum = self.sources
        return np.einsum('i,i,i->', self.vm.ctheta(svec), slum,
                         soga) * self.scale

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
    def illum(self):
        """illuminance"""
        return np.einsum('i,i,i->', self.ctheta, self.lum,
                         self.omega) * self.scale

    @property
    @functools.lru_cache(1)
    def avglum(self):
        """average luminance"""
        return (np.einsum('i,i->', self.lum, self.omega) *
                self.scale/self.view_area)

    @property
    @functools.lru_cache(1)
    def avgraylum(self):
        """average luminance (not weighted by omega"""
        return np.average(self.lum) * self.scale

    @property
    @functools.lru_cache(1)
    def gcr(self):
        """a unitless measure of relative contrast defined as the average of
        the squared luminances divided by the average luminance squared"""
        a2lum = (np.einsum('i,i,i->', self.lum, self.lum, self.omega) *
                 self.scale**2/self.view_area)
        return a2lum/self.avglum**2

    @property
    @functools.lru_cache(1)
    def dgp(self):
        ll = 1
        if self.illum < 1000:
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
        return np.maximum(0, 8 * np.log10(0.25 * self.pwsl2 / self.backlum))

    @property
    @functools.lru_cache(1)
    def ugp(self):
        return self.ugr * .0325

    @property
    @functools.lru_cache(1)
    def density(self):
        return self.omega.size / self.view_area

    @property
    @functools.lru_cache(1)
    def reldensity(self):
        try:
            avgdensity = self.kwargs['lfcnt']/self.kwargs['lfang']
        except KeyError:
            return 0.0
        else:
            return self.density/avgdensity

    @property
    @functools.lru_cache(1)
    def lumcenter(self):
        try:
            return np.average(self.radians,
                              weights=self.lum*self.omega) * 180/np.pi
        except ZeroDivisionError:
            return 0.0
