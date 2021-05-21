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


class BaseMetricSet(object):
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
    kwargs:
        additional arguments that may be required by additional properties
    """

    #: available metrics (and the default return set)
    defaultmetrics = ["illum", "avglum", "gcr"]

    allmetrics = defaultmetrics + ["density", "avgraylum"]

    safe2sum = {"illum", "avglum"}

    def __init__(self, vec, omega, lum, vm, metricset=None, scale=179.,
                 omega_as_view_area=True, **kwargs):
        if metricset is not None:
            self.check_metrics(metricset, True)
            self.defaultmetrics = metricset
        self.vm = vm
        self.view_area = vm.area
        self._correct_omega = not omega_as_view_area
        v = translate.norm(vec)
        if self.vm.aspect == 2:
            self._vec = v
            self._lum = lum
            self.omega = omega
        else:
            mask = self.vm.in_view(v)
            self._vec = v[mask]
            try:
                self._lum = lum[mask]
            except IndexError:
                print(v, lum, omega)
                self._lum = np.zeros(len(self._vec))
            self.omega = omega[mask]
        self.scale = scale
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

    @classmethod
    def check_metrics(cls, metrics, raise_error=False):
        """returns list of valid metric names from argument
        if raise_error is True, raises an Atrribute Error"""
        good = [m for m in metrics if m in cls.allmetrics]
        if raise_error and len(good) != len(list(metrics)):
            bad = [m for m in metrics if m not in cls.allmetrics]
            raise AttributeError(f"'{bad}' are not defined in "
                                 f"MetricSet: {cls.allmetrics}")
        return good

    @classmethod
    def check_safe2sum(cls, metrics):
        """checks if list if metrics is safe to compute for seperate
        sources before adding"""
        mset = set(metrics)
        return mset.issubset(cls.safe2sum)

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
        if self._correct_omega and len(self.vec) > 100:
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
        else:
            self._omega = og
            self.view_area = np.sum(og)

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

    # -----------------metric functions (return single value)-----------------

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
        if self.avglum > 0:
            return a2lum/self.avglum**2
        else:
            return 1.0

    @property
    @functools.lru_cache(1)
    def density(self):
        return self.omega.size / self.view_area
