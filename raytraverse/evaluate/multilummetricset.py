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


class MultiLumMetricSet(BaseMetricSet):
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
        (N, M) luminance of all rays in view (multiplied by "scale")
    metricset: list, optional
        keys of metrics to return, same as property names
    scale: float, optional
        scalefactor for luminance
    kwargs:
        additional arguments that may be required by additional properties
    """

    def __init__(self, vec, omega, lum, vm, metricset=None, scale=179.,
                 omega_as_view_area=True, **kwargs):
        super().__init__(vec, omega, lum, vm, metricset=metricset, scale=scale,
                         omega_as_view_area=omega_as_view_area, **kwargs)
        self._lum = self._lum.reshape(self.vec.shape[0], -1)

    # -----------------metric functions (return array)-----------------

    @property
    @functools.lru_cache(1)
    def illum(self):
        """illuminance"""
        return np.squeeze(np.einsum('i,ij,i->j', self.ctheta, self.lum,
                          self.omega) * self.scale)

    @property
    @functools.lru_cache(1)
    def avglum(self):
        """average luminance"""
        return np.squeeze(np.einsum('ij,i->j', self.lum, self.omega) *
                          self.scale/self.view_area)

    @property
    @functools.lru_cache(1)
    def avgraylum(self):
        """average luminance (not weighted by omega)"""
        return np.squeeze(np.average(self.lum, axis=0) * self.scale)

    @property
    @functools.lru_cache(1)
    def gcr(self):
        """a unitless measure of relative contrast defined as the average of
        the squared luminances divided by the average luminance squared"""
        a2lum = (np.einsum('ij,ij,i->j', self.lum, self.lum, self.omega) *
                 self.scale**2/self.view_area)
        gcr = np.squeeze(a2lum/self.avglum**2)
        return np.where(np.isnan(gcr), 1, gcr)

