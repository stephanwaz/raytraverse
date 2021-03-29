# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import functools

import numpy as np
from scipy import stats

from raytraverse import translate
from raytraverse.evaluate.positionindex import PositionIndex
from raytraverse.evaluate.basemetricset import BaseMetricSet
from raytraverse.mapper import ViewMapper


class Ray:

    def __init__(self, xyz):
        self.xyz = np.asarray(xyz).ravel()[0:3]
        self.mag = np.linalg.norm(self.xyz)
        self.unit = self.xyz / self.mag

    def __repr__(self):
        vout = "{:.04f} {:.04f} {:.04f}".format(*self.unit)
        return f"<Ray vec: [{vout}] mag: {self.mag:.06g}>"

    @property
    @functools.lru_cache(1)
    def tp(self):
        """overall vector (with magnitude)"""
        return translate.xyz2tp(self.unit).ravel()

    @property
    @functools.lru_cache(1)
    def aa(self):
        """overall vector (with magnitude)"""
        return translate.xyz2aa(self.unit).ravel()


class FieldMetric(BaseMetricSet):
    """calculate metrics on full spherical point clouds rather than view based
    metrics.

    Parameters
    ----------
    vec: np.array
        (N, 3) directions of all rays
    omega: np.array
        (N,) solid angle of all rays
    lum: np.array
        (N,) luminance of all rays (multiplied by "scale")
    metricset: list, optional
        keys of metrics to return, same as property names
    scale: float, optional
        scalefactor for luminance
    npts: int, optional
        for equatorial metrics, the number of points to interpolate
    close: bool, optional
        include npts+1 duplicate to draw closed curve
    sigma: float, optional
        scale parameter of gaussian for kernel estimated metrics
    omega_as_view_area: bool, optional
        set to true when vectors either represent a whole sphere or a subset
        that does not match the viewmapper. if False, corrects boundary omega
        to properly trim to correct size.
    kwargs:
        additional arguments that may be required by additional properties
    """

    def __init__(self, vec, omega, lum, vm=None, scale=1., npts=360, close=True,
                 sigma=.05, omega_as_view_area=True, **kwargs):
        if vm is None:
            vm = ViewMapper((0, 0, 1))
        self.npts = npts
        self.close = close
        self.sigma = sigma
        super().__init__(vec, omega, lum, vm, scale=scale,
                         omega_as_view_area=omega_as_view_area, **kwargs)

    # -----------------dependencies-----------------

    @property
    @functools.lru_cache(1)
    def tp(self):
        """vectors in spherical coordinates"""
        return translate.xyz2tp(self.vec)

    @property
    @functools.lru_cache(1)
    def phi(self):
        """interpolated output phi values"""
        nx = np.linspace(0, 2*np.pi, self.npts + 1)
        if not self.close:
            nx = nx[:-1]
        return nx

    @property
    @functools.lru_cache(1)
    def eq_xyz(self):
        """interpolated output xyz vectors"""
        theta = np.full(self.phi.shape, np.pi/2)
        return translate.tp2xyz(np.stack((theta, self.phi)).T)

    # -------------------------single Ray metrics---------------------------

    @property
    @functools.lru_cache(1)
    def avg(self):
        """overall vector (with magnitude)"""
        return Ray(np.einsum('ij,i,i->j', self.vec, self.lum, self.omega) *
                   self.scale/self.view_area)

    @property
    @functools.lru_cache(1)
    def peak(self):
        """overall vector (with magnitude)"""
        i = np.argmax(self.lum)
        return Ray(self.vec[i] * self.lum[i])

    # ----------equatorial metrics (where vm.xyz is the north pole-------------

    @property
    @functools.lru_cache(1)
    def eq_lum(self):
        """luminance along an interpolated equator with a bandwidth=sigma"""
        weights = np.sin(self.tp[:, 0]) * self.omega
        x = (np.mod(self.tp[:, 1], 2*np.pi) - np.pi).reshape(-1)
        nx = (np.mod(self.phi, 2*np.pi) - np.pi).reshape(-1, 1)
        n = stats.norm(scale=self.sigma)

        def polar_kernel(c):
            ang = np.abs(c - x)
            ang = np.where(ang > np.pi, 2*np.pi - ang, ang)
            nweight = n.pdf(ang)
            return np.average(self.lum, weights=nweight*weights)

        return np.apply_along_axis(polar_kernel, 1, nx) * self.scale

    @property
    @functools.lru_cache(1)
    def eq_density(self):
        """ray density along an interpolated equator"""
        extended = np.concatenate((self.tp[:, 1] - 2*np.pi, self.tp[:, 1],
                                   self.tp[:, 1] + 2*np.pi))
        k = stats.gaussian_kde(extended, bw_method=self.sigma/3)
        return k(self.phi) * 3

    @property
    @functools.lru_cache(1)
    def eq_illum(self):
        """illuminiance along an interpolated equator"""
        ctheta = np.maximum(np.einsum("ki,ji->kj", self.eq_xyz, self.vec), 0)
        return np.einsum('kj,j,j->k', ctheta, self.lum, self.omega)*self.scale

    @property
    @functools.lru_cache(1)
    def eq_gcr(self):
        """cosine weighted gcr along an interpolated equator"""
        ctheta = np.maximum(np.einsum("ki,ji->kj", self.eq_xyz, self.vec), 0)
        a2lum = np.einsum('kj,j,j,j->k', ctheta, self.lum, self.lum, self.omega)
        return a2lum/np.square(self.eq_illum/self.scale)

    @property
    @functools.lru_cache(1)
    def eq_loggc(self):
        ev = self.eq_illum
        slum = self.sources[:, -1]
        soga = self.sources[:, -2]
        pos = PositionIndex().positions_vec(self.eq_xyz, self.sources[:, 0:3])
        pwsl2 = np.sum(np.square(slum[None])*soga[None] / np.square(pos), axis=1)
        return np.log10(1 + pwsl2/ev**1.87)

    @property
    @functools.lru_cache(1)
    def eq_dgp(self):
        ev = self.eq_illum
        ll = np.where(ev < 1000,
                      np.exp(0.024*ev - 4)/(1 + np.exp(0.024*ev - 4)), 1)
        t2 = 9.18 * 10**-2 * self.eq_loggc
        return np.minimum(ll*(5.87 * 10**-5 * ev + t2 + 0.16), 1.0)
