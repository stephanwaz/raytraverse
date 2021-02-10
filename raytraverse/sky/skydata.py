# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from raytraverse import translate
from raytraverse.sky import skycalc


class SkyData(object):
    """class to generate sky conditions

    This class provides an interface to generate sky data using the perez sky
    model

    Parameters
    ----------
    wea: str np.array
        path to epw, wea, or .npy file or np.array, if loc not set attempts to
        extract location data (if needed). The Integrator does not need to be
        initialized with weather data but for convinience can be. However,
        self.skydata must be initialized (directly or through self.sky) before
        calling integrate.
    suns: raytraverse.sky.Suns, optional
    loc: (float, float, int), optional
        location data given as lat, lon, mer with + west of prime meridian
        overrides location data in wea (but not in sunfield)
    skyro: float, optional
        angle in degrees counter-clockwise to rotate sky
        (to correct model north, equivalent to clockwise rotation of scene)
        does not override rotation in SunField)
    ground_fac: float, optional
        ground reflectance
    skyres: float, optional
        approximate square patch size in degrees
    """

    def __init__(self, wea, suns=None, loc=None, skyro=0.0, ground_fac=0.15,
                 skyres=10.0):
        self.skyres = skyres
        self.suns = suns
        self.ground_fac = ground_fac
        if loc is None and wea is not None:
            try:
                loc = skycalc.get_loc_epw(wea)
            except ValueError:
                pass
        #: location and sky rotation information
        self._loc = loc
        self._skyro = skyro
        self.skydata = wea
        self._proxysort = np.argsort(self.sunproxy[:, 1], kind='stable')
        self._invsort = np.argsort(self.proxysort, kind='stable')

    @property
    def skyres(self):
        return self._skyres

    @skyres.setter
    def skyres(self, s):
        self._skyres = int(np.floor(90/s)*2)

    @property
    def skyro(self):
        """sky rotation (in degrees, ccw)"""
        return self._skyro

    @property
    def loc(self):
        """lot, lon, mer (in degrees, west is positive)"""
        return self._loc

    @property
    def smtx(self):
        """shape (np.sum(daysteps), skyres**2 + 1) coefficients for each sky
        patch each row is a timestep, coefficients exclude sun"""
        return self._smtx

    @property
    def sun(self):
        """shape (np.sum(daysteps), 5) sun position (index 0,1,2) and
        coefficients for sun at each timestep assuming the true solid angle of
        the sun (index 3) and the weighted value for the sky patch (index 4)."""
        return self._sun

    @property
    def daysteps(self):
        """shape (len(skydata),) boolean array masking timesteps when sun is
        below horizon"""
        return self._daysteps

    @property
    def skydata(self):
        """sun position and dirnorm diffhoriz"""
        return self._skydata

    @property
    def sunproxy(self):
        """array of sun proxy data
        shape (len(daysteps), 2). column 0 is the corresponding sky bin
        (column of smtx), column 1 is the row of self.suns
        """
        return self._sunproxy

    @property
    def proxysort(self):
        """sorting indices to arange daystep axis by solar proxy
        this is useful when combining sky/sun kdtrees without writing to disk
        to only do the interpolation once for a set of sky conditions."""
        return self._proxysort

    @property
    def invsort(self):
        """reverse sorting indices to restore input daystep order"""
        return self._invsort

    @property
    def serr(self):
        """the error (in degrees) between the actual sun position and the
        applied sunproxy"""
        return self._serr

    @skydata.setter
    def skydata(self, wea):
        """calculate sky matrix data and store in self._skydata

        Parameters
        ----------
        wea: str, np.array
            This method takes either a file path or np.array. File path can
            point to a wea, epw, or .npy file. Loaded array must be one of
            the following:
            - 4 col: alt, az, dir, diff
            - 5 col: dx, dy, dz, dir, diff
            - 5 col: m, d, h, dir, diff
        """
        skydata = self._format_skydata(wea)
        daysteps = skydata[:, 2] + skydata[:, 3] > 0
        sxyz = skydata[daysteps, 0:3]
        dirdif = skydata[daysteps, 3:]
        smtx, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, self.skyres,
                                          ground_fac=self.ground_fac)
        # ratio between actual solar disc and patch
        omegar = np.square(0.2665 * np.pi * self.skyres / 180) * .5
        plum = sun[:, -1] * omegar
        sun = np.hstack((sun, plum[:, None]))
        smtx = np.hstack((smtx, grnd[:, None]))
        self._smtx = smtx
        self._sun = sun
        self._daysteps = daysteps
        self._skydata = skydata
        self.sunproxy = sxyz

    @sunproxy.setter
    def sunproxy(self, sxyz):
        sunbins = translate.xyz2skybin(sxyz, self.skyres)
        try:
            si, serrs = self.suns.proxy_src(sxyz,
                                            tol=180*2**.5/self.suns.sunres)
        except AttributeError:
            si = [0] * len(sunbins)
            serrs = sunbins
        self._serr = serrs
        self._sunproxy = np.stack((sunbins, si)).T

    def smtx_patch_sun(self):
        """generate smtx with solar energy applied to proxy patch
        for directly applying to skysampler data (without direct sun components
        can also be used in a partial mode (with sun view / without sun
        reflection."""
        wsun = np.copy(self.smtx)
        r = range(wsun.shape[0])
        wsun[range(wsun.shape[0]), self.sunproxy[:, 0]] += self.sun[r, 4]
        return wsun

    def header(self):
        """generate image header string"""
        try:
            hdr = "LOCATION= lat: {} lon: {} tz: {} ro: {}".format(*self.loc,
                                                                   self.skyro)
        except TypeError:
            hdr = "LOCATION= None"
        return hdr

    def _format_skydata(self, dat):
        """process dat argument as skydata

        see sky.setter for details on argument

        Returns
        -------
        np.array
            dx, dy, dz, dir, diff
        """
        loc = self._loc
        if hasattr(dat, 'shape'):
            skydat = dat
        else:
            try:
                skydat = np.loadtxt(dat)
            except ValueError:
                if loc is None:
                    loc = skycalc.get_loc_epw(dat)
                skydat = skycalc.read_epw(dat)
        skydat = np.atleast_2d(skydat)
        if skydat.shape[1] == 4:
            xyz = translate.aa2xyz(skydat[:, 0:2])
            skydat = np.hstack((xyz, skydat[:, 2:]))
        elif skydat.shape[1] == 5:
            if np.max(skydat[:, 2]) > 2:
                if loc is None:
                    raise ValueError("cannot parse wea data without a Location")
                times = skycalc.row_2_datetime64(skydat[:, 0:3])
                xyz = skycalc.sunpos_xyz(times, *loc, ro=self._skyro)
                skydat = np.hstack((xyz, skydat[:, 3:]))
        else:
            raise ValueError('input data should be one of the following:'
                             '\n4 col: alt, az, dir, diff'
                             '\n5 col: dx, dy, dz, dir, diff'
                             '\n5 col: m, d, h, dir, diff')
        return skydat

