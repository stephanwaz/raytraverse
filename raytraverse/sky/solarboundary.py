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


class SolarBoundary(object):
    """sky location data object

    Parameters
    ----------
    loc: tuple
        lat, lon, tz (in degrees, west is positive
    skyro: float
        sky rotation (in degrees, ccw)
    """

    def __init__(self, loc, skyro=0.0):
        #: float: ccw rotation (in degrees) for sky
        self.skyro = skyro
        self.loc = loc

    @property
    def solarbounds(self):
        """read only extent of solar bounds for given location
        set via loc

        :getter: Returns solar bounds
        :type: (np.array, np.array)
        """
        return self._solarbounds

    @property
    def loc(self):
        """scene location

        :getter: Returns location
        :setter: Sets location and self.solarbounds
        :type: (float, float, int)
        """
        return self._loc

    @loc.setter
    def loc(self, loc):
        """
        generate UV coordinates for jun 21 and dec 21 to use for masking
        sky positions
        """
        self._loc = loc
        if loc is None:
            self._solarbounds = None
        else:
            jun = np.arange('2020-06-21', '2020-06-22', 5,
                            dtype='datetime64[m]')
            dec = np.arange('2020-12-21', '2020-12-22', 5,
                            dtype='datetime64[m]')
            jxyz = skycalc.sunpos_xyz(jun, *loc)
            dxyz = skycalc.sunpos_xyz(dec, *loc)
            juv = translate.xyz2uv(jxyz[jxyz[:, 2] > 0], flipu=False)
            duv = translate.xyz2uv(dxyz[dxyz[:, 2] > 0], flipu=False)
            juv = juv[juv[:, 0].argsort()]
            duv = duv[duv[:, 0].argsort()]
            self._solarbounds = (juv, duv)

    def in_solarbounds(self, uv, size=0.0):
        """
        for checking if src direction is in solar transit

        Parameters
        ----------
        uv: np.array
            source directions
        size: float
            offset around UV to test

        Returns
        -------
        result: np.array
            Truth of ray.src within solar transit
        """
        if self.loc is None:
            return np.full(uv.shape[0], True)
        if abs(self.skyro) > 0:
            xyz = translate.uv2xyz(uv, xsign=1)
            rxyz = translate.rotate_elem(xyz, -self.skyro)
            uv = translate.xyz2uv(rxyz, flipu=False)
        o = size/2
        juv, duv = self.solarbounds
        vlowleft = duv[np.searchsorted(duv[:, 0], uv[:, 0] - o) - 1]
        vlowright = duv[np.searchsorted(duv[:, 0], uv[:, 0] - o) - 1]
        vupleft = juv[np.searchsorted(juv[:, 0], uv[:, 0] + o) - 1]
        vupright = juv[np.searchsorted(juv[:, 0], uv[:, 0] + o) - 1]
        inbounds = np.stack((vlowleft[:, 1] <= uv[:, 1] - o,
                             vlowright[:, 1] <= uv[:, 1] - o,
                             uv[:, 1] + o <= vupleft[:, 1],
                             uv[:, 1] + o <= vupright[:, 1]))
        return np.all(inbounds, 0)
