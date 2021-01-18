# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from raytraverse.sky.solarboundary import SolarBoundary
from raytraverse.sky.suns import Suns


class SunsLoc(Suns):
    """select suns to sample based on sky pdf, scene, and location.

    Parameters
    ----------
    scene: str,
        path of scene
    loc: tuple
        lat, lon, tz (in degrees, west is positive)
    srct: float, optional
        threshold of sky contribution for determining appropriate srcn
    skyro: float, optional
        sky rotation (in degrees, ccw)
    reload: bool
        if True reloads existing sun positions, else always generates new
    """

    def __init__(self, scene, loc, skyro=0.0, **kwargs):
        #: raytraverse.sky.SolarBoundary
        self.sky = SolarBoundary(loc, skyro)
        super().__init__(scene, skyro=skyro, **kwargs)

    def choose_suns(self):
        si = np.stack(np.unravel_index(np.arange(self.sunres**2),
                                       (self.sunres, self.sunres)))
        uv = si.T/self.sunres
        ib = self.sky.in_solarbounds(uv + .5/self.sunres)
        xyz = self._jitter_suns(si.T[ib])
        return xyz

