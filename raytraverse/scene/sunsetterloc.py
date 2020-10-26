# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate
from raytraverse.scene.skyinfo import SkyInfo
from raytraverse.scene.sunsetter import SunSetter


class SunSetterLoc(SunSetter):
    """select suns to sample based on sky pdf, scene, and location.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
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
        #: raytraverse.scene.SkyInfo
        self.sky = SkyInfo(loc, skyro)
        super().__init__(scene, skyro=skyro, **kwargs)

    def choose_suns(self):
        uvsize = self.sunres
        si = np.stack(np.unravel_index(np.arange(uvsize**2),
                                       (uvsize, uvsize)))
        skyb = self.load_sky_facs()
        uv = si.T/uvsize
        ib = self.sky.in_solarbounds(uv + .5/uvsize,
                                     size=1/uvsize)
        ib = (skyb*ib > self.srct)
        border = 2*uvsize/180
        j = np.random.default_rng().uniform(border, 1 - border, si.T[ib].shape)
        # j = .5
        uv = (si.T[ib] + j)/uvsize
        xyz = translate.uv2xyz(uv, xsign=1)
        return xyz
