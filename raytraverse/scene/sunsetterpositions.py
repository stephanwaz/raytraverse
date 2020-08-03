# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse import translate, skycalc
from raytraverse.scene.skyinfo import SkyInfo
from raytraverse.scene.sunsetter import SunSetter


class SunSetterPositions(SunSetter):
    """select suns to sample based on sky pdf and scene.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    wea: str, np.array, optional
        path to sun position file or wea file, or array of sun positions
    srct: float, optional
        threshold of sky contribution for determining appropriate srcn
    skyro: float, optional
        sky rotation (in degrees, ccw)
    reload: bool
        if True reloads existing sun positions, else always generates new
    """

    def __init__(self, scene, wea, skyro=0.0, **kwargs):
        #: raytraverse.scene.Scene
        self.scene = scene
        #: float: ccw rotation (in degrees) for sky
        self.skyro = skyro
        #: raytraverse.scene.SkyInfo
        self.candidates = wea
        super().__init__(scene, skyro=skyro, **kwargs)

    @property
    def candidates(self):
        """candidate sun positions

        :getter: Returns the sun source array
        :setter: Set the sun source array and write to files
        :type: np.array
        """
        return self._candidates

    @candidates.setter
    def candidates(self, wea):
        """load candidate sun positions"""
        try:
            xyz = np.loadtxt(wea)
            if xyz.shape[1] != 3:
                raise ValueError('sunpos should have 3 col: dx, dy, dz')
            self._candidates = translate.norm(xyz)
        except ValueError:
            loc = skycalc.get_loc_epw(wea)
            wdat = skycalc.read_epw(wea)
            times = skycalc.row_2_datetime64(wdat[:, 0:3])
            self._candidates = skycalc.sunpos_xyz(times, *loc, ro=self.skyro)

    def choose_suns(self, uvsize):
        cbins = translate.uv2bin(translate.xyz2uv(self.candidates,
                                                  normalize=True, flipu=False),
                                 self.scene.skyres)
        cidxs = np.arange(cbins.size)
        sbins = np.arange(uvsize**2)
        skyb = self.load_sky_facs()
        if skyb is 1:
            skyb = np.full(sbins.shape[0], 1)
        idxs = []
        for b in sbins:
            if skyb[b] > self.srct:
                try:
                    a = np.random.choice(cidxs[cbins == b])
                except ValueError:
                    pass
                else:
                    idxs.append(a)
        return self.candidates[idxs]

