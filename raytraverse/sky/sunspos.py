# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse import translate
from raytraverse.sky import skycalc
from raytraverse.sky.suns import Suns


class SunsPos(Suns):
    """select suns to sample based on sun positions.
    the wea argument provides a list of sun positions to draw from rather than
    randomly generating the sun position like Suns and SunsLoc.

    Parameters
    ----------
    sscene: str,
        path of scene
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
        #: float: ccw rotation (in degrees) for sky
        self.skyro = skyro
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
            dat = np.atleast_2d(np.loadtxt(wea))
            if dat.shape[1] == 3:
                dummydir = np.ones((dat.shape[0], 1))
                self._candidates = np.hstack((translate.norm(dat), dummydir))
            elif dat.shape[1] == 4:
                xyz = translate.aa2xyz(dat[:, 0:2])
                self._candidates = np.hstack((xyz, dat[:, 3:4]))
            elif dat.shape[1] == 5 and np.max(dat[:, 0:3] <= 1):
                xyz = translate.norm(dat[:, 0:3])
                self._candidates = np.hstack((xyz, dat[:, 3:4]))
            else:
                raise ValueError
        except ValueError:
            loc = skycalc.get_loc_epw(wea)
            wdat = skycalc.read_epw(wea)
            times = skycalc.row_2_datetime64(wdat[:, 0:3])
            xyz = skycalc.sunpos_xyz(times, *loc, ro=self.skyro)
            self._candidates = np.hstack((xyz, wdat[:, 3:4]))

    def choose_suns(self):
        uvsize = self.sunres
        cbins = translate.uv2bin(translate.xyz2uv(self.candidates[:, 0:3],
                                                  normalize=True, flipu=False),
                                 uvsize)
        cidxs = np.arange(cbins.size)
        sbins = np.arange(uvsize**2)
        idxs = []
        for b in sbins:
            # choose sun position, priortizing values closer to center of bin
            p = self.candidates[cbins == b, 3]
            # only if there is a sun with a dirnorm > 0
            if np.sum(p) != 0:
                binc = translate.uv2xyz(translate.bin2uv([b], uvsize),
                                        xsign=1)
                bp = np.linalg.norm(self.candidates[cbins == b, 0:3] - binc,
                                    axis=1)
                a = np.random.choice(cidxs[cbins == b], p=bp/np.sum(bp))
                idxs.append(a)
        return self.candidates[idxs, 0:3]

