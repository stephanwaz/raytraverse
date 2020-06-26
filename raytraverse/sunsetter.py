# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os

import numpy as np
from clasp import script_tools as cst

from raytraverse import wavelet, translate
from raytraverse.sunmapper import SunMapper


class SunSetter(object):
    """select suns to sample based on sky pdf and scene.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    sunres: float, optional
        minimum average seperation between sources (twice desired error)
    srct: float, optional
        threshold of sky contribution for determining appropriate srcn
    """

    def __init__(self, scene, sunres=5.0, srct=.01, reload=True, **kwargs):
        #: float: threshold of sky contribution for determining appropriate srcn
        self.srct = srct
        #: bool: reuse existing sun positions (if found)
        self.reload = reload
        #: raytraverse.scene.Scene
        self.scene = scene
        if sunres < .7:
            print('Warning! minimum sunres is .7 to avoid overlap and allow')
            print('for jittering position, sunres set to .7')
            sunres = .7
        self.suns = sunres
        self.map = SunMapper(self.suns)

    @property
    def suns(self):
        """holds sun positions

        :getter: Returns the sun source array
        :setter: Set the sun source array and write to files
        :type: np.array
        """
        return self._suns

    @suns.setter
    def suns(self, sunres):
        """set the skydetail array and determine sample count and spacing"""
        sunfile = f"{self.scene.outdir}/suns.rad"
        if os.path.isfile(sunfile) and self.reload:
            self.load_suns(sunfile)
        else:
            uvsize = int(np.floor(90/sunres)*2)
            skyb = self.load_sky_facs(uvsize)
            si = np.stack(np.unravel_index(np.arange(skyb.size), skyb.shape))
            uv = si.T/uvsize
            ib = self.scene.in_solarbounds(uv + .5/uvsize,
                                           size=1/uvsize).reshape(skyb.shape)
            suncount = np.sum(skyb*ib > self.srct)
            skyd = wavelet.get_detail(skyb, (0, 1)).reshape(skyb.shape)
            sb = (skyb + skyd)/np.min(skyb + skyd)
            sd = (sb * ib).ravel()
            sdraws = np.random.default_rng().choice(skyb.size, suncount,
                                                    replace=False,
                                                    p=sd/np.sum(sd))
            si = np.stack(np.unravel_index(sdraws, skyb.shape))
            # keep solar discs from overlapping
            border = .3*uvsize/180
            j = np.random.default_rng().uniform(border, 1-border, si.T.shape)
            uv = (si.T + j)/uvsize
            self._suns = translate.uv2xyz(uv, xsign=1)
            self.write_suns(sunfile)

    def load_suns(self, sunfile):
        """load suns from file

        void light solar00001 0 0 3 1 1 1
        solar00001 source sun00001 0 0 4 X Y Z 0.533
        .
        .
        .
        void light solarNNNNN 0 0 3 1 1 1
        solarNNNNN source sunNNNNN 0 0 4 X Y Z 0.533

        Parameters
        ----------
        sunfile
        """
        f = open(sunfile, 'r')
        sund = f.read().split('source')[1:]
        self._suns = np.array([s.split()[4:7] for s in sund]).astype(float)

    def write_suns(self, sunfile):
        """write suns to file and make sun octree

        Parameters
        ----------
        sunfile
        """
        sunoct = f'{self.scene.outdir}/sun.oct'
        f = open(sunfile, 'w')
        g = open(f'{self.scene.outdir}/sun_modlist.txt', 'w')
        for i, s in enumerate(self.suns):
            mod = f"solar{i:05d}"
            name = f"sun{i:05d}"
            d = f"{s[0]} {s[1]} {s[2]}"
            print(mod, file=g)
            print(f"void light {mod} 0 0 3 1 1 1", file=f)
            print(f"{mod} source {name} 0 0 4 {d} 0.533", file=f)
        f.close()
        g.close()
        f = open(sunoct, 'wb')
        cst.pipeline([f'oconv -f -i {self.scene.outdir}/scene.oct {sunfile}'],
                     outfile=f, close=True)

    def load_sky_facs(self, uvsize):
        try:
            skyb = np.load(f'{self.scene.outdir}/sky_skydetail.npy')
        except FileNotFoundError:
            print('Warning! sunsetter initialized without sky weights')
            skyb = np.ones((uvsize, uvsize))
        else:
            skyb = translate.interpolate2d(skyb, (uvsize, uvsize))
        return skyb
