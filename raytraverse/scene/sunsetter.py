# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from raytraverse import draw, translate, plot
from raytraverse.scene.skyinfo import SkyInfo
from raytraverse.mapper import SunMapper


class SunSetter(object):
    """select suns to sample based on sky pdf and scene.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    loc: tuple
        lat, lon, tz (in degrees, west is positive
    srct: float, optional
        threshold of sky contribution for determining appropriate srcn
    skyro: float, optional
        sky rotation (in degrees, ccw)
    reload: bool
        if True reloads existing sun positions, else always generates new
    """

    def __init__(self, scene, loc, srct=.01, skyro=0.0, reload=True, **kwargs):
        #: float: threshold of sky contribution for determining appropriate srcn
        self.srct = srct
        #: float: ccw rotation (in degrees) for sky
        self.skyro = skyro
        #: raytraverse.scene.SkyInfo
        self.sky = SkyInfo(loc, skyro)
        #: bool: reuse existing sun positions (if found)
        self.reload = reload
        #: raytraverse.scene.Scene
        self.scene = scene
        sunfile = f"{self.scene.outdir}/suns.rad"
        self.suns = sunfile
        self.map = SunMapper(self.suns)
        if not (os.path.isfile(sunfile) and self.reload):
            self._write_suns(sunfile)
        self.sun_kd = None

    @property
    def suns(self):
        """holds sun positions

        :getter: Returns the sun source array
        :setter: Set the sun source array and write to files
        :type: np.array
        """
        return self._suns

    @property
    def sun_kd(self):
        """sun kdtree for directional queries"""
        if self._sun_kd is None:
            self._sun_kd = cKDTree(self.suns)
        return self._sun_kd

    @sun_kd.setter
    def sun_kd(self, sun_kd):
        self._sun_kd = sun_kd

    @suns.setter
    def suns(self, sunfile):
        """set the skydetail array and determine sample count and spacing"""
        if os.path.isfile(sunfile) and self.reload:
            f = open(sunfile, 'r')
            sund = f.read().split('source')[1:]
            xyz = np.array([s.split()[4:7] for s in sund]).astype(float)
        else:
            uvsize = self.scene.skyres
            skyb = self.load_sky_facs()
            si = np.stack(np.unravel_index(np.arange(skyb.size), skyb.shape))
            uv = si.T/uvsize
            ib = self.sky.in_solarbounds(uv + .5/uvsize,
                                         size=1/uvsize).reshape(skyb.shape)
            suncount = np.sum(skyb*ib > self.srct)
            skyd = draw.get_detail(skyb, (0, 1)).reshape(skyb.shape)
            sb = (skyb + skyd)/np.min(skyb + skyd)
            sd = (sb * ib).ravel()
            sdraws = np.random.default_rng().choice(skyb.size, suncount,
                                                    replace=False,
                                                    p=sd/np.sum(sd))
            si = np.stack(np.unravel_index(sdraws, skyb.shape))
            # keep solar discs from overlapping
            border = uvsize/180
            j = np.random.default_rng().uniform(border, 1-border, si.T.shape)
            uv = (si.T + j)/uvsize
            xyz = translate.uv2xyz(uv, xsign=1)
        self._suns = xyz

    def load_sky_facs(self):
        outf = f'{self.scene.outdir}/sky_skydetail.npy'
        if os.path.isfile(outf) and self.reload:
            return np.load(outf)
        dfile = f'{self.scene.outdir}/sky_kd_lum_map.pickle'
        f = open(dfile, 'rb')
        skylums = pickle.load(f)
        f.close()
        zeros = np.zeros(len(skylums), dtype=int)
        with ThreadPoolExecutor() as exc:
            mxs = list(exc.map(np.max, skylums, zeros))
        sd = np.max(np.stack(mxs), 0).reshape(self.scene.skyres,
                                              self.scene.skyres)
        np.save(outf, sd)
        return sd

    def direct_view(self):
        sxy = translate.xyz2xy(self.suns, flip=False)
        lums, fig, ax, norm, lev = plot.mk_img_setup([0, 1], ext=1)
        ax.scatter(sxy[:, 0], sxy[:, 1], s=15,
                   marker='o', facecolors='yellow')
        ax.set_facecolor((0, 0, 0))
        outf = f"{self.scene.outdir}_suns.png"
        plot.save_img(fig, ax, outf)

    def write_sun(self, i):
        s = self.suns[i]
        mod = f"solar{i:05d}"
        name = f"sun{i:05d}"
        d = f"{s[0]} {s[1]} {s[2]}"
        dec = f"void light {mod} 0 0 3 1 1 1\n"
        dec += f"{mod} source {name} 0 0 4 {d} 0.533\n"
        return dec, mod

    def proxy_src(self, tsuns, tol=10.0):
        """check if sun directions have matching source in SunSetter

        Parameters
        ----------
        tsuns: np.array
            (N, 3) array containing sun source vectors to check
        tol: float
            tolerance (in degrees)

        Returns
        -------
        np.array
            (N,) boolean array if sun has a match
        np.array
            (N,) index to proxy src
        """
        stol = translate.theta2chord(tol*np.pi/180)
        suns = translate.norm(tsuns)
        serrs, sis = self.sun_kd.query(suns)
        return serrs < stol, sis

    def _write_suns(self, sunfile):
        """write suns to file

        Parameters
        ----------
        sunfile
        """
        f = open(sunfile, 'w')
        g = open(f'{self.scene.outdir}/sun_modlist.txt', 'w')
        for i in range(self.suns.shape[0]):
            dec, mod = self.write_sun(i)
            print(dec, file=f)
            print(mod, file=g)
        f.close()
        g.close()
