# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from raytraverse import translate, plot
from raytraverse.scene.sunsetterbase import SunSetterBase


class SunSetter(SunSetterBase):
    """select suns to sample based on sky pdf and scene.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    srct: float, optional
        threshold of sky contribution for determining appropriate srcn
    skyro: float, optional
        sky rotation (in degrees, ccw)
    reload: bool
        if True reloads existing sun positions, else always generates new
    """

    def __init__(self, scene, srct=.01, skyro=0.0, reload=True, sunres=10.0,
                 **kwargs):
        #: float: threshold of sky contribution for determining appropriate srcn
        self.srct = srct
        #: float: ccw rotation (in degrees) for sky
        self.skyro = skyro
        self.sunres = sunres
        self.srcsize = 0.533
        #: bool: reuse existing sun positions (if found)
        if not reload:
            try:
                os.remove(f'{scene.outdir}/sky_skydetail.npy')
            except FileNotFoundError:
                pass
        super().__init__(scene, reload=reload)
        self.sun_kd = None

    @property
    def sunres(self):
        return self._sunres

    @sunres.setter
    def sunres(self, s):
        self._sunres = int(np.floor(90/s)*2)

    @property
    def sun_kd(self):
        """sun kdtree for directional queries"""
        if self._sun_kd is None:
            self._sun_kd = cKDTree(self.suns)
        return self._sun_kd

    @sun_kd.setter
    def sun_kd(self, skd):
        self._sun_kd = skd

    @property
    def suns(self):
        """holds sun positions

        :getter: Returns the sun source array
        :setter: Set the sun source array and write to files
        :type: np.array
        """
        return self._suns

    @suns.setter
    def suns(self, sunfile):
        """set the skydetail array and determine sample count and spacing"""
        if os.path.isfile(sunfile):
            f = open(sunfile, 'r')
            sund = f.read().split('source')[1:]
            xyz = np.array([s.split()[4:7] for s in sund]).astype(float)
            self._suns = xyz
        else:
            self._suns = self.choose_suns()
            self._write_suns(sunfile)

    def choose_suns(self):
        uvsize = self.sunres
        si = np.stack(np.unravel_index(np.arange(uvsize**2),
                                       (uvsize, uvsize)))
        skyb = self.load_sky_facs()
        print(uvsize)
        uv = si.T/uvsize
        ib = np.full(uv.shape[0], True)
        ib = (skyb*ib > self.srct)
        border = 2*uvsize/180
        j = np.random.default_rng().uniform(border, 1 - border, si.shape)
        # j = .5
        uv = (si + j).T[ib]/uvsize
        xyz = translate.uv2xyz(uv, xsign=1)
        return xyz

    def load_sky_facs(self):
        outf = f'{self.scene.outdir}/sky_skydetail.npy'
        if os.path.isfile(outf):
            sd = np.load(outf)
        else:
            dfile = f'{self.scene.outdir}/sky_kd_lum_map.pickle'
            try:
                f = open(dfile, 'rb')
            except FileNotFoundError:
                print('Warning! suns initialized without sky detail, first'
                      ' create a SCBinField', file=sys.stderr)
                return np.ones(self.sunres * self.sunres)
            else:
                skylums = pickle.load(f)
                f.close()
                zeros = np.zeros(len(skylums), dtype=int)
                with ThreadPoolExecutor() as exc:
                    mxs = list(exc.map(np.max, skylums.values(), zeros))
                sd = np.max(np.stack(mxs), 0)[:-1].reshape(self.scene.skyres,
                                                           self.scene.skyres)
                np.save(outf, sd)
        sd = translate.interpolate2d(sd, (self.sunres, self.sunres))
        return sd.ravel()

    def direct_view(self):
        sxy = translate.xyz2xy(self.suns, flip=False)
        sbins = translate.uv2bin(translate.xyz2uv(self.suns, flipu=False),
                                 self.sunres)
        lums, fig, ax, norm, lev = plot.mk_img_setup([0, 1], ext=1)
        outf = f'{self.scene.outdir}/sky_skydetail.npy'
        if os.path.isfile(outf):
            sf = translate.interpolate2d(np.load(outf),
                                         (self.sunres, self.sunres)).ravel()
            borders = translate.bin_borders(np.arange(sf.size),
                                            self.sunres)
            sfxyz = translate.uv2xyz(borders.reshape(-1, 2), xsign=1)
            sfxy = translate.xyz2xy(sfxyz, flip=False).reshape(-1, 4, 2)
            cmap = plot.colormap('gray',
                                 plot.Normalize(vmin=np.log10(self.srct) - 1,
                                                vmax=0))
            colors = cmap.to_rgba(np.log10(sf))
            patcha = [(c, p) for c, p in zip(colors, sfxy)]
            patchb = [(c, p) for c, p in zip(colors[sbins], sfxy[sbins])]
            plot.plot_patches(ax, patcha, {'zorder': -2, 'lw': .25,
                                           'edgecolor': (.2, .2, .2)})
            plot.plot_patches(ax, patchb, {'zorder': -1, 'lw': 3,
                                           'edgecolor': (1, .5, 0)})
        ax.plot(sxy[:, 0], sxy[:, 1], lw=.5, ms=0, color='grey')
        ax.scatter(sxy[:, 0], sxy[:, 1], s=30, marker='o', color=(1, .5, 0))
        ax.set_facecolor((.2, .3, .5))
        outf = f"{self.scene.outdir}_suns.png"
        plot.save_img(fig, ax, outf)

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
            (N,) index to proxy src
        list
            (N,) error in degrees to proxy sun
        """
        stol = translate.theta2chord(tol*np.pi/180)
        suns = translate.norm(tsuns)
        serrs, sis = self.sun_kd.query(suns)
        serrs = translate.chord2theta(serrs) * 180 / np.pi
        sis = np.where(serrs < stol, sis, self.suns.shape[0])
        return sis, serrs
