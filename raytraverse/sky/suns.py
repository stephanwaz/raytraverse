# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os

import numpy as np
from scipy.spatial import cKDTree

from raytraverse import translate, plot


class Suns(object):
    """select suns to sample based on sky pdf and scene.

    Parameters
    ----------
    scene: str,
        path of scene
    skyro: float, optional
        sky rotation (in degrees, ccw)
    reload: bool
        if True reloads existing sun positions, else always generates new
    sunres: float
    prefix: str
    suns: np.array
        shape (N, 3) to directly set suns.
    """

    def __init__(self, scene, skyro=0.0, reload=True, sunres=10.0,
                 prefix="suns", suns=None, **kwargs):
        #: float: ccw rotation (in degrees) for sky
        self.skyro = skyro
        self.sunres = sunres
        self.srcsize = 0.533
        self.scene = scene
        self.sunfile = f"{scene}/{prefix}.dat"
        if not reload:
            try:
                os.remove(self.sunfile)
            except FileNotFoundError:
                pass
        self._sun_kd = None
        self.suns = suns

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
    def suns(self, s):
        """set the skydetail array and determine sample count and spacing"""
        if s is not None:
            self._suns = s
        elif os.path.isfile(self.sunfile):
            xyz = np.loadtxt(self.sunfile)
            self._suns = xyz
        else:
            self._suns = self.choose_suns()
            try:
                np.savetxt(self.sunfile, self._suns)
            except FileNotFoundError:
                os.mkdir(self.scene)
                np.savetxt(self.sunfile, self._suns)
        self._sbins = translate.uv2bin(translate.xyz2uv(self.suns, flipu=False),
                                       self.sunres)
        self.sun_kd = None

    @property
    def sbins(self):
        """holds sun bin numbers
        """
        return self._sbins

    def _jitter_suns(self, si):
        border = 2*self.sunres/180
        j = np.random.default_rng().uniform(border, 1 - border, si.shape)
        uv = (si + j)/self.sunres
        return translate.uv2xyz(uv, xsign=1)

    def choose_suns(self):
        si = np.stack(np.unravel_index(np.arange(self.sunres**2),
                                       (self.sunres, self.sunres)))
        xyz = self._jitter_suns(si.T)
        return xyz

    def direct_view(self):
        sxy = translate.xyz2xy(self.suns, flip=False)
        lums, fig, ax, norm, lev = plot.mk_img_setup([0, 1], ext=1)
        ax.plot(sxy[:, 0], sxy[:, 1], lw=.5, ms=0, color='grey')
        ax.scatter(sxy[:, 0], sxy[:, 1], s=30, marker='o', color=(1, .5, 0))
        ax.set_facecolor((.2, .3, .5))
        outf = f"{self.scene}_suns.png"
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
        suns = translate.norm(tsuns)
        serrs, sis = self.sun_kd.query(suns)
        serrs = translate.chord2theta(serrs) * 180 / np.pi
        sis = np.where(serrs < tol, self.sbins[sis], self.sunres**2)
        return sis, serrs
