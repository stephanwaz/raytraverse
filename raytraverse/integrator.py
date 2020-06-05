# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from raytraverse import translate, io, optic


class Integrator(object):
    """loads scene and sampling data for processing

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    """
    @staticmethod
    def mk_tree(datafiles):
        first = True
        for lf in datafiles:
            lev = np.load(lf)
            lev = lev[1:]
            vecn = lev[:, 0:6]
            lum2 = lev[:, 6:]
            if not first:
                lum = np.vstack((lum, lum2))
                vec = np.vstack((vec, vecn))
            else:
                first = False
                lum = lum2
                vec = vecn
        kd = cKDTree(vec[:, (0, 1, 3, 4, 5)])
        skside = int(np.sqrt(lum.shape[-1]))
        lum = lum.reshape(-1, skside, skside)
        return kd, lum, vec

    def __init__(self, scene, levels):
        self.scene = scene
        #: np.array: sampling scheme from Sampler
        self.levels = levels
        self.kd = scene.outdir

    @property
    def scene(self):
        """scene information

        :getter: Returns this integrator's scene
        :setter: Set this integrator's scene
        :type: raytraverse.scene.Scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        """Set this integrator's scene and load samples"""
        self._scene = scene

    @property
    def lum(self):
        """scene information

        :getter: Returns luminance array
        :type: np.array
        """
        return self._lum

    @property
    def vec(self):
        """scene information

        :getter: Returns vector array
        :type: np.array
        """
        return self._vec

    @property
    def kd(self):
        """scene information

        :getter: Returns kd tree structure
        :setter: Set this integrator's kd tree and scene data
        :type: scipy.spatial.cKDTree
        """
        return self._kd

    @kd.setter
    def kd(self, outdir):
        """Set this integrator's kd tree and scene data"""
        kdfile = f'{outdir}/kd_data.pickle'
        if os.path.isfile(kdfile):
            f = open(kdfile, 'rb')
            self._kd = pickle.load(f)
            self._lum = pickle.load(f)
            self._vec = pickle.load(f)
            f.close()
        else:
            dfiles = glob(f'{self.scene.outdir}/*.npy')
            self._kd, self._lum, self._vec = self.mk_tree(dfiles)
            f = open(kdfile, 'wb')
            pickle.dump(self._kd, f, protocol=4)
            pickle.dump(self._lum, f, protocol=4)
            pickle.dump(self._vec, f, protocol=4)
            f.close()

    def query(self, vpts, vdirs, viewangle=180.0, treecnt=30):
        """gather all rays from a point within a view cone

        Parameters
        ----------
        vpts: np.array
            points to search for (broadcastable to directions)
        vdirs: np.array
            directions to search for
        viewangle: float, optional
            degree opening of view cone
        treecnt: int, optional
            number of queries at which a scipy.cKDtree.query_ball_tree is
            used instead of scipy.cKDtree.query_ball_point

        Returns
        -------
        vdirs: np.array
            2d array of normalized view directions
        idxs: list of np.array
            for each vdir, list contains an array of all idxs in self.kd.data
            matching the query

        """
        vdirs = translate.norm(vdirs)
        vs = translate.theta2chord(viewangle/360 * np.pi)
        try:
            vpn = np.hstack((vpts[:, 0:2], vdirs))
        except (ValueError, IndexError):
            vpts = np.broadcast_to(vpts[0:2], (vdirs.shape[0], 2))
            vpn = np.hstack((vpts, vdirs))
        if vdirs.shape[0] > treecnt:
            ptree = cKDTree(vpn)
            idxs = ptree.query_ball_tree(self.kd, vs)
        else:
            idxs = self.kd.query_ball_point(vpn, vs)
        return vdirs, idxs

    def view(self, vpts, vdirs, decades=4, maxl=0.0, skyvecs=None, treecnt=30):
        """generate angular fisheye falsecolor luminance views

        Parameters
        ----------
        vpts: np.array
            points to search for (broadcastable to directions)
        vdirs: np.array
            directions to search for
        decades: real, optional
            number of log decades below max for minimum of color scale
        maxl: real, optional
            maximum log10(lum/179) for color scale
        skyvecs: np.array
            array of (N,) + lum.shape[1:] (sky vector)
        treecnt: int, optional
            number of queries at which a scipy.cKDtree.query_ball_tree is
            used instead of scipy.cKDtree.query_ball_point

        Returns
        -------

        """
        vdirs, idxs = self.query(vpts, vdirs, treecnt=treecnt)
        if skyvecs is None:
            lum = np.sum(self.lum, (1, 2)).reshape(1, -1)
        else:
            skyvecs = skyvecs.reshape(-1, self.lum.shape[1]**2).T
            lum = (self.lum.reshape(-1, self.lum.shape[1]**2)@skyvecs).T
        lum = np.log10(lum/179)
        for i, v in zip(idxs, vdirs):
            xyz = translate.rotate(self.vec[i, 3:], v, (0, 1, 0))
            vec = translate.xyz2xy(xyz, axes=(0, 2, 1))
            for j in range(lum.shape[0]):
                io.mk_img(lum[j, i], vec, decades=decades, maxl=maxl, mark=False)
        plt.show()

    def calc_omega(self):
        """calculate solid angle of each ray in self.vec

        Returns
        -------
        omegas: np.array
            estimated solid angles normalized to sphere (should be renormalized
            when subsampling, ie illuminance hemisphere)
        """
        d, i = self.kd.query(self.kd.data, 2)
        thetas = translate.chord2theta(d[:,1])
        omegas = 2*np.pi * (1 - np.cos(thetas))
        tom = np.sum(omegas)
        if np.abs(np.pi*4 - tom) > np.pi*.2:
            print("Warning, solid angle estimate off by > 5%! "
                  f"{tom} =/= {np.pi*2}")
        return omegas * 4*np.pi / tom

    def illum(self, vpts, vdirs, skyvecs=None, treecnt=30):
        """calculate illuminance for given sensor locations and skyvecs

        Parameters
        ----------
        vpts: np.array
            points to search for (broadcastable to directions)
        vdirs: np.array
            directions to search for
        skyvecs: np.array
            array of (N,) + lum.shape[1:] (sky vector)
        treecnt: int, optional
            number of queries at which a scipy.cKDtree.query_ball_tree is
            used instead of scipy.cKDtree.query_ball_point

        Returns
        -------
        illum: np.array (vdirs.shape[0], skyvecs.shape[0])
            illuminance at each point/direction and sky weighting
        """
        vdirs, idxs = self.query(vpts, vdirs, treecnt=treecnt)
        if skyvecs is None:
            lum = np.sum(self.lum, (1, 2)).reshape(1, -1)
        else:
            skyvecs = skyvecs.reshape(-1, self.lum.shape[1]**2).T
            lum = (self.lum.reshape(-1, self.lum.shape[1]**2) @ skyvecs).T
        omegas = self.calc_omega()
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(optic.calc_illum, v, self.vec[i, 3:].T,
                                       omegas[i], lum[:, i])
                       for i, v in zip(idxs, vdirs)]
        return np.stack([future.result() for future in futures])
