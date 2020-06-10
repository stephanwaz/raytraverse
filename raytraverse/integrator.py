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
from scipy.spatial import cKDTree, SphericalVoronoi

from raytraverse import translate, io, optic, ViewMapper


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
            vecn = lev[:, 3:6]
            lum2 = lev[:, 6:]
            if not first:
                lum = np.vstack((lum, lum2))
                vec = np.vstack((vec, vecn))
            else:
                first = False
                lum = lum2
                vec = vecn
        kd = cKDTree(vec)
        voronoi = SphericalVoronoi(vec)
        omegas = voronoi.calculate_areas()
        skside = int(np.sqrt(lum.shape[-1]))
        lum = lum.reshape(-1, skside, skside)
        return kd, lum, vec, omegas

    def __init__(self, scene, levels, rebuild=False):
        self.scene = scene
        #: np.array: sampling scheme from Sampler
        self.levels = levels
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
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
        """luminance array

        :getter: Returns luminance array
        :type: np.array
        """
        return self._lum

    @property
    def omega(self):
        """solid angles

        :getter: Returns solid angles
        :type: np.array
        """
        return self._omega

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
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._kd = pickle.load(f)
            self._lum = pickle.load(f)
            self._vec = pickle.load(f)
            self._omega = pickle.load(f)
            f.close()
        else:
            dfiles = glob(f'{self.scene.outdir}/*.npy')
            self._kd, self._lum, self._vec, self._omega = self.mk_tree(dfiles)
            f = open(kdfile, 'wb')
            pickle.dump(self._kd, f, protocol=4)
            pickle.dump(self._lum, f, protocol=4)
            pickle.dump(self._vec, f, protocol=4)
            pickle.dump(self._omega, f, protocol=4)
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
        if vdirs.shape[0] > treecnt:
            ptree = cKDTree(vdirs)
            idxs = ptree.query_ball_tree(self.kd, vs)
        else:
            idxs = self.kd.query_ball_point(vdirs, vs)
        return vdirs, idxs

    def view(self, vpts, vdirs, decades=4, maxl=0.0, skyvecs=None, treecnt=30,
             ring=150):
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
        ring: int, optional
            add points around perimeter to clean up edge, set to 0 to disable

        Returns
        -------

        """
        vdirs, idxs = self.query(vpts, vdirs, treecnt=treecnt)
        if skyvecs is None:
            lum = np.sum(self.lum, (1, 2)).reshape(1, -1)
        else:
            skyvecs = skyvecs.reshape(-1, self.lum.shape[1]**2).T
            lum = (self.lum.reshape(-1, self.lum.shape[1]**2)@skyvecs).T
        lum = np.log10(lum)
        vm = ViewMapper(viewangle=180)
        if ring > 0:
            tp = np.stack((np.full(ring, np.pi/2),
                           np.arange(0,2*np.pi, 2*np.pi/ring))).T
            rvecs = translate.tp2xyz(tp)
            rxy = translate.xyz2xy(rvecs)
        for k, (i, v) in enumerate(zip(idxs, vdirs)):
            vm.dxyz = v
            vec = vm.xyz2xy(self.vec[i])
            if ring > 0:
                trvecs = vm.view2world(rvecs)
                d, tri = self.kd.query(trvecs)
                vec = np.vstack((vec, rxy))
                vi = np.concatenate((i, tri))
            else:
                vi = i
            for j in range(lum.shape[0]):
                try:
                    fig, ax = io.mk_img(lum[j, vi], vec, inclmarks=len(i),
                                        decades=decades, maxl=maxl, mark=True)
                    fig.suptitle(f"{v}")
                    plt.savefig(f"{k}_{j}.png")
                    plt.close(fig)
                except ValueError:
                    pass

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
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(optic.calc_illum, v, self.vec[i].T,
                                       self.omega[i], lum[:, i])
                       for i, v in zip(idxs, vdirs)]
        return np.stack([future.result() for future in futures])
