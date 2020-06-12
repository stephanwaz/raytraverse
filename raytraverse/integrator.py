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

    def __init__(self, scene, levels, rebuild=False):
        self.scene = scene
        #: np.array: sampling scheme from Sampler
        self.levels = levels
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
        self.pt_kd = scene.outdir

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
        """luminance arrays grouped by point

        :getter: Returns luminance array
        :type: list of np.array
        """
        return self._lum

    @property
    def omega(self):
        """solid angles grouped by point

        :getter: Returns solid angles
        :type: list of np.array
        """
        return self._omega

    @property
    def vec(self):
        """direction vectors grouped by point

        :getter: Returns vector array
        :type: list of np.array
        """
        return self._vec

    @property
    def d_kd(self):
        """list of direction kdtrees

        :getter: Returns kd tree structure
        :type: list of scipy.spatial.cKDTree
        """
        return self._d_kd

    @property
    def pt_kd(self):
        """point kdtree

        :getter: Returns kd tree structure
        :setter: Set this integrator's kd tree and scene data
        :type: scipy.spatial.cKDTree
        """
        return self._pt_kd

    @pt_kd.setter
    def pt_kd(self, outdir):
        """Set this integrator's kd tree and scene data"""
        kdfile = f'{outdir}/kd_data.pickle'
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._pt_kd = pickle.load(f)
            self._d_kd = pickle.load(f)
            self._lum = pickle.load(f)
            self._vec = pickle.load(f)
            self._omega = pickle.load(f)
            f.close()
        else:
            dfiles = glob(f'{self.scene.outdir}/*.npy')
            self._pt_kd, self._d_kd, self._lum, self._vec, self._omega = self.mk_tree(dfiles)
            f = open(kdfile, 'wb')
            pickle.dump(self.pt_kd, f, protocol=4)
            pickle.dump(self.d_kd, f, protocol=4)
            pickle.dump(self.lum, f, protocol=4)
            pickle.dump(self.vec, f, protocol=4)
            pickle.dump(self.omega, f, protocol=4)
            f.close()

    def mk_tree(self, datafiles):
        first = True
        for lf in datafiles:
            lev = np.load(lf)
            if not first:
                samps = np.vstack((samps, lev))
            else:
                first = False
                samps = lev
        samps = samps[samps[:,0].argsort()]
        pidx = samps[:, 0]
        pts = self.pts()
        pt_div = np.searchsorted(pidx, np.arange(len(pts)), side='right')
        pt_kd = cKDTree(pts)
        d_kd = [None]*pts.shape[0]
        vecs = [None]*pts.shape[0]
        lums = [None]*pts.shape[0]
        omegas = [None]*pts.shape[0]
        skside = int(np.sqrt(samps.shape[1] - 4))
        pt0 = 0
        for i, pt in enumerate(pt_div):
            d_kd[i] = cKDTree(samps[pt0:pt, 1:4])
            vecs[i] = samps[pt0:pt, 1:4]
            omegas[i] = SphericalVoronoi(samps[pt0:pt, 1:4]).calculate_areas()
            lums[i] = samps[pt0:pt, 4:].reshape(-1, skside, skside)
            pt0 = pt
        return pt_kd, d_kd, lums, vecs, omegas

    def idx2pt(self, idx):
        shape = self.levels[-1, 0:2]
        si = np.stack(np.unravel_index(idx, shape)).T
        return self.scene.area.uv2pt((si + .5)/shape)

    def pts(self):
        shape = self.levels[-1, 0:2]
        return self.idx2pt(np.arange(np.product(shape)))

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
        perrs: list or np.array
            for each of vpts, the distance to the returned values
        pis: list np.array
            for each of vpts, the index of the returned point
        idxs: list of list of np.array
            for each of vpts, for each vdir, an array of all idxs in
            self.kd.data matching the query

        """
        vdirs = translate.norm(vdirs)
        vs = translate.theta2chord(viewangle/360*np.pi)
        treedir = vdirs.shape[0] > treecnt
        if treedir:
            dtree = cKDTree(vdirs)
        with ProcessPoolExecutor() as exc:
            perrs, pis = zip(*exc.map(self.pt_kd.query, vpts))
            # print(pidxs, perr, self.idx2pt(pidxs))
            futures = []
            for perr, pi in zip(perrs, pis):
                if treedir:
                    futures.append(exc.submit(dtree.query_ball_tree,
                                              self.d_kd[pi], vs))
                else:
                    futures.append(exc.submit(self.d_kd[pi].query_ball_point,
                                              vdirs, vs))
        idxs = [future.result() for future in futures]
        return perrs, pis, idxs

    def apply_skyvecs(self, pis, skyvecs):
        if skyvecs is None:
            return [np.sum(self.lum[pi], (1, 2)).reshape(1, -1) for pi in pis]
        skyvecs = skyvecs.reshape(-1, self.lum[pis[0]].shape[1]**2).T
        return [(self.lum[pi].reshape(-1, self.lum[pi].shape[1]**2)@skyvecs).T
                for pi in pis]

    def view(self, vpts, vdirs, decades=4, maxl=0.0, skyvecs=None, treecnt=30,
             ring=150, viewangle=180.0):
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
        viewangle: float, optional
            degree opening of view cone

        Returns
        -------

        """
        perrs, pis, idxs = self.query(vpts, vdirs, treecnt=treecnt,
                                      viewangle=viewangle)
        lum = [np.log10(l) for l in self.apply_skyvecs(pis, skyvecs)]
        vm = ViewMapper(viewangle=180)
        ext = translate.xyz2xy(translate.tp2xyz(np.array([[viewangle*np.pi/360,
                                                           np.pi]])))[0, 0]
        tp = np.stack((np.linspace(0, np.pi/2, 11), np.full(11, np.pi))).T
        rvecs = translate.tp2xyz(tp)
        rxy = translate.xyz2xy(rvecs)
        if ring > 0:
            tp = np.stack((np.full(ring, viewangle*np.pi/360),
                           np.arange(0,2*np.pi, 2*np.pi/ring))).T
            rvecs = translate.tp2xyz(tp)
            rxy = translate.xyz2xy(rvecs)
        for k, v in enumerate(vdirs):
            vm.dxyz = v
            if ring > 0:
                trvecs = vm.view2world(rvecs)
            for li, (perr, pi, idx) in enumerate(zip(perrs, pis, idxs)):
                vec = vm.xyz2xy(self.vec[pi][idx[k]])
                if ring > 0:
                    d, tri = self.d_kd[pi].query(trvecs)
                    vec = np.vstack((vec, rxy))
                    vi = np.concatenate((idx[k], tri))
                else:
                    vi = idx[k]
                for j in range(lum[li].shape[0]):
                    try:
                        fig, ax = io.mk_img(lum[li][j, vi], vec,
                                            decades=decades, maxl=maxl,
                                            mark=True, ext=ext, inclmarks=len(idx[k]))
                        ax.set_title(f"{self.idx2pt([pi])} {v}")
                        plt.tight_layout()
                        plt.savefig(f'{pi}_{k}_{j}.png')
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
        perrs, pis, idxs = self.query(vpts, vdirs, treecnt=treecnt)
        vdirs = translate.norm(vdirs)
        lum = self.apply_skyvecs(pis, skyvecs)
        futures = []
        with ProcessPoolExecutor() as executor:
            for j in range(lum[0].shape[0]):
                for li, (perr, pi, idx) in enumerate(zip(perrs, pis, idxs)):
                    for k, v in enumerate(vdirs):
                        vec = self.vec[pi][idx[k]]
                        omega = self.omega[pi][idx[k]]
                        futures.append(executor.submit(optic.calc_illum, v,
                                                       vec.T, omega,
                                                       lum[li][j, idx[k]]))
        return np.array([fut.result()
                         for fut in futures]).reshape((skyvecs.shape[1],
                                                       len(pis), len(vdirs)))
