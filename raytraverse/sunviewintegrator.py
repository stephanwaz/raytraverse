# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi
import clasp.script_tools as cst
from raytraverse import translate, io, optic, ViewMapper, Integrator


class SunViewIntegrator(Integrator):
    """loads scene and sampling data for processing

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        prefix of data files to integrate
    rebuild: bool, optional
        build kd-tree even if one exists
    """

    def __init__(self, scene, suns, rebuild=False):
        self.suns = suns
        super().__init__(scene, rebuild, 'sunview')

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
        kdfile = f'{outdir}/{self.prefix}_kd_data.pickle'
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._pt_kd = pickle.load(f)
            self._d_kd = pickle.load(f)
            self._lum = pickle.load(f)
            self._vec = pickle.load(f)
            self._omega = pickle.load(f)
            self._svs = pickle.load(f)
            self._pidx = pickle.load(f)
            self._isort = pickle.load(f)
            f.close()
        else:
            dfile = f'{self.scene.outdir}/{self.prefix}_vals.out'
            vfile = f'{self.scene.outdir}/{self.prefix}_vecs.out'
            if not (os.path.isfile(dfile) and os.path.isfile(vfile)):
                raise FileNotFoundError("No results files found, have you run"
                                        f" a Sampler of type {self.prefix} for"
                                        f" scene {self.scene.outdir}?")
            (self._pt_kd, self._d_kd, self._lum,
             self._vec, self._omega, self._svs,
             self._pidx, self._isort) = self.mk_tree(vfile, dfile)
            f = open(kdfile, 'wb')
            pickle.dump(self.pt_kd, f, protocol=4)
            pickle.dump(self.d_kd, f, protocol=4)
            pickle.dump(self.lum, f, protocol=4)
            pickle.dump(self.vec, f, protocol=4)
            pickle.dump(self.omega, f, protocol=4)
            pickle.dump(self.svs, f, protocol=4)
            pickle.dump(self.pidx, f, protocol=4)
            pickle.dump(self.isort, f, protocol=4)
            f.close()

    def mk_tree(self, vfile, dfile):
        fvecs = io.bytefile2np(open(vfile, 'rb'), (-1, 4))
        sorting = fvecs[:, 0].argsort()
        fvals = optic.rgb2rad(io.bytefile2np(open(dfile, 'rb'), (-1, 3)))
        fvals = fvals.reshape(fvecs.shape[0], -1)[sorting]
        fvecs = fvecs[sorting]
        pidx = fvecs[:, 0]
        pts = self.scene.pts()
        pt_div = np.searchsorted(pidx, np.arange(len(pts)), side='right')
        pt_kd = cKDTree(pts)
        sun_kd = cKDTree(self.suns)
        d_kd = [None]*pts.shape[0]
        vecs = [None]*pts.shape[0]
        lums = [None]*pts.shape[0]
        svs = [None]*pts.shape[0]
        omegas = [None]*pts.shape[0]
        pt0 = 0
        for i, pt in enumerate(pt_div):
            d_kd[i] = cKDTree(fvecs[pt0:pt, 1:4])
            vecs[i] = fvecs[pt0:pt, 1:4]
            try:
                svs[i] = SphericalVoronoi(fvecs[pt0:pt, 1:4])
                omegas[i] = svs[i].calculate_areas()
            except ValueError as e:
                print(f'Warning, SphericalVoronoi not set at point {i}:')
                print(e)
                print(f'Source Solid angle calculation failed')
            lums[i] = fvals[pt0:pt]
            pt0 = pt
        return pt_kd, d_kd, lums, vecs, omegas, svs, pidx, sorting

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
