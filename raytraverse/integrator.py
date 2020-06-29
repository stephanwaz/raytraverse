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
from raytraverse import translate, io, optic, ViewMapper


class Integrator(object):
    """loads scene and sampling data for processing

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    rebuild: bool, optional
        build kd-tree even if one exists
    prefix: str, optional
        prefix of data files to integrate
    """

    def __init__(self, scene, rebuild=False, prefix='sky'):
        self.scene = scene
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
        #: str: prefix of data files from sampler (stype)
        self.prefix = prefix
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
    def isort(self):
        """indexes used to sort from sampling to points

        :getter: Returns sort indices array
        :type: np.array
        """
        return self._isort

    @property
    def pidx(self):
        """point indices of samples

        :getter: Returns point indices array
        :type: np.array
        """
        return self._pidx

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

    @property
    def svs(self):
        """spherical voronoi at each point

        :getter: Return list of spherical voronoi
        :type: list of scipy.spatial.SphericalVoronoi
        """
        return self._svs

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
            except ValueError as e:
                print(f'Warning, SphericalVoronoi not set at point {i}:')
                print(e)
                print(f'Source Solid angle calculation failed')
            else:
                omegas[i] = svs[i].calculate_areas()
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

    def apply_coefs(self, pis, coefs=None):
        cnt = self.lum[pis[0]].shape[1]
        if coefs is None:
            lum = [np.sum(self.lum[pi], 1).reshape(1, -1) for pi in pis]
        elif coefs.shape[-1] == cnt:
            skyvecs = coefs.reshape(-1, cnt).T
            lum = [(self.lum[pi].reshape(-1, cnt)@skyvecs).T for pi in pis]
        else:
            coefs = coefs.reshape(-1, 2).T
            bins = coefs[0].astype(int)
            c = coefs[1]
            lum = [(self.lum[pi].reshape(-1, cnt)[:, bins]*c).T for pi in pis]
        return lum

    def view(self, vpts, vdirs, decades=4, maxl=0.0, coefs=None, treecnt=30,
             ring=150, viewangle=180.0, ringtol=15.0, scatter=False, **kwargs):
        """generate angular fisheye falsecolor luminance views
        additional kwargs passed to io.mk_img

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
        coefs: np.array
            array of (N,) + lum.shape[1:] (coefficient vector)
            or array of (N, 2) (index and coefficient)
        treecnt: int, optional
            number of queries at which a scipy.cKDtree.query_ball_tree is
            used instead of scipy.cKDtree.query_ball_point
        ring: int, optional
            add points around perimeter to clean up edge, set to 0 to disable
        viewangle: float, optional
            degree opening of view cone
        ringtol: float, optional
            tolerance (in degrees) for adding ring points

        Returns
        -------

        """
        popts = np.get_printoptions()
        np.set_printoptions(2)
        rt = translate.theta2chord(ringtol/180*np.pi)
        vm = ViewMapper(viewangle=viewangle)
        ext = translate.xyz2xy(translate.tp2xyz(np.array([[viewangle*np.pi/360,
                                                           np.pi]])))[0, 0]
        perrs, pis, idxs = self.query(vpts, vdirs, treecnt=treecnt,
                                      viewangle=viewangle)
        lum = self.apply_coefs(pis, coefs=coefs)
        lum = [np.log10(l) for l in lum]
        rvecs, rxy = translate.mkring(viewangle, ring)
        futures = []
        if scatter:
            func = io.mk_img_scatter
        else:
            func = io.mk_img
        with ProcessPoolExecutor() as exc:
            for k, v in enumerate(vdirs):
                vm.dxyz = v
                trvecs = vm.view2world(rvecs)
                for li, (perr, pi, idx) in enumerate(zip(perrs, pis, idxs)):
                    pt = self.scene.idx2pt([pi])[0]
                    if len(idx[k]) == 0:
                        print(f'Warning: No rays found at point: {pt} '
                              f'direction: {v}')
                    else:
                        vec = vm.xyz2xy(self.vec[pi][idx[k]])
                        d, tri = self.d_kd[pi].query(trvecs,
                                                     distance_upper_bound=rt)
                        vec = np.vstack((vec, rxy[d < ringtol]))
                        vi = np.concatenate((idx[k], tri[d < ringtol]))
                        for j in range(lum[li].shape[0]):
                            kw = dict(decades=decades, maxl=maxl,
                                      inclmarks=len(idx[k]), title=f"{pt} {v}",
                                      ext=ext, **kwargs)
                            futures.append(exc.submit(func, lum[li][j, vi],
                                                      vec, f'{self.prefix}_{pi}_{k}_{j}.png', **kw))
        np.set_printoptions(**popts)
        return [fut.result() for fut in futures]

    def voronoi(self, vpts, vdirs, decades=4, maxl=0.0, coefs=None, treecnt=30,
                viewangle=180.0, **kwargs):
        """generate angular fisheye falsecolor luminance views
        additional kwargs passed to io.mk_img

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
        coefs: np.array
            array of (N,) + lum.shape[1:] (coefficient vector)
            or array of (N, 2) (index and coefficient)
        treecnt: int, optional
            number of queries at which a scipy.cKDtree.query_ball_tree is
            used instead of scipy.cKDtree.query_ball_point
        viewangle: float, optional
            degree opening of view cone

        Returns
        -------

        """
        popts = np.get_printoptions()
        np.set_printoptions(2)
        vm = ViewMapper(viewangle=viewangle)
        ext = translate.xyz2xy(translate.tp2xyz(np.array([[viewangle*np.pi/360,
                                                           np.pi]])))[0, 0]
        perrs, pis, idxs = self.query(vpts, vdirs, treecnt=treecnt,
                                      viewangle=viewangle)
        lum = self.apply_coefs(pis, coefs=coefs)
        lum = [np.log10(l) for l in lum]
        futures = []
        with ProcessPoolExecutor() as exc:
            for k, v in enumerate(vdirs):
                vm.dxyz = v
                for li, (perr, pi, idx) in enumerate(zip(perrs, pis, idxs)):
                    pt = self.scene.idx2pt([pi])[0]
                    if len(idx[k]) == 0:
                        print(f'Warning: No rays found at point: {pt} '
                              f'direction: {v}')
                    else:
                        vi = idx[k]
                        vec = vm.xyz2xy(self.vec[pi][vi])
                        verts = vm.xyz2xy(self.svs[pi].vertices)
                        regions = self.svs[pi].regions
                        for j in range(lum[li].shape[0]):
                            kw = dict(decades=decades, maxl=maxl,
                                      title=f"{pt} {v}", ext=ext,
                                      outf=f'{self.prefix}_{pi}_{k}_{j}.png',
                                      **kwargs)
                            futures.append(
                                exc.submit(io.mk_img_voronoi, lum[li][j],
                                           vec, verts, regions, vi, **kw))
        np.set_printoptions(**popts)
        return [fut.result() for fut in futures]

    def illum(self, vpts, vdirs, coefs=None, treecnt=30):
        """calculate illuminance for given sensor locations and skyvecs

        Parameters
        ----------
        vpts: np.array
            points to search for (broadcastable to directions)
        vdirs: np.array
            directions to search for
        coefs: np.array
            array of (N,) + lum.shape[1:] (coefficient vector)
            or array of (N, 2) (index and coefficient)
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
        lum = self.apply_coefs(pis, coefs=coefs)
        futures = []
        with ProcessPoolExecutor() as exc:
            for j in range(lum[0].shape[0]):
                for li, (perr, pi, idx) in enumerate(zip(perrs, pis, idxs)):
                    for k, v in enumerate(vdirs):
                        vec = self.vec[pi][idx[k]]
                        omega = self.omega[pi][idx[k]]
                        futures.append(exc.submit(optic.calc_illum, v, vec.T,
                                                  omega, lum[li][j, idx[k]]))
        return np.array([fut.result()
                         for fut in futures]).reshape((lum[0].shape[0],
                                                       len(pis), len(vdirs)))
