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
from raytraverse import translate, io, optic, plot
from raytraverse.mapper import ViewMapper
from raytraverse.lightfield.lightfield import LightField


class SCBinField(LightField):
    """container for accessing sampled data

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    rebuild: bool, optional
        build kd-tree even if one exists
    """

    def __init__(self, scene, rebuild=False):
        super().__init__(scene, rebuild=rebuild, prefix='sky')

    @property
    def vlo(self):
        """sunview data indexed by (point, sun)

        item per point: direction vector (3,) luminance (srcn,), omega (1,)

        :type: list of np.array
        """
        return self._vlo

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
    def svs(self):
        """spherical voronoi at each point

        :getter: Return list of spherical voronoi
        :type: list of scipy.spatial.SphericalVoronoi
        """
        return self._svs

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
        kdfile = f'{scene.outdir}/{self.prefix}_kd_data.pickle'
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._pt_kd = pickle.load(f)
            self._d_kd = pickle.load(f)
            self._vlo = pickle.load(f)
            self._svs = pickle.load(f)
            self._pidx = pickle.load(f)
            self._isort = pickle.load(f)
            f.close()
        else:
            self._pt_kd = cKDTree(self.scene.pts())
            dfile = f'{self.scene.outdir}/{self.prefix}_vals.out'
            vfile = f'{self.scene.outdir}/{self.prefix}_vecs.out'
            if not (os.path.isfile(dfile) and os.path.isfile(vfile)):
                raise FileNotFoundError("No results files found, have you run"
                                        f" a Sampler of type {self.prefix} for"
                                        f" scene {self.scene.outdir}?")
            (self._d_kd, self._vlo,
             self._svs, self._pidx,
             self._isort) = self.mk_tree(vfile, dfile)
            f = open(kdfile, 'wb')
            pickle.dump(self.pt_kd, f, protocol=4)
            pickle.dump(self.d_kd, f, protocol=4)
            pickle.dump(self.vlo, f, protocol=4)
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
        npts = np.product(self.scene.ptshape)
        pt_div = np.searchsorted(pidx, np.arange(npts), side='right')
        d_kd = []
        vlo = []
        svs = []
        pt0 = 0
        for i, pt in enumerate(pt_div):
            d_kd.append(cKDTree(fvecs[pt0:pt, 1:4]))
            try:
                svs.append(SphericalVoronoi(fvecs[pt0:pt, 1:4]))
                omega = svs[-1].calculate_areas()
            except ValueError as e:
                print(f'Warning, SphericalVoronoi not set at point {i}:')
                print(e)
                print(f'Source Solid angle calculation failed')
                omega = np.zeros(pt-pt0)
            vlo.append(np.vstack((fvecs[pt0:pt, 1:4].T, fvals[pt0:pt].T,
                                  omega.reshape(1, -1))).T)
            pt0 = pt
        return d_kd, vlo, svs, pidx, sorting

    def query(self, vpts, vdirs, viewangle=180.0, dtol=1.0, treecnt=30):
        """gather all rays from a point within a view cone

        Parameters
        ----------
        vpts: np.array
            points to search for (broadcastable to directions)
        vdirs: np.array
            directions to search for
        viewangle: float, optional
            degree opening of view cone
        dtol: float, optional
            distance tolerance for point query
        treecnt: int, optional
            number of queries at which a scipy.cKDtree.query_ball_tree is
            used instead of scipy.cKDtree.query_ball_point

        Returns
        -------
        idxs: np.array
            shape: (pts, views, 2) item[:, :, 0] is point index,
            item[:, :, 1] is np.array of vlo indices
        errs: np.array
            position error for each vpt
        """
        vdirs = translate.norm(vdirs)
        vs = translate.theta2chord(viewangle/360*np.pi)
        treedir = vdirs.shape[0] > treecnt
        if treedir:
            dtree = cKDTree(vdirs)
        with ProcessPoolExecutor() as exc:
            errs, pis = zip(*exc.map(self.pt_kd.query, vpts))
            futures = []
            for pi in pis:
                if treedir:
                    futures.append(exc.submit(dtree.query_ball_tree,
                                              self.d_kd[pi], vs))
                else:
                    futures.append(exc.submit(self.d_kd[pi].query_ball_point,
                                              vdirs, vs))
        idxs = []
        for pi, future in zip(pis, futures):
            for v in future.result():
                idxs.append([pi, np.array(v)])
        idxs = np.array(idxs).reshape(len(pis), len(vdirs), 2)
        return idxs, np.array(errs)

    def direct_view(self, vpts):
        """create a summary image for each of vpts"""
        idx, errs = self.query(vpts, ((0, -1, 0), ), 360)
        vm = ViewMapper((0,-1, 0))
        for i, pt in enumerate(vpts):
            vlo = self.vlo[idx[i, 0, 0]]
            lum = np.log(np.maximum(np.sum(vlo[:, 3:-1], 1), 1e-3))
            uv = vm.xyz2uv(vlo[:,0:3])
            outf = f"{self.scene.outdir}_{self.prefix}_{i:04d}.png"
            lums, fig, ax, norm, lev = plot.mk_img_setup(lum, figsize=[20, 10],
                                                         ext=((0,2), (0, 1)))
            ax.tricontourf(uv[:, 0], uv[:, 1], lums, norm=norm,
                           levels=lev, extend='both')
            ax.scatter(uv[:, 0], uv[:, 1], s=15,
                       marker='o',
                       facecolors='none', edgecolors=(1, 1, 1, .7),
                       linewidths=.25)
            plot.save_img(fig, ax, outf, title=pt)
