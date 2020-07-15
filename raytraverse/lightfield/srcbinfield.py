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


class SrcBinField(LightField):
    """container for accessing sampled data where every ray has a value for
    each source
    """

    @property
    def vlo(self):
        """sky data indexed by (point)

        item per point: direction vector (3,) luminance (srcn,), omega (1,)

        :type: list of np.array
        """
        return self._vlo

    def _get_vl(self, npts, pref=''):
        dfile = f'{self.scene.outdir}/{self.prefix}{pref}_vals.out'
        vfile = f'{self.scene.outdir}/{self.prefix}{pref}_vecs.out'
        if not (os.path.isfile(dfile) and os.path.isfile(vfile)):
            raise FileNotFoundError("No results files found, have you run"
                                    f" a Sampler of type {self.prefix} for"
                                    f" scene {self.scene.outdir}?")
        fvecs = io.bytefile2np(open(vfile, 'rb'), (-1, 4))
        sorting = fvecs[:, 0].argsort()
        fvals = optic.rgb2rad(io.bytefile2np(open(dfile, 'rb'), (-1, 3)))
        fvals = fvals.reshape(fvecs.shape[0], -1)[sorting]
        fvecs = fvecs[sorting]
        pidx = fvecs[:, 0]
        pt_div = np.searchsorted(pidx, np.arange(npts), side='right')
        pt0 = 0
        vl = []
        for i, pt in enumerate(pt_div):
            vl.append(np.hstack((fvecs[pt0:pt, 1:4], fvals[pt0:pt])))
            pt0 = pt
        return vl

    def mk_tree(self):
        npts = np.product(self.scene.ptshape)
        vls = self._get_vl(npts)
        d_kd = []
        vlo = []
        for vl in vls:
            d_kd.append(cKDTree(vl[:, 0:3]))
            omega = SphericalVoronoi(vl[:, 0:3]).calculate_areas()[:, None]
            vlo.append(np.hstack((vl, omega)))
        return d_kd, vlo

    def measure(self, pi, vecs, coefs=1, interp=1):
        d, i = self.d_kd[pi].query(vecs, k=interp)
        srcn = self.scene.skyres**2
        coefs = np.asarray(coefs)
        if np.mod(coefs.size, srcn) == 0:
            c = coefs.reshape(-1, srcn)
        else:
            c = np.broadcast_to(coefs, (coefs.size, self.scene.skyres**2))
        lum = np.einsum('ij,kj->ik', c, self.vlo[pi][:, 3:-1])
        if interp > 1:
            wgts = np.broadcast_to(1/d, (lum.shape[0],) + d.shape)
            lum = np.average(lum[:, i], weights=wgts, axis=-1)
        else:
            lum = lum[:, i]
        print(lum.shape)
        return lum

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
        pt_kd = self.scene.pt_kd
        if treedir:
            dtree = cKDTree(vdirs)
        with ProcessPoolExecutor() as exc:
            errs, pis = zip(*exc.map(pt_kd.query, vpts))
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
            lum = np.log(np.maximum(np.sum(vlo[:, 3:-1], 1), 1e-6))
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
