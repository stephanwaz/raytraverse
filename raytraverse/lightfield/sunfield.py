# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
import itertools

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi
from scipy.spatial.qhull import ConvexHull
from scipy.cluster.hierarchy import fclusterdata

from raytraverse import translate, plot, io, optic
from raytraverse.lightfield.srcbinfield import SrcBinField
from raytraverse.lightfield.sunviewfield import SunViewField
from raytraverse.mapper import ViewMapper


class SunField(SrcBinField):
    """container for sun view data

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
        #: np.array: sun positions
        self.suns = suns.suns
        #: raytraverse.lightfield.SunViewField
        self.view = SunViewField(scene, suns, rebuild=rebuild)
        self._pt_kd = self.view.pt_kd
        super().__init__(scene, rebuild=rebuild, prefix='sun')

    @property
    def vlo(self):
        """sun data indexed by (point, sun)

        key (i, j) val: direction vector (3,) luminance (srcn,), omega (1,)

        :type: raytraverse.translate.ArrayDict
        """
        return self._vlo

    def mk_tree(self):
        npts = np.product(self.scene.ptshape)
        vlamb = self._get_vl(npts, pref='ambient')
        d_kd = {(-1, -1): None}
        vlo = translate.ArrayDict({(-1, -1): None})
        svs = {(-1, -1): None}
        for i in range(self.suns.shape[0]):
            vlsun = self._get_vl(npts, pref=f'_{i:04d}')
            for j in range(npts):
                vecs = np.vstack((vlamb[j][:, 0:3], vlsun[j][:, 0:3]))
                lums = np.hstack((vlamb[j][:, i + 3], vlsun[j][:, 3]))[:, None]
                svs[(j, i)] = SphericalVoronoi(vecs)
                omega = svs[(j, i)].calculate_areas()[:, None]
                vlo[(j, i)] = np.hstack((vecs, lums, omega))
                d_kd[(j, i)] = cKDTree(vecs)
        return d_kd, vlo, svs

    def query(self, vpts, suns, vdirs=((0, 0, 1), ), viewangle=180.0, dtol=1.0,
              stol=10.0, treecnt=30):
        """return indices for matching point and sun pairs

        Parameters
        ----------
        vpts: np.array
            points to search for
        suns: np.array
            sun positions to search for
        vdirs: np.array, optional
            directions to search for
        viewangle: float, optional
            degree opening of view cone
        dtol: float, optional
            distance tolerance for point query
        stol: float, optional
            angle (degrees) tolorence for direct sun query
        treecnt: int, optional
            number of queries at which a scipy.cKDtree.query_ball_tree is
            used instead of scipy.cKDtree.query_ball_point

        Returns
        -------
        idxs: np.array
            shape: (pts, suns, views, 2) item[:, :, :, 0] is point index,
            item[:, :, :, 1] is sun index, returns (-1, -1) (null vector) if
            sun is not visible
        errs: np.array
            tuples of position and sun errors for each item, err=-1 where sun
            is beyond tolerance
        """
        vdirs = translate.norm(vdirs)
        suns = translate.norm(suns)
        vs = translate.theta2chord(viewangle/360*np.pi)
        stol = translate.theta2chord(stol*np.pi/180)
        treedir = vdirs.shape[0] > treecnt
        if treedir:
            dtree = cKDTree(vdirs)
        with ProcessPoolExecutor() as exc:
            perrs, pis = zip(*exc.map(self._pt_kd.query, vpts))
            serrs, sis = zip(*exc.map(self.view.sun_kd.query, suns))
            errs = []
            idx = []
            iterator = itertools.product(zip(pis, perrs), zip(sis, serrs))
            fu = []
            for (pi, perr), (si, serr) in iterator:
                errs.append((perr, serr))
                idx.append((pi, si))
                if treedir:
                    fu.append(exc.submit(dtree.query_ball_tree,
                                         self.d_kd[(pi, si)], vs))
                else:
                    fu.append(exc.submit(self.d_kd[(pi, si)].query_ball_point,
                                         vdirs, vs))
        idxs = []
        for i, future in zip(idx, fu):
            for v in future.result():
                idxs.append([*i, np.array(v)])
        idxs = np.array(idxs).reshape((-1, len(vdirs), 3))
        return idxs, np.array(errs)

    def direct_view(self, vpts):
        """create a summary image for each of vpts"""
        vdirs = ((0, -1, 0), (0, 1, 0))
        idxs, errs = self.query(vpts, self.suns, vdirs)
        ssq = int(np.ceil(np.sqrt(self.suns.shape[0])))
        self.view.direct_view(vpts)
        for v in range(2):
            vm = ViewMapper(vdirs[v])
            for i, pt in enumerate(vpts):
                sidx = idxs[:, v, :][idxs[:, v, 0] == i]
                outf = f"{self.scene.outdir}_{self.prefix}_{i:04d}_{v}.png"
                lums, fig, ax, norm, lev = plot.mk_img_setup([-8, -2], figsize=[10, 10],
                                                             ext=((0, ssq), (0, ssq)))
                # cmap = plot.colormap('viridis', norm)
                # patches = []
                for j in sidx:
                    vlo = self.vlo[(j[0], j[1])][j[2]]
                    lum = np.log10(np.maximum(np.sum(vlo[:, 3:-1], 1), 1e-8))
                    sxy = np.unravel_index(j[1], (ssq, ssq))
                    sxy = np.array((sxy[1], sxy[0]))
                    uv = vm.xyz2xy(vlo[:, 0:3])/2 + sxy + .5
                    ax.tricontourf(uv[:, 0], uv[:, 1], lum, norm=norm, levels=lev,
                                   extend='both', zorder=-1)
                    # try:
                    #     vlums = cmap.to_rgba(self.view.vlo[(j[0], j[1])][:, 3])
                    #     paths = self.view.paths[(j[0], j[1])]
                    #     xy = [vm.xyz2xy(self.view.sunmap.uv2xyz(p, j[1]))/2 + sxy + .5 for p in
                    #           paths]
                    #     patches += list(zip(vlums, xy))
                    # except KeyError:
                    #     pass
                # plot.plot_patches(ax, patches, lw=10)
                # ax.set_facecolor((0, 0, 0))
                plot.save_img(fig, ax, outf, title=pt)
