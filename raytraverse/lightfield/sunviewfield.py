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
from scipy.spatial import cKDTree
from scipy.spatial.qhull import ConvexHull
from scipy.cluster.hierarchy import fclusterdata

from raytraverse import translate, plot
from raytraverse.helpers import ArrayDict
from raytraverse.lightfield.lightfield import LightField


class SunViewField(LightField):
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
    nullvlo = np.zeros((1, 5))

    def __init__(self, scene, suns, rebuild=False):
        #: np.array: sun positions
        self.suns = suns.suns
        #: raytraverse.sunmapper.SunMapper
        self.sunmap = suns.map
        super().__init__(scene, rebuild=rebuild, prefix='sunview')

    @property
    def vlo(self):
        """sunview data indexed by (point, sun)

        key (i, j) val: np.array (N, 5) x, y, z, lum, omega

        :type: raytraverse.translate.ArrayDict
        """
        return self._vlo

    @property
    def paths(self):
        """boundary paths indexed by (point, sun)

        key: (i, j) val: list of matplotlib.path.Path

        :type: dict
        """
        return self._paths

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
        dfile = f'{self.scene.outdir}/{self.prefix}_result.pickle'
        if not os.path.isfile(dfile):
            raise FileNotFoundError("No results files found, have you run"
                                    f" a Sampler of type {self.prefix} for"
                                    f" scene {self.scene.outdir}?")
        kdfile = f'{self.scene.outdir}/{self.prefix}_kd_data.pickle'
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._pt_kd = pickle.load(f)
            self.sun_kd = pickle.load(f)
            self._vlo = pickle.load(f)
            self._paths = pickle.load(f)
            f.close()
        else:
            self._pt_kd = cKDTree(self.scene.pts())
            self.sun_kd = cKDTree(self.suns)
            f = open(dfile, 'rb')
            vls = [pickle.load(f) for i in range(3)]
            f.close()
            self._vlo, self._paths = self._build_clusters(*vls)
            f = open(kdfile, 'wb')
            pickle.dump(self.pt_kd, f, protocol=4)
            pickle.dump(self.sun_kd, f, protocol=4)
            pickle.dump(self.vlo, f, protocol=4)
            pickle.dump(self.paths, f, protocol=4)
            f.close()

    def _cluster(self, vecs, lums, shape, suni):
        """group adjacent rays within sampling tolerance"""
        scalefac = ((self.sunmap.viewangle/2*np.pi/180)**2)
        omega0 = scalefac*np.pi/np.prod(shape)
        r0 = .5/shape[0]
        p0 = np.array([(-r0, -r0), (r0, -r0), (r0, r0), (-r0, r0)])
        maxr = 2*np.sqrt(2)/shape[0]
        xy = translate.uv2xy(vecs)*shape[0]/(shape[0] - 1)
        cluster = fclusterdata(vecs, maxr, 'distance')
        xyz = self.sunmap.uv2xyz(vecs, i=suni)
        vlo = []
        path = []
        for cidx in range(np.max(cluster)):
            grp = cluster == cidx + 1
            ptxy = xy[grp]
            ptxyz = xyz[grp]
            lgrp = lums[grp]
            uv = vecs[grp]
            if len(ptxy) > 2:
                ch = ConvexHull(ptxy)
                vlo.append([*np.mean(ptxyz, 0), np.mean(lgrp),
                            ch.volume*scalefac])
                path.append(uv[ch.vertices])
            else:
                for k in range(len(lgrp)):
                    vlo.append([*ptxyz[k], lgrp[k], omega0])
                    path.append(p0 + uv[k])
        return np.array(vlo), path

    def _build_clusters(self, vecs, lums, shape):
        """loop through points/suns and group adjacent rays"""
        vlo = ArrayDict({(-1, -1): self.nullvlo})
        paths = {(-1, -1): None}
        iterator = itertools.product(range(np.product(self.scene.ptshape)),
                                     range(self.suns.shape[0]))
        for i, j in iterator:
            if len(vecs[i][j]) > 0:
                v, p = self._cluster(vecs[i][j], lums[i][j], shape, j)
                vlo[(i, j)] = v
                paths[(i, j)] = p
        return vlo, paths

    def query(self, vpts, suns, vdirs=((0, 0, 1), ), viewangle=180.0, dtol=1.0,
              stol=10.0):
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

        Returns
        -------
        idxs: np.array
            shape: (pts, suns, views, 2) item[:, :, :, 0] is point index,
            item[:, :, :, 1] is sun index, returns (-1, -1) (null vector) if
            sun is not visible
        errs: np.array
            tuples of position and sun errors for each item, err=-1 where sun
            is not visible (or beyond tolerance)
        """
        vdirs = translate.norm(vdirs)
        suns = translate.norm(suns)
        a, b = np.broadcast_arrays(suns, vdirs[:, None, :])
        vizs = np.arccos(np.einsum('ijk,ijk->ij', a, b))*180/np.pi < viewangle/2
        stol = translate.theta2chord(stol*np.pi/180)
        with ProcessPoolExecutor() as exc:
            perrs, pis = zip(*exc.map(self._pt_kd.query, vpts))
            serrs, sis = zip(*exc.map(self.sun_kd.query, suns))
        idx = []
        errs = []
        iterator = itertools.product(zip(pis, perrs), zip(sis, serrs, vizs.T))
        for (pi, perr), (si, serr, viz) in iterator:
            mk = serr < stol and perr < dtol and (pi, si) in self.vlo.keys()
            for i in range(vdirs.shape[0]):
                if mk and viz[i]:
                    idx.append((pi, si))
                    errs.append((perr, serr))
                else:
                    idx.append((-1, -1))
                    errs.append((-1, -1))
        idxs = np.array(idx).reshape((-1, len(vdirs), 2))
        return idxs, np.array(errs)

    def direct_view(self, vpts):
        """create a summary image of all sun discs from each of vpts"""
        idx, errs = self.query(vpts, self.suns)
        ssq = int(np.ceil(np.sqrt(self.suns.shape[0])))
        square = np.array([[0,0], [0, 1], [1, 1], [1, 0]])
        block = []
        for i in np.arange(self.suns.shape[0], ssq**2):
            sxy = np.unravel_index(i, (ssq, ssq))
            sxy = np.array((sxy[1], sxy[0]))
            block.append(((1,1,1), square + sxy))
        for i, pt in enumerate(vpts):
            sidx = idx[idx[:, :, 0] == i]
            lums, fig, ax, norm, lev = plot.mk_img_setup([0,1], ext=(0, ssq))
            cmap = plot.colormap('viridis', norm)
            patches = block[:]
            for j in sidx:
                sxy = np.unravel_index(j[1], (ssq, ssq))
                sxy = np.array((sxy[1], sxy[0]))
                lums = cmap.to_rgba(self.vlo[j][:, 3])
                paths = self.paths[tuple(j)]
                xy = [(translate.uv2xy(p)+1)/2 + sxy for p in paths]
                patches += list(zip(lums, xy))
            plot.plot_patches(ax, patches)
            ax.set_facecolor((0, 0, 0))
            outf = f"{self.scene.outdir}_{self.prefix}_{i:04d}.png"
            plot.save_img(fig, ax, outf, title=pt)

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
            (N,) boolean array if sun has a match
        np.array
            (N,) index to proxy src
        """
        stol = translate.theta2chord(tol*np.pi/180)
        suns = translate.norm(tsuns)
        with ProcessPoolExecutor() as exc:
            serrs, sis = zip(*exc.map(self.sun_kd.query, suns))
        return serrs < stol, sis
