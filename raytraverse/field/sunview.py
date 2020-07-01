# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
from functools import reduce
import itertools

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.qhull import ConvexHull
from scipy.cluster.hierarchy import fclusterdata
from matplotlib.path import Path

from raytraverse import translate


class SunView(object):
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
        #: np.array: sun positions
        self.suns = suns.suns
        #: raytraverse.sunmapper.SunMapper
        self.sunmap = suns.map
        #: str: prefix
        self.prefix = 'sunview'
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
        self.iterator = itertools.product(range(np.product(scene.ptshape)),
                                          range(self.suns.shape[0]))
        self.scene = scene

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
            self._sun_kd = pickle.load(f)
            self._vlo = pickle.load(f)
            self._paths = pickle.load(f)
            f.close()
        else:
            self._pt_kd = cKDTree(self.scene.pts())
            self._sun_kd = cKDTree(self.suns)
            f = open(dfile, 'rb')
            vls = [pickle.load(f) for i in range(3)]
            f.close()
            self._vlo, self._paths = self.build_clusters(*vls)
            f = open(kdfile, 'wb')
            pickle.dump(self._pt_kd, f, protocol=4)
            pickle.dump(self._sun_kd, f, protocol=4)
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
                path.append(Path(uv[ch.vertices], closed=True))
            else:
                for k in range(len(lgrp)):
                    vlo.append([*ptxyz[k], lgrp[k], omega0])
                    path.append(Path(p0 + uv[k], closed=True))
        return np.array(vlo), path

    def build_clusters(self, vecs, lums, shape):
        """loop through points/suns and group adjacent rays"""
        vlo = {}
        paths = {}
        vlo[-1] = None
        paths[-1] = None
        for i, j in self.iterator:
            if len(vecs[i][j]) > 0:
                v, p = self._cluster(vecs[i][j], lums[i][j], shape, j)
                vlo[(i, j)] = v
                paths[(i, j)] = p
        return vlo, paths

    @property
    def vlo(self):
        """sunview data indexed by (point, sun)

        :type: dict key (i, j) val: np.array (N, 5) x, y, z, lum, omega
        """
        return self._vlo

    @property
    def paths(self):
        """boundary paths indexed by (point, sun)

        :type: dict key: (i, j) val: matplotlib.path.Path
        """
        return self._paths

    def query(self, vpts, suns, dtol=1.0, stol=10.0):
        """return indices for matching point and sun pairs

        Parameters
        ----------
        vpts: np.array
            points to search for (broadcastable to directions)
        suns: np.array
            sun positions to search for (broadcastable to directions)

        Returns
        -------
        idxs: list
            tuple indices to index self.lum and self.vec
        errs: list
            tuples of position and sun errors for each item
        pis: list np.array
            for each of vpts, the index of the returned point
        idxs: list of list of np.array
            for each of vpts, for each vdir, an array of all idxs in
            self.kd.data matching the query

        """
        stol = translate.theta2chord(stol*np.pi/180)
        with ProcessPoolExecutor() as exc:
            perrs, pis = zip(*exc.map(self._pt_kd.query, vpts))
            serrs, sis = zip(*exc.map(self._sun_kd.query, suns))
        idx = []
        errs = []
        for pi, perr in zip(pis, perrs):
            for si, serr in zip(sis, serrs):
                if (pi, si) in self.vlo.keys() and serr < stol and perr < dtol:
                    idx.append((pi, si))
                else:
                    idx.append(-1)
                errs.append((perr, serr))

        return idx, np.array(errs)

    def apply_coefs(self, pis, sis, coefs):
        lum = []
        idx = []
        coefs = np.broadcast_to(coefs, len(sis))
        for pi in pis:
            for i, si in enumerate(sis):
                idx.append((pi, si))
                lum.append(self.lum[pi][si] * coefs[i])
        return lum, idx
