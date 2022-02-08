# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import shutil
import sys
from concurrent.futures import wait, FIRST_COMPLETED


import numpy as np
from scipy.spatial import cKDTree, distance_matrix

from raytraverse import translate
from raytraverse.lightfield.sets import MultiLightPointSet
from raytraverse.lightfield.lightfield import LightField
from raytraverse.translate import theta2chord, chord2theta
from raytraverse.utility import pool_call


class SunsPlaneKD(LightField):
    """collection of lightplanes with KDtree structure for sun position query
    """

    @property
    def vecs(self):
        """indexing vectors (sx, sy, sz, px, py, pz)"""
        return self._vecs

    @vecs.setter
    def vecs(self, pt):
        pts, idx, samplelevel = self._load_vecs(pt)
        # calculate sun sampling resolution for weighting query vecs
        s0 = pts[samplelevel == 0]
        dm = distance_matrix(s0, s0)
        cm = np.ma.MaskedArray(dm, np.eye(dm.shape[0]))
        sund = np.average(np.min(cm, axis=0).data)
        self._normalization = sund / (self.pm.ptres / 2 * 2**.5)
        s_pts = []
        s_idx = []
        s_lev = []
        for i, pt, sl in zip(idx, pts, samplelevel):
            source = f"{self.src}_{i:04d}"
            ptf = f"{self.scene.outdir}/{self.pm.name}/{source}_points.tsv"
            spt, sidx, slev = self._load_vecs(ptf)
            s_pts.append(np.hstack((np.broadcast_to(pt[None], spt.shape), spt)))
            s_idx.append(np.stack((np.broadcast_to([i], sidx.shape), sidx)).T)
            s_lev.append(np.stack((np.broadcast_to([sl], slev.shape), slev)).T)
        self._vecs = np.concatenate(s_pts)
        self._kd = None
        self._suns = pts
        self._sunkd = None
        self._samplelevel = np.concatenate(s_lev)
        self.omega = None
        self.data = np.concatenate(s_idx)

    @property
    def suns(self):
        return self._suns

    @property
    def data(self):
        """LightPlaneSet"""
        return self._data

    @data.setter
    def data(self, idx):
        self._data = MultiLightPointSet(self.scene, self.vecs, idx, self.src,
                                        self.pm.name)

    @property
    def kd(self):
        """kdtree for spatial queries built on demand"""
        if self._kd is None:
            weighted_vecs = np.copy(self.vecs)
            weighted_vecs[:, 3:] *= self._normalization
            self._kd = cKDTree(weighted_vecs)
        return self._kd

    @property
    def sunkd(self):
        """kdtree for sun position queries built on demand"""
        if self._sunkd is None:
            self._sunkd = cKDTree(self.suns)
        return self._sunkd

    def query(self, vecs):
        """return the index and distance of the nearest vec to each of vecs

        Parameters
        ----------
        vecs: np.array
            shape (N, 6) vectors to query.

        Returns
        -------
        i: np.array
            integer indices of closest ray to each query
        d: np.array
            distance from query to point, positional distance is normalized
            by the average chord-length between leveel 0 sun samples divided by
            the PlanMapper ptres * sqrt(2).
        """
        weighted_vecs = np.copy(np.atleast_2d(vecs))
        weighted_vecs[:, 3:] *= self._normalization
        d, i = self.kd.query(weighted_vecs)
        return i, d

    def query_by_sun(self, sunvec, fixed_points=None, ptfilter=.25, stol=10,
                     minsun=1):
        """for finding vectors across zone, sun vector based query

        Parameters
        ----------
        sunvec: Sequence
            sun direction vector (normalized, xyz)
        fixed_points: Sequence, optional
            2d array like, shape (N, 3) of additional fixed points to return
            use for example with a matching sky query. Note that if point filter
            is to large not all of these points are necessarily returned.
        ptfilter: Union[float, int], optional
            minimum seperation for returned points
        stol: Union[float, int], optional
            maximum angle (in degrees) for matching sun vectors
        minsun: int, optional
            if atleast these many suns are not returned based on stol, directly
            query for this number of results (regardless of sun error)

        Returns
        -------
        vecs: np.array
            shape (N, 6) final vectors, because of fixed_points, this may not
            match exactly with self.vecs[i] so this array mus be used in further
            processing
        i: np.array
            integer indices of the closest rays to each query
        d: np.array
            angle (in degrees) between queried sunvec and returned index

        """
        # find suns within tolerance or minimum number
        i = self.sunkd.query_ball_point(np.ravel(sunvec),
                                        theta2chord(stol*np.pi/180))
        if len(i) < minsun:
            d, i = self.sunkd.query(sunvec, minsun)
            if minsun == 1:
                i = [i]

        # filter points corresponding to these suns
        sunfilt = np.isin(self.data.idx[:, 0], i)
        vecs = self.vecs[sunfilt]
        idx = np.argwhere(sunfilt).ravel()

        # prepend fixed_points
        if fixed_points is not None:
            fixed_points = np.atleast_2d(fixed_points)
            pkd = cKDTree(vecs[:, 3:])
            d, i = pkd.query(fixed_points)
            fvecs = np.hstack((vecs[i, 0:3], fixed_points))
            vecs = np.concatenate((fvecs, vecs), axis=0)
            idx = np.concatenate((idx[i], idx))

        # sort points by ascending by sunerror
        sund = np.linalg.norm(vecs[:, 0:3] - np.atleast_2d(sunvec), axis=1)
        sorti = np.argsort(sund, kind='stable')

        # cull points within tolerance prioritizing minimum sun error
        flt = translate.cull_vectors(vecs[sorti, 3:], ptfilter)
        # sort back to original order and return indices with sun errors
        flt = flt[np.argsort(sorti, kind='stable')]
        d = chord2theta(sund[flt]) * 180/np.pi
        return vecs[flt], idx[flt], d

    def query_by_suns(self, sunvecs, fixed_points=None, ptfilter=.25, stol=10,
                      minsun=1):
        """parallel processing call to query_by_sun for 2d array of sunvecs

        Parameters
        ----------
        sunvecs: np.array
            shape (N, 3) sun direction vectors (normalized, xyz)
        fixed_points: Sequence, optional
            2d array like, shape (N, 3) of additional fixed points to return
            use for example with a matching sky query. Note that if point filter
            is to large not all of these points are necessarily returned.
        ptfilter: Union[float, int], optional
            minimum seperation for returned points
        stol: Union[float, int], optional
            maximum angle (in degrees) for matching sun vectors
        minsun: int, optional
            if atleast these many suns are not returned based on stol, directly
            query for this number of results (regardless of sun error)

        Returns
        -------
        vecs: list
            list of np.array, one for each sunvec (see query_by_sun)
        idx: list
            list of np.array, one for each sunvec (see query_by_sun)
        d: list
            list of np.array, one for each sunvec (see query_by_sun)

        """
        result = pool_call(self.query_by_sun, sunvecs, expandarg=False,
                           desc="finding SunPlane points",
                           fixed_points=fixed_points, ptfilter=ptfilter,
                           stol=stol, minsun=minsun)
        vecs, idx, d = zip(*result)
        return vecs, idx, d
