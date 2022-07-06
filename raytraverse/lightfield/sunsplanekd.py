# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.spatial import cKDTree, distance_matrix

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
        if pt is None:
            pt = f"{self.scene.outdir}/{self.pm.name}/{self.src}positions.tsv"
        pts, idx, samplelevel = self._load_vecs(pt)
        s_pts, s_lev, s_idx, nmz = self._load_points(idx, pts, samplelevel)
        self._normalization = nmz
        self._vecs = s_pts
        self._samplelevel = s_lev
        self.data = s_idx
        self._kd = None
        self._suns = pts
        self._sunkd = None
        self.omega = None

    def _load_points(self, idx, pts, samplelevel):
        """loop over source point files to build indices"""
        s_pts = []
        s_idx = []
        s_lev = []
        pdist = []
        # fall back to guesstimate of level 0 in case component planes
        # do not have levels indexed
        l0 = self.pm.point_grid(False).shape[0]
        haslevels = np.sum(samplelevel) > 0
        if haslevels:
            # calculate sun sampling resolution for weighting query vecs
            sund = self._estimate_grid_size(pts[samplelevel == 0])
        else:
            sund = self._estimate_grid_size(pts[:45])

        for i, pt, sl in zip(idx, pts, samplelevel):
            spt, sidx, slev = self._load_point(f"{self.src}_{i:04d}")
            # only the unmasked level 0 suns will provide a good
            # estimate for level0 point distances
            if haslevels and sl == 0:
                if np.sum(slev) > 0:
                    s = spt[slev == 0]
                else:
                    s = spt[:l0]
                pdist.append(self._estimate_grid_size(s))

            s_pts.append(np.hstack((np.broadcast_to(pt[None], spt.shape), spt)))
            s_idx.append(np.stack((np.broadcast_to([i], sidx.shape), sidx)).T)
            s_lev.append(np.stack((np.broadcast_to([sl], slev.shape), slev)).T)
        if haslevels:
            normalization = sund/np.average(pdist)
        else:
            pdist = self.pm.ptres * (np.sqrt(2)/2+.5)
            normalization = sund/pdist
        s_pts = np.concatenate(s_pts)
        s_lev = np.concatenate(s_lev)
        s_idx = np.concatenate(s_idx)
        return s_pts, s_lev, s_idx, normalization

    def _load_point(self, source):
        ptf = f"{self.scene.outdir}/{self.pm.name}/{source}_points.tsv"
        return self._load_vecs(ptf)

    @staticmethod
    def _estimate_grid_size(v):
        dm = distance_matrix(v, v)
        cm = np.ma.MaskedArray(dm, np.eye(dm.shape[0]))
        return np.average(np.min(cm, axis=0).data)

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
            by the average chord-length between level 0 sun samples divided by
            the average distance between level 0 pt samples.
        """
        weighted_vecs = np.copy(np.atleast_2d(vecs)).astype(float)
        weighted_vecs[:, 3:] *= self._normalization
        d, i = self.kd.query(weighted_vecs)
        return i, d

    def query_by_sun(self, sunvec, fixed_points=None, stol=10,
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
        return _query_by_sun(sunvec, self.sunkd, self.vecs, self.data.idx[:, 0],
                             fixed_points=fixed_points,
                             stol=stol, minsun=minsun,
                             normalization=self._normalization)

    def query_by_suns(self, sunvecs, fixed_points=None, stol=10,
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
        result = pool_call(_query_by_sun, sunvecs, self.sunkd, self.vecs,
                           self.data.idx[:, 0], expandarg=False,
                           desc="finding SunPlane points",
                           pbar=self.scene.dolog, fixed_points=fixed_points,
                           stol=stol, minsun=minsun,
                           normalization=self._normalization)
        vecs, idx, d = zip(*result)
        return vecs, idx, d


def _query_by_sun(sunvec, sunkd, svecs, sidx, fixed_points=None,
                  stol=10, minsun=1, normalization=1.0):
    """for finding vectors across zone, sun vector based query"""

    # first find all points with suns within tolerance
    cstol = theta2chord(stol*np.pi/180)
    i = sunkd.query_ball_point(np.ravel(sunvec), cstol)

    if len(i) < minsun:
        d, i = sunkd.query(sunvec, minsun)
        if minsun == 1:
            i = [i]

    # filter points corresponding to these suns
    sunfilt = np.isin(sidx, i)
    vecs = svecs[sunfilt]
    idx = np.argwhere(sunfilt).ravel()
    # prepend fixed_points (using closest values)
    if fixed_points is not None:
        fixed_points = np.atleast_2d(fixed_points)
        pkd = cKDTree(vecs[:, 3:])
        d, i = pkd.query(fixed_points)
        fvecs = np.hstack((vecs[i, 0:3], fixed_points))
        vecs = np.concatenate((fvecs, vecs), axis=0)
        idx = np.concatenate((idx[i], idx))

    # calculate sun errors
    sund = np.linalg.norm(vecs[:, 0:3] - np.atleast_2d(sunvec), axis=1)
    cpts = np.copy(vecs[:, 3:])
    # add sun error penalty
    cpts[:, 2] += sund/normalization
    ckd = cKDTree(cpts)
    # reassign original vectors based on penalized query
    d, i = ckd.query(vecs[:, 3:])
    # eliminate unused vectors
    i = np.unique(i)
    d = chord2theta(sund[i]) * 180/np.pi
    return vecs[i], idx[i], d
