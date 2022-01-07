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
from raytraverse.evaluate import MetricSet
from raytraverse.lightfield.lightplanekd import LightPlaneKD
from raytraverse.lightfield.sets import MultiLightPointSet
from raytraverse.lightfield.lightfield import LightField
from raytraverse.lightfield.lightresult import ResultAxis, LightResult
import raytraverse.lightfield._helpers as intg
from raytraverse.mapper import ViewMapper
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
        self._normalization = sund / self.pm.ptres * 2 * 2**.5
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
        self._samplelevel = np.concatenate(s_lev)
        self.omega = None
        self.data = np.concatenate(s_idx)

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
