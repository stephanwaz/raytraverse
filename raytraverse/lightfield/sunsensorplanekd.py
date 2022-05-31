# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.spatial import distance_matrix

from raytraverse.lightfield.sensorplanekd import SensorPlaneKD
from raytraverse.lightfield.sets import SensorPointSet
from raytraverse.lightfield.sunsplanekd import SunsPlaneKD


class SunSensorPlaneKD(SunsPlaneKD):
    """collection of sensorplanes with KDtree structure for sun position query

    data has shape (pts * suns, sensors, sources, bands)
    """

    @property
    def sensors(self):
        return self._sensors

    @property
    def vecs(self):
        """indexing vectors (sx, sy, sz, px, py, pz)"""
        return self._vecs

    @vecs.setter
    def vecs(self, pt):
        if pt is None:
            pt = f"{self.scene.outdir}/{self.pm.name}/{self.src}positions.tsv"
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
        data = []
        self._sensors = None
        for i, pt, sl in zip(idx, pts, samplelevel):
            stype = f"{self.src}_{i:04d}"
            file = f"{self.scene.outdir}/{self.pm.name}/{stype}.npz"
            slp = SensorPlaneKD(self.scene, file, self.pm, stype)
            if self._sensors is None:
                self._sensors = slp.sensors
            spt = slp.vecs
            sidx = np.arange(slp.vecs.shape[0])
            slev = slp.samplelevel
            data.append(slp.data)
            s_pts.append(np.hstack((np.broadcast_to(pt[None], spt.shape), spt)))
            s_idx.append(np.stack((np.broadcast_to([i], sidx.shape), sidx)).T)
            s_lev.append(np.stack((np.broadcast_to([sl], slev.shape), slev)).T)
        self._vecs = np.concatenate(s_pts)
        self._kd = None
        self._suns = pts
        self._sunkd = None
        self._samplelevel = np.concatenate(s_lev)
        self.omega = None
        self.data = (data, np.concatenate(s_idx))

    @property
    def suns(self):
        return self._suns

    @property
    def data(self):
        """LightPlaneSet"""
        return self._data

    @data.setter
    def data(self, idx):
        self._data = SensorPointSet(*idx)

    @staticmethod
    def apply_coef(data, coefs):
        """apply coefficient vector to data

        Parameters
        ----------
        data: np.array
            ndims should match self.data (N, M, nsrcs, nfeatures)
        coefs: np.array int float list
            shape (L, self.srcn) or broadcastable

        Returns
        -------
        alum: np.array
            shape (L, N, M, nfeatures)
        """
        features = data.shape[-1]
        srcn = data.shape[-2]
        if features > 1:
            try:
                c = np.asarray(coefs).reshape(-1, srcn, features)
            except ValueError:
                c = np.broadcast_to(coefs, (1, srcn, features))
            sstring = 'abc,debc->adec'
        else:
            try:
                c = np.asarray(coefs).reshape(-1, srcn)
            except ValueError:
                c = np.broadcast_to(coefs, (1, srcn))
            sstring = 'ab,debc->adec'
        return np.einsum(sstring, c, data)


