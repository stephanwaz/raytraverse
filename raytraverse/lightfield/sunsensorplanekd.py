# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

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

    def _load_points(self, idx, pts, samplelevel):
        self._sensors = None
        self._dataload = []
        s_pts, s_lev, s_idx, nmz = super()._load_points(idx, pts, samplelevel)
        s_idx = (self._dataload, s_idx)
        self._dataload = None
        return s_pts, s_lev, s_idx, nmz

    def _load_point(self, source):
        slp = SensorPlaneKD(self.scene, None, self.pm, source)
        if self._sensors is None:
            self._sensors = slp.sensors
        self._dataload.append(slp.data)
        spt = slp.vecs
        sidx = np.arange(slp.vecs.shape[0])
        slev = slp.samplelevel
        return spt, sidx, slev

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


