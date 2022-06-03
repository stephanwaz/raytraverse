# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate
from raytraverse.lightfield.lightplanekd import LightPlaneKD


class SensorPlaneKD(LightPlaneKD):
    """collection of sensor results with KDtree structure for positional
    query

    data has shape (pts, sensors, sources, bands)
    """

    @property
    def sensors(self):
        return self._sensors

    @property
    def vecs(self):
        """indexing vectors (such as position, sun positions, etc.)"""
        return self._vecs

    @vecs.setter
    def vecs(self, pt):
        if pt is None:
            pt = f"{self.scene.outdir}/{self.pm.name}/{self.src}.npz"
        with np.load(pt) as result:
            self._vecs = result['vecs']
            self._sensors = result['sensors']
            self.data = result['lum']
        self._kd = None
        if self._vecs.shape[-1] > 3:
            self._samplelevel = self.vecs[:, 0]
            self._vecs = self._vecs[:, -3:]
        else:
            self._samplelevel = np.zeros(len(self.vecs))
        self.omega = None

    @property
    def data(self):
        """light data"""
        return self._data

    @data.setter
    def data(self, lum):
        self._data = lum

    @staticmethod
    def apply_coef(data, coefs):
        """apply coefficient vector to data

        Parameters
        ----------
        data: np.array
            ndims should match self.data (N, sensors, nsrcs, nfeatures)
        coefs: np.array int float list
            shape (L, self.srcn) or broadcastable

        Returns
        -------
        alum: np.array
            shape (L, N, sensors, nfeatures)
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

    def evaluate(self, skyvec, points=None, sensoridx=None, mask=True,
                 **kwargs):
        if points is None:
            qidx = np.arange(len(self.vecs))
        else:
            qidx, d = self.query(points)
            if mask:
                omask = self.pm.mask
                self.pm.mask = True
                qidx = qidx[self.pm.in_view(points, False)]
                self.pm.mask = omask
        if len(qidx) > len(self.data):
            data = translate.simple_take(self.data, sensoridx, axes=1)
            r = self.apply_coef(data, skyvec)[:, qidx]
        else:
            data = translate.simple_take(self.data, qidx, sensoridx)
            r = self.apply_coef(data, skyvec)
        return r

    def direct_view(self, res=512, showsample=True, area=False,
                    interp=False, sensoridx=None, **kwargs):
        """create a summary image of lightplane showing samples and areas"""
        if area:
            outf = self._datadir.replace("/", "_") + f"{self.src}_area.hdr"
            self.make_image(outf, self.omega, res=res, showsample=showsample,
                            interp=False)
        result = self.evaluate(1, sensoridx=sensoridx)
        if sensoridx is None:
            sensoridx = np.arange(result.shape[-2])
        else:
            sensoridx = np.ravel(sensoridx)
        for i, j in enumerate(sensoridx):
            vd = "vd-{:.1f}_{:.1f}_{:.1f}".format(*self.sensors[j, 3:])
            outf = self._datadir.replace("/", "_") + f"{self.src}_{vd}.hdr"
            self.make_image(outf, result[0, :, i, 0], res=res,
                            showsample=showsample, interp=interp)
