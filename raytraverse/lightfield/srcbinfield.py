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

from raytraverse import translate, optic
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

    def _mk_tree(self):
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
            c = np.broadcast_to(coefs, (coefs.size, srcn))
        lum = np.einsum('ij,kj->ik', c, self.vlo[pi][:, 3:-1])
        if interp > 1:
            wgts = np.broadcast_to(1/d, (lum.shape[0],) + d.shape)
            lum = np.average(lum[:, i], weights=wgts, axis=-1)
        else:
            lum = lum[:, i]
        return np.squeeze(lum)

    def gather(self, pi, vecs, coefs=1, viewangle=180):
        vs = translate.theta2chord(viewangle/360*np.pi)
        i = self.d_kd[pi].query_ball_point(translate.norm1(vecs), vs)
        srcn = self.scene.skyres**2
        coefs = np.asarray(coefs)
        if np.mod(coefs.size, srcn) == 0:
            c = coefs.reshape(-1, srcn)
        else:
            c = np.broadcast_to(coefs, (coefs.size, srcn))
        lum = np.einsum('ij,kj->ik', c, self.vlo[pi][i, 3:-1])
        vec = self.vlo[pi][i, 0:3]
        omega = self.vlo[pi][i, -1]
        return lum, vec, omega
