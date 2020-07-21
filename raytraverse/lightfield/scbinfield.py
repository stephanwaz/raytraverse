# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi

from raytraverse.lightfield.lightfieldkd import LightFieldKD


class SCBinField(LightFieldKD):
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

    def apply_coef(self, pi, coefs):
        srcn = self.scene.skyres**2
        coefs = np.asarray(coefs)
        if np.mod(coefs.size, srcn) == 0:
            c = coefs.reshape(-1, srcn)
        else:
            c = np.broadcast_to(coefs, (coefs.size, srcn))
        return super().apply_coef(pi, c)

