# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

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

    def apply_coef(self, pi, coefs):
        srcn = self.scene.skyres**2
        coefs = np.asarray(coefs)
        if np.mod(coefs.size, srcn) == 0:
            c = coefs.reshape(-1, srcn)
        else:
            c = np.broadcast_to(coefs, (coefs.size, srcn))
        return np.einsum('ij,kj->ik', c, self.vlo[pi][:, 3:-1])

