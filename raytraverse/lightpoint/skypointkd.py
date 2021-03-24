# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import functools

import numpy as np
from raytraverse.lightpoint.lightpointkd import LightPointKD

from raytraverse import translate


class SkyPointKD(LightPointKD):
    """is aware of sky discretization scene (can recover source directions for
    compression and source extraction)"""

    @property
    @functools.lru_cache(1)
    def srcdir(self):
        return translate.skybin2xyz(np.arange(self.srcn),
                                    np.sqrt(self.srcn - 1))

    def source_weighted_vector(self, weight=4):
        """perform cluster fit with the Birch algoritm on a 4D vector
        using lf.vec and uniform weighted lf.lum

        Parameters
        ----------
        weight: float, optional
            coefficient for min-max normalized luminances, bigger values weight
            luminance more strongly compared to vector direction, meaning with
            higher numbers clusters will have less variance in luminance.

        Returns
        -------
        weighted_vector: np.array
            (N, 7) ray direction, source direction and source brightness
        """
        v1 = super().source_weighted_vector(weight)
        src = np.einsum('jk,ij->ik', self.srcdir, self.lum)
        return np.hstack((v1[:, 0:3], src, v1[:, -1:]))



