# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.spatial import cKDTree
from raytraverse import translate
from raytraverse.lightpoint.lightpointkd import LightPointKD


class SunPointKD(LightPointKD):
    """removes stray rays from accidental direct sun hits and incorporates
    sunviewsampler results"""

    def __init__(self, scene, vec=None, lum=None, sun=(0, 0, 0), **kwargs):
        self.sunpos = translate.norm1(np.asarray(sun).flatten()[0:3])
        super().__init__(scene, vec, lum, **kwargs)

    def _build(self, vec, lum, srcn):
        """load samples and build data structure"""
        d_kd, vec, lum, clear = super()._build(vec, lum, srcn)
        blind_squirrel = d_kd.query_ball_point(self.sunpos, 0.0046513)
        if len(blind_squirrel) > 0:
            vec = np.delete(vec, blind_squirrel)
            lum = np.delete(lum, blind_squirrel)
            d_kd = cKDTree(vec)
        return d_kd, vec, lum, clear
