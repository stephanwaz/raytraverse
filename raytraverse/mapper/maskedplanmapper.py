# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import re

import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

from raytraverse import translate
from raytraverse.mapper.planmapper import PlanMapper


class MaskedPlanMapper(PlanMapper):
    """translate between world positions on a horizontal plane and
    normalized UV space for a given view angle. pixel projection yields a
    parallel plan projection

    Parameters
    ----------
    pm: raytraverse.mapper.PlanMapper
        the source mapper to copy
    valid: np.array
        a list of valid points used to make a mask, grid cells not represented
        by one of valid will be masked
    level: int, optional
        the level at which to grid the valid candidates
    """

    def __init__(self, pm, valid, level):
        self.rotation = pm.rotation
        self.ptres = pm.ptres
        self._bbox = pm.bbox
        self._sf = pm._sf
        self._path = pm._path
        self._zheight = pm._zheight
        self._candidates = pm._candidates
        super(PlanMapper, self).__init__(name=pm.name, sf=self._sf,
                                         bbox=self.bbox)
        self._mask = None
        self._maskshape = None
        self.update_mask(valid, level)

    def update_mask(self, valid, level):
        self._maskshape = self.shape(level)
        uv = self.xyz2uv(valid)
        self._mask = np.unique(self.uv2idx(uv, self._maskshape))

    def in_view_uv(self, uv, indices=True, usemask=True):
        if not usemask:
            return super().in_view_uv(uv, indices)
        path = self._path
        uvs = uv.reshape(-1, 2)
        result = np.empty((len(path), uvs.shape[0]), bool)
        for i, p in enumerate(path):
            result[i] = p.contains_points(uvs)
        idx = self.uv2idx(uvs, self._maskshape)
        mask = np.logical_and(np.any(result, 0), np.isin(idx, self._mask))
        if indices:
            return np.unravel_index(np.arange(mask.size)[mask],
                                    uv.shape[:-1])
        else:
            return mask
