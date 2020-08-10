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

from raytraverse.mapper.spacemapper import SpaceMapper


class SpaceMapperPt(SpaceMapper):
    """translate between world coordinates and normalized UV space"""

    def __init__(self, pts):
        #: float: ccw rotation (in degrees) for point grid on plane
        self.rotation = 0.0
        self._path = None
        self._sf = 1
        #: np.array: points [(x, y, z), ...]
        self.bbox = pts

    @property
    def bbox(self):
        """np.array of shape (3,2): bounding box"""
        return self._bbox

    @property
    def path(self):
        """list of matplotlib.path.Path: boundary paths"""
        return self._path

    @property
    def sf(self):
        """bbox scale factor"""
        return self._sf

    @bbox.setter
    def bbox(self, plane):
        """read radiance geometry file as boundary path"""
        paths, bbox = self._rad_scene_to_bbox(plane)
        self._bbox = bbox

    def uv2pt(self, uv):
        """convert UV --> world

        Parameters
        ----------
        uv: np.array
            normalized UV coordinates of shape (N, 2)

        Returns
        -------
        pt: np.array
            world xyz coordinates of shape (N, 3)
        """
        sf = self.bbox[1] - self.bbox[0]
        uv = np.hstack((uv, np.zeros(len(uv)).reshape(-1, 1)))
        pt = self.bbox[0] + uv * sf
        return self.ro_pts(pt)

    def pt2uv(self, xyz):
        """convert world --> UV

        Parameters
        ----------
        xyz: np.array
            world xyz coordinates, shape (N, 3)

        Returns
        -------
        uv: np.array
            normalized UV coordinates of shape (N, 2)
        """
        uv = (self.ro_pts(xyz, rdir=1) - self.bbox[0])[:, 0:2] / self._sf
        return uv
