# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.mapper.spacemapper import SpaceMapper


class SpaceMapperPt(SpaceMapper):
    """translate between world coordinates and normalized UV space"""

    @property
    def bbox(self):
        """np.array of shape (3,2): bounding box"""
        return self._bbox

    @property
    def sf(self):
        """bbox scale factor"""
        return self._sf

    @property
    def ptshape(self):
        """shape of point grid"""
        return self._ptshape

    @bbox.setter
    def bbox(self, plane):
        """read radiance geometry file as boundary path"""
        try:
            self._pts = np.loadtxt(plane)[:, 0:3]
        except IndexError:
            self._pts = np.loadtxt(plane)[0:3].reshape(1, 3)
        except TypeError:
            self._pts = plane.reshape(1, 3)
        self._bbox = np.stack((np.min(self._pts, 0) - self.tolerance,
                               np.max(self._pts, 0) - self.tolerance))
        self._sf = len(self._pts)
        self._ptshape = np.array([self._sf, 1])

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

        idx = (uv[:, 0] * self.sf).astype(int)
        return self._pts[idx]

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
        perrs, pis = self.pt_kd.query(xyz)
        uv = np.full((xyz.shape[0], 2), .5)
        uv[:, 0] = (pis + .5) / self.sf
        return uv

    def idx2pt(self, idx):
        return self._pts[idx]

    def pts(self):
        return self._pts

    def in_area(self, xyz):
        """check if point is in boundary path

        Parameters
        ----------
        xyz: np.array
            uv coordinates, shape (N, 3)

        Returns
        -------
        mask: np.array
            boolean array, shape (N,)
        """
        perrs, pis = self.pt_kd.query(xyz)
        return perrs <= self.tolerance
