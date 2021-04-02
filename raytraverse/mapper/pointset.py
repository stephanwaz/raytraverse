# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.spatial import cKDTree


class PointSet(object):
    """manage point based queries and indexing for a scene

    Parameters
    ----------
    pm: raytraverse.mapper.PlanMapper
    points: str np.array, optional
        np.array of points or path to points file loadable by np.loadtxt()
    precision: int, optional
        digits to round point coordinates
    """

    def __init__(self, pm, points=None, fill=False, jitter=True, precision=3):
        self.pm = pm
        self._pt_kd = None
        self.precision = precision
        self._points = np.empty((0, 3))
        if points is not None:
            # set level to match precision so no points are excluded for
            level = int(np.log2(self.pm.ptres/precision) - 1)
            self.add_points(points, level=level)
        if fill:
            self.add_points(self.pm.point_grid(jitter=jitter))

    @property
    def points(self):
        """points indexed in spacemapper"""
        return self._points

    @property
    def pt_kd(self):
        """point kdtree for spatial queries built at first use"""
        if self._pt_kd is None:
            self._pt_kd = cKDTree(self.points)
        return self._pt_kd

    def add_points(self, points, level=0):
        """add points to PlanMapper unless alrady present,
        saves points to file, resets kd-tree, and updates bounding box
        if necessary

        Parameters
        ----------
        points: np.array
            shape (N, 3)
        level: int, optional
            for setting tolerance level for excluding duplicates.
        """
        try:
            points = np.loadtxt(points).reshape(-1, 3)
        except (TypeError, ValueError):
            points = np.atleast_2d(points)
        idxs, ds = self.query_pt(points)
        tol = self.pm.ptres/2**(level + 1)
        newpts = np.round(points[ds > tol], self.precision)
        self._points = np.concatenate((self._points, newpts), 0)
        # reset pt_kd (rebuilt on next access)
        self._pt_kd = None

    def query_pt(self, points):
        """return the index and distance of the nearest point to each of points

        Parameters
        ----------
        points: np.array
            shape (N, 3) positions to query.

        Returns
        -------
        i: np.array
            integer indices of closest ray to each query
        d: np.array
            distance from query to point in spacemapper.
        """
        d, i = self.pt_kd.query(points)
        return i, d

