# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import pickle

import numpy as np

from scipy.spatial import cKDTree, Voronoi
from shapely.geometry import Polygon
from clasp.script_tools import try_mkdir


class LightPlaneKD(object):
    """light distribution from a point with KDtree structure for directional
    query

    Parameters
    ----------
    scene: raytraverse.scene.BaseScene
    points: np.array str
        points as array or file shape (N,3) or (N,4) if 3, indexed from 0
    pm: raytraverse.mapper.PlanMapper
    src: str, optional
        name of source group. will govern file naming so must be set to
        avoid clobbering writes.
    """

    def __init__(self, scene, points, pm=None, src='sky'):
        self._datadir = f"{scene.outdir}/{pm.name}/{src}"
        self.scene = scene
        self.pm = pm
        self.src = src
        self.points = points
        self._pt_kd = cKDTree(self.points)
        self.omega = None

    @property
    def points(self):
        """direction vector (N,3)"""
        return self._points

    @points.setter
    def points(self, pt):
        try:
            pts = np.loadtxt(pt)
        except TypeError:
            pts = pt
        try:
            pts = np.reshape(pts, (-1, 4))
            idx = pts[:, 0].astype(int)
            if not np.allclose(idx, pts[:, 0], atol=1e-4):
                raise ValueError
            pts = pts[:, 1:]
        except ValueError:
            pts = np.reshape(pts, (-1, 3))
            idx = np.arange(pts.shape[0])
        self._points = pts
        self._lplist = [f"{self._datadir}/{i:06d}.rytpt" for i in idx]

    @property
    def lp(self):
        """luminance (N,srcn)"""
        return self._lp

    @property
    def pt_kd(self):
        """point kdtree for spatial queries built on demand"""
        if self._pt_kd is None:
            self._pt_kd = cKDTree(self.points)
        return self._pt_kd

    @property
    def omega(self):
        """solid angle (N)

        :getter: Returns array of solid angles
        :setter: sets soolid angles with viewmapper
        :type: np.array
        """
        return self._omega

    @omega.setter
    def omega(self, oga):
        """calculate area"""
        if oga is None:
            pm = self.pm
            # border capture any infinite edges
            bordered = np.concatenate((self.points,
                                       pm.bbox_vertices(pm.area**.5 * 10)))
            vor = Voronoi(bordered[:, 0:2])
            omega = []
            for i in range(len(self.points)):
                region = vor.regions[vor.point_region[i]]
                p = Polygon(vor.vertices[region])
                area = 0
                for bord in pm.borders():
                    mask = Polygon(bord)
                    area += p.intersection(mask).area
                omega.append(area)
            self._omega = np.asarray(omega)
        else:
            self._omega = np.zeros(self.points.shape[0])

    def query_pt(self, points, clip=True):
        """return the index and distance of the nearest point to each of points

        Parameters
        ----------
        points: np.array
            shape (N, 3) positions to query.
        clip: bool, optional
            return d = 1e6 if not in_view

        Returns
        -------
        i: np.array
            integer indices of closest ray to each query
        d: np.array
            distance from query to point in spacemapper.
        """
        d, i = self.pt_kd.query(points)
        if clip:
            omask = self.pm.mask
            self.pm.mask = True
            d = np.where(self.pm.in_view(points, False), d, 1e6)
            self.pm.mask = omask
        return i, d

    def query_ball(self, pts, dist=1.0):
        """return set of points within a distance

        Parameters
        ----------
        pts: np.array
            shape (N, 3) points to query.
        dist: int float
            radius

        Returns
        -------
        i: list np.array
            if pts is a single point, a list of point indices of points
            within radius. if points is a set of points an array of lists, one
            for each point is returned.
        """
        return self.pt_kd.query_ball_point(pts, dist)

    def direct_view(self, res=512, showsample=False, showweight=True, rnd=False,
                    srcidx=None, interp=False, omega=False, scalefactor=1,
                    vm=None, fisheye=True):
        """create an unweighted summary image of lightplane"""
        pass
        # return outf

    def add(self, lf2, src=None, calcomega=True, write=False):
        """add light planes of distinct sources together
        """
        pass
        # return type(self)()
