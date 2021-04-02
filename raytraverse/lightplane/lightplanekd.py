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

from scipy.spatial import cKDTree, SphericalVoronoi, Voronoi
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import Polygon
from clasp.script_tools import try_mkdir

from raytraverse import io, translate
from raytraverse.mapper import ViewMapper
from raytraverse.lightpoint.srcviewpoint import SrcViewPoint


class LightPlaneKD(object):
    """light distribution from a point with KDtree structure for directional
    query

    Parameters
    ----------
    scene: raytraverse.scene.BaseScene
    pm: raytraverse.mapper.PlanMapper, optional
    src: str, optional
        name of source group. will govern file naming so must be set to
        avoid clobbering writes.
    write: bool, optional
        whether to save ray data to disk.
    """

    def __init__(self, scene, points=None, pm=None, src='sky', write=True):
        try:
            #: str: out directory for saving points
            self.outdir = f"{scene.outdir}/{src}"
        except AttributeError:
            self.outdir = f"{scene}/{src}"

    def load(self):
        pts = np.loadtxt(f"{self.outdir}/points.tsv")
        self._points = pts.reshape(-1, 3)

    def dump(self):
        try_mkdir(f"{self.outdir}")
        np.savetxt(f"{self.outdir}/points.tsv", self.points,
                   f'%.{self.precision}f', '\t')

    @property
    def points(self):
        """direction vector (N,3)"""
        return self._points

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

    def calc_omega(self, write=True):
        """calculate area

        Parameters
        ----------
        write: bool, optional
            update/write kdtree data to file
        """
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
            for mask in pm.borders():
                area += p.intersection(mask).area
            omega.append(area)
        self._omega = np.asarray(omega)
        if write:
            self.dump()

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
        """return set of rays within a view cone

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
