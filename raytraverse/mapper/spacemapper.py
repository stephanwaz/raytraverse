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
from scipy.spatial import cKDTree, ConvexHull
from clasp import script_tools as cst


class SpaceMapper(object):
    """manage point based queries and indexing for a scene

    Parameters
    ----------
    scene: raytraverse.scene.Scene raytraverse.scene.BaseScene str
        scene object or out directory for saving points
    area: str, optional
        radiance scene geometry defining a plane to sample
    mask: bool, optional
        if False, do not constrain points to border
    reload: bool, optional
        include scene/points.dat if it exists.
    ptres: float, optional
        resolution for considering points duplicates, border generation
        (1/2) and add_grid(). updateable
    rotation: float, optional
        positive Z rotation for point grid alignment
    fill: bool, optional
        create a full grid of points spaced by ptres on initialization
        (calls self.add_grid())
    precision: int, optional
        digits to round point coordinates
    """

    def __init__(self, scene, points=None, area=None, mask=True,
                 reload=True, ptres=1.0, rotation=0.0, fill=False, precision=3):
        try:
            #: str: out directory for saving points
            self.outdir = scene.outdir
        except AttributeError:
            self.outdir = scene
            cst.try_mkdir(scene)
        self._bbox = None
        #: float: ccw rotation (in degrees) for point grid on plane
        self.rotation = rotation
        #: float: point resolution for area look ups
        self.ptres = ptres
        self._pt_kd = None
        self.precision = precision
        #: bool: wheether to test points against border paths
        self.mask = mask
        if reload:
            self.points = f"{self.outdir}/points.dat"
        else:
            self._points = np.empty((0, 3))
        if area is not None:
            self.update_bbox(area)
        if points is not None:
            self.add_points(points)
        if fill:
            self.add_grid()

    @property
    def bbox(self):
        """np.array: boundary frame for translating between coordinates
        [[xmin ymin zmin] [xmax ymax zmax]]"""
        return self._bbox

    @property
    def points(self):
        """points indexed in spacemapper"""
        return self._points

    @points.setter
    def points(self, arg):
        """set points from a file"""
        try:
            self._points = np.reshape(np.loadtxt(arg), (-1, 3))
        except (ValueError, TypeError, IOError):
            self._points = np.empty((0, 3))

    @property
    def pt_kd(self):
        """point kdtree for spatial queries built at first use"""
        if self._pt_kd is None:
            self._pt_kd = cKDTree(self.points)
        return self._pt_kd

    @property
    def sf(self):
        """bbox scale factor"""
        return self._sf

    def update_bbox(self, plane):
        """read radiance geometry file as boundary path"""
        if plane is not None:
            paths, z = self._rad_scene_to_paths(plane)
        else:
            paths, z = self._calc_border()
        bbox = np.squeeze([np.amin(paths, 1), np.amax(paths, 1)])
        bbox = np.hstack([bbox, [[z], [z]]])
        self._bbox = bbox
        self._sf = self.bbox[1, 0:2] - self.bbox[0, 0:2]
        self._path = []
        for pt in paths:
            p = (np.concatenate((pt, [pt[0]])) - bbox[0, 0:2])/self.sf
            xy = Path(p, closed=True)
            self._path.append(xy)

    def add_points(self, points):
        points = np.atleast_2d(points)
        ds, idxs = self.pt_kd.query(points)
        newpts = points[ds > self.ptres]
        if self.bbox is not None:
            newpts = np.round(newpts[self.in_area(newpts)], self.precision)
        self._points = np.concatenate((self._points, newpts), 0)
        self._pt_kd = None
        np.savetxt(f"{self.outdir}/points.dat", self.points,
                   f'%.{self.precision}f', '\t')
        if self.bbox is None or not self.mask:
            self.update_bbox(None)

    def add_grid(self, jitter=True):
        shape = np.ceil(self.sf/self.ptres).astype(int)
        idx = np.arange(np.product(shape))
        si = np.stack(np.unravel_index(idx, shape)).T
        if jitter:
            offset = np.random.default_rng().random(si.shape)
        else:
            offset = 0.5
        self.add_points(self.uv2pt((si + offset)/shape))

    def _ro_pts(self, points, rdir=-1):
        """
        rotate points

        Parameters
        ----------
        points: np.ndarray
            world coordinate points of shape (N, 3)
        rdir: -1 or 1
            rotation direction:
                -1 to rotate from uv space
                1 to rotate to uvspace

        Returns
        -------
        """
        cos = np.cos(rdir * self.rotation * np.pi / 180)
        sin = np.sin(rdir * self.rotation * np.pi / 180)
        rmtx = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        rm = np.matmul(rmtx, points.T)
        return rm.T

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
        uv = np.hstack((uv, np.zeros(len(uv)).reshape(-1, 1)))
        pt = self.bbox[0] + uv * [*self.sf, 0]
        return self._ro_pts(pt)

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
        uv = (self._ro_pts(xyz, rdir=1) - self.bbox[0])[:, 0:2] / self.sf
        return uv

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
        uv = self.pt2uv(xyz)
        path = self._path
        if path is None or not self.mask:
            return np.full((uv.shape[0]), True)
        else:
            result = np.empty((len(path), uv.shape[0]), bool)
            for i, p in enumerate(path):
                result[i] = p.contains_points(uv)
        return np.any(result, 0)

    def query_pt(self, points, clip=True):
        """return the index and distance of the nearest point to each of points

        Parameters
        ----------
        points: np.array
            shape (N, 3) positions to query.
        clip: bool, optional
            return d = 1e6 if not in_area

        Returns
        -------
        i: np.array
            integer indices of closest ray to each query
        d: np.array
            distance from query to point in spacemapper.
        """
        d, i = self.pt_kd.query(points)
        if clip:
            omask = self.mask
            self.mask = True
            d = np.where(self.in_area(points), d, 1e6)
            self.mask = omask
        return i, d

    def _rad_scene_to_paths(self, plane):
        with open(plane, 'r') as f:
            dl = [i for i in re.split(r'[\n\r]+', f.read().strip())
                  if not bool(re.match(r'#', i))]
            rad = " ".join(dl).split()
        pgs = [i for i, x in enumerate(rad) if x == "polygon"]
        z = -1e10
        paths = []
        for pg in pgs:
            npt = int(int(rad[pg + 4])/3)
            pt = np.reshape([i for i in rad[pg + 5:pg + 5 + npt*3]],
                            (npt, 3)).astype(float)
            z = max(z, max(pt[:, 2]))
            pt2 = self._ro_pts(pt, rdir=1)
            paths.append(pt2[:, 0:2])
        return np.array(paths), z

    def _calc_border(self):
        if len(self.points) > 2:
            hull = ConvexHull(self.points[:, 0:2])
            p = Polygon(hull.points[hull.vertices])
            b = p.boundary.parallel_offset(self.ptres/2, join_style=2)
            pts = np.array(b.xy).T
        else:
            o = self.ptres/2
            offset = np.array([[o, o], [o, -o], [-o, -o], [-o, o]])
            pts = (self.points[:, 0:2] + offset[:, None]).reshape(-1, 2)
        z = np.max(self.points[:, 2])
        hull = ConvexHull(pts)
        pt = hull.points[hull.vertices]
        pt2 = np.full((pt.shape[0], 3), z)
        pt2[:, 0:2] = pt
        pt2 = self._ro_pts(pt2, rdir=1)
        return pt2[None, :, 0:2], z
