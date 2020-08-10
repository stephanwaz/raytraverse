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
from scipy.spatial import cKDTree


class SpaceMapper(object):
    """translate between world coordinates and normalized UV space"""

    def __init__(self, dfile, ptres=1.0, rotation=0.0, tolerance=1.0):
        #: float: ccw rotation (in degrees) for point grid on plane
        self.rotation = rotation
        #: float: tolerance for point search when using point list for area
        self.tolerance = tolerance
        #: float: point resolution for area
        self.ptres = ptres
        self.pt_kd = None
        #: np.array: boundary frame for translating between coordinates
        #: [[xmin ymin zmin] [xmax ymax zmax]]
        self.bbox = dfile

    @property
    def pt_kd(self):
        """point kdtree for spatial queries built at first use"""
        if self._pt_kd is None:
            self._pt_kd = cKDTree(self.pts())
        return self._pt_kd

    @pt_kd.setter
    def pt_kd(self, pt_kd):
        self._pt_kd = pt_kd

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

    @property
    def npts(self):
        """number of points"""
        return int(np.product(self.ptshape))

    @bbox.setter
    def bbox(self, plane):
        """read radiance geometry file as boundary path"""
        paths, bbox = self._rad_scene_to_bbox(plane)
        self._bbox = bbox
        self._sf = self.bbox[1, 0:2] - self.bbox[0, 0:2]
        self._path = []
        for pt in paths:
            p = (np.concatenate((pt, [pt[0]])) - bbox[0, 0:2])/self._sf
            xy = Path(p, closed=True)
            self._path.append(xy)
        size = (bbox[1, 0:2] - bbox[0, 0:2])/self.ptres
        self._ptshape = np.ceil(size).astype(int)

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
        sf = self.bbox[1] - self.bbox[0]
        uv = np.hstack((uv, np.zeros(len(uv)).reshape(-1, 1)))
        pt = self.bbox[0] + uv * sf
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
        uv = (self._ro_pts(xyz, rdir=1) - self.bbox[0])[:, 0:2] / self._sf
        return uv

    def idx2pt(self, idx):
        shape = self.ptshape
        si = np.stack(np.unravel_index(idx, shape)).T
        return self.uv2pt((si + .5)/shape)

    def pts(self):
        shape = self.ptshape
        return self.idx2pt(np.arange(np.product(shape)))

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
        if path is None:
            return np.full((uv.shape[0]), True)
        else:
            result = np.empty((len(path), uv.shape[0]), bool)
            for i, p in enumerate(path):
                result[i] = p.contains_points(uv)
        return np.any(result, 0)

    def _rad_scene_to_bbox(self, plane):
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
        bbox = np.array(paths)
        bbox = np.squeeze([np.amin(bbox, 1), np.amax(bbox, 1)])
        bbox = np.hstack([bbox, [[z], [z]]])
        return paths, bbox
