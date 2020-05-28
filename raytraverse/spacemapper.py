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


class SpaceMapper(object):
    """translate between world coordinates and normalized UV space"""

    def __init__(self, bbox, rotation=0.0):
        #: float: ccw rotation (in degrees) for point grid on plane
        self.rotation = rotation
        self._path = None
        #: np.array: boundary frame for translating between coordinates
        #: [[xmin ymin zmin] [xmax ymax zmax]]
        self.bbox = bbox

    @property
    def bbox(self):
        """np.array of shape (3,2): bounding box"""
        return self._bbox

    @property
    def path(self):
        """list of matplotlib.path.Path: boundary paths"""
        return self._path

    @bbox.setter
    def bbox(self, plane):
        """read radiance geometry file as boundary path"""
        paths, bbox = self._rad_scene_to_bbox(plane)
        self._bbox = bbox
        self._path = []
        for pt in paths:
            xy = Path(np.concatenate((pt, [pt[0]])), closed=True)
            self._path.append(xy)

    def ro_pts(self, points, rdir=-1):
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
        sf = self.bbox[1, 0] - self.bbox[0, 0]
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
        sf = self.bbox[1, 0] - self.bbox[0, 0]
        uv = (self.ro_pts(xyz, 1) - self.bbox[0])[:, 0:2] / sf
        return uv

    def _rad_scene_to_bbox(self, plane):
        with open(plane, 'r') as f:
            dl = [i for i in re.split(r'[\n\r]+', f.read().strip())
                  if not bool(re.match(r'#', i))]
            rad = " ".join(dl).split()
        pgs = [i for i, x in enumerate(rad) if x == "polygon"]
        bbox = []
        z = -1e10
        paths = []
        for pg in pgs:
            npt = int(int(rad[pg + 4])/3)
            pt = np.reshape([i for i in rad[pg + 5:pg + 5 + npt*3]],
                            (npt, 3)).astype(float)
            z = max(z, max(pt[:, 2]))
            pt2 = self.ro_pts(pt, rdir=1)
            p = Path(pt2[:, 0:2], closed=True)
            bbox.append(p.get_extents().get_points())
            paths.append(pt[:, 0:2])
        bbox = np.array(bbox)
        bbox = np.array([np.amin(bbox[:, 0], 0), np.amax(bbox[:, 1], 0)])
        md = max(bbox[1, :] - bbox[0, :])
        x0 = (md - (bbox[1, 0] - bbox[0, 0]))/2
        y0 = (md - (bbox[1, 1] - bbox[0, 1]))/2
        bbox = np.hstack([bbox, [[z], [z]]])
        bbox = (bbox + np.array([(-x0, -y0, 0), (x0, y0, 0)]))
        return paths, bbox
