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
from shapely.ops import unary_union

from raytraverse import translate, io
from raytraverse.mapper.mapper import Mapper


class PlanMapper(Mapper):
    """translate between world positions on a horizontal plane and
    normalized UV space for a given view angle. pixel projection yields a
    parallel plan projection

    Parameters
    ----------
    area: str np.array, optional
        radiance scene geometry defining a plane to sample, tsv file of
        points to generate bounding box, or np.array of points.
    ptres: float, optional
        resolution for considering points duplicates, border generation
        (1/2) and add_grid(). updateable
    rotation: float, optional
        positive Z rotation for point grid alignment
    zheight: float, optional
        override calculated zheight
    name: str, optional
        plan mapper name used for output file naming
    jitterrate: float, optional
        proportion of cell to jitter within
    autorotate: bool, optional
        if true set rotation based on long axis of area geometry
    autogrid: int, optional
        if given, autoset ptres based on this minimum number of points at
        level 0 along the minimum dimemsion (width or height)
    """

    def __init__(self, area, ptres=1.0, rotation=0.0, zheight=None,
                 name="plan", jitterrate=0.5, autorotate=False, autogrid=None):
        if autorotate:
            rotation = self.__class__(area, autorotate=False,
                                      autogrid=None)._auto_rotate()
        if autogrid is not None:
            pm = self.__class__(area, autorotate=False, autogrid=None)
            hv = pm.bbox[1] - pm.bbox[0]
            if hv[0] > hv[1]:
                ptres = hv[1]/(autogrid - 1) - 1e-3
            else:
                ptres = hv[0]/(autogrid - 1) - 1e-3
        #: float: ccw rotation (in degrees) for point grid on plane
        self.rotation = rotation
        #: float: point resolution for area look ups and grid
        self.ptres = ptres
        self._bbox = None
        self._zheight = zheight
        self._path = []
        self._candidates = None
        self._boundary = None
        xyz = self.update_bbox(area, updatez=zheight is None)
        super().__init__(dxyz=xyz, name=name, sf=self._sf, bbox=self.bbox,
                         jitterrate=jitterrate)

    @property
    def dxyz(self):
        """(float, float, float) central view point"""
        return self._dxyz

    @dxyz.setter
    def dxyz(self, xyz):
        """set view parameters"""
        self._dxyz = np.asarray(xyz).ravel()[0:3]

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, r):
        cos = np.cos(r*np.pi/180)
        sin = np.sin(r*np.pi/180)
        rmtx = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        self._rmtx = (rmtx, np.eye(3))
        self._rotation = r

    @property
    def bbox(self):
        """np.array: boundary frame for translating between coordinates
        [[xmin ymin zmin] [xmax ymax zmax]]"""
        return self._bbox

    def update_bbox(self, plane, level=0, updatez=True):
        """handle bounding box generation from plane or points"""
        try:
            points = np.atleast_2d(io.load_txt(plane))[:, 0:3]
            paths, z = self._calc_border(points, level)
        except TypeError:
            points = np.atleast_2d(plane)[:, 0:3]
            paths, z = self._calc_border(points, level)
        except ValueError:
            paths, z = self._rad_scene_to_paths(plane)
        bbox = np.full((2, 2), np.inf)
        bbox[1] *= -1
        for p in paths:
            bbox[0] = np.minimum(np.amin(p, 0), bbox[0])
            bbox[1] = np.maximum(np.amax(p, 0), bbox[1])
        self._bbox = bbox
        if updatez:
            self._zheight = z
        xyz = np.concatenate((np.average(self.bbox, 0), [self._zheight]))
        self._sf = self.bbox[1] - self.bbox[0]
        self._path = []
        for pt in paths:
            p = (np.concatenate((pt, [pt[0]])) - bbox[0, 0:2])/self._sf
            xy = Path(p, closed=True)
            self._path.append(xy)
        return xyz

    def uv2xyz(self, uv, stackorigin=False):
        """transform from mapper UV space to world xyz"""
        uvshape = uv.shape
        uv = self.bbox[None, 0] + np.reshape(uv, (-1, 2))*self._sf[None, :]
        pt = np.hstack((uv, np.full((len(uv), 1), self._zheight)))
        return self.view2world(pt).reshape(*uvshape[:-1], 3)

    def in_view_uv(self, uv, indices=True, **kwargs):
        path = self._path
        uvs = uv.reshape(-1, 2)
        result = np.empty((len(path), uvs.shape[0]), bool)
        for i, p in enumerate(path):
            result[i] = p.contains_points(uvs)
        mask = np.any(result, 0)
        if indices:
            return np.unravel_index(np.arange(mask.size)[mask],
                                    uv.shape[:-1])
        else:
            return mask

    def in_view(self, vec, indices=True):
        """check if point is in boundary path

        Parameters
        ----------
        vec: np.array
            xyz coordinates, shape (N, 3)
        indices: bool, optional
            return indices of True items rather than boolean array

        Returns
        -------
        mask: np.array
            boolean array, shape (N,)
        """
        return self.in_view_uv(self.xyz2uv(vec), indices)

    def header(self, **kwargs):
        vp = self.dxyz
        vu = self.view2world([0, 1, 0])[0]
        vs = self.bbox[1] - self.bbox[0]
        return ("VIEW= -vtl -vp {} {} {} -vu {} {} {} ".format(*vp, *vu) +
                "-vd 0 0 -1 -vh {} -vv {}".format(vs[0], vs[1]))

    def borders(self):
        """world coordinate vertices of planmapper boundaries"""
        return [self.uv2xyz(p.vertices) for p in self._path]

    @property
    def boundary(self):
        if self._boundary is None:
            self._boundary = unary_union([Polygon(b) for b in self.borders()])
        return self._boundary

    def bbox_vertices(self, offset=0, close=False):
        b = self.bbox
        v = [(b[0, 0] - offset, b[0, 1] - offset, self._zheight),
             (b[1, 0] + offset, b[0, 1] - offset, self._zheight),
             (b[1, 0] + offset, b[1, 1] + offset, self._zheight),
             (b[0, 0] - offset, b[1, 1] + offset, self._zheight)]
        if close:
            v.append(v[0])
        return self.view2world(np.asarray(v))

    def shape(self, level=0):
        return np.ceil(self._sf/self.ptres - 1e-4).astype(int)*2**level

    def point_grid(self, jitter=True, level=0, masked=True, snap=None):
        """generate a grid of points

        Parameters
        ----------
        jitter: bool, optional
            if None, use the instance default, if True jitters point samples
            within stratified grid
        level: int, optional
            sets the resolution of the grid as a power of 2 from ptres
        masked: bool, optional
            apply in_view to points before returning
        snap: int, optional
            level to snap samples to when jitter=False should be > level

        Returns
        -------
        np.array
            shape (N, 3)
        """
        return self.uv2xyz(self.point_grid_uv(jitter, level, masked, snap))

    def point_grid_uv(self, jitter=True, level=0, masked=True, snap=None):
        """add a grid of UV coordinates

        Parameters
        ----------
        jitter: bool, optional
            if None, use the instance default, if True jitters point samples
            within stratified grid
        level: int, optional
            sets the resolution of the grid as a power of 2 from ptres
        masked: bool, optional
            apply in_view to points before returning
        snap: int, optional
            level to snap samples to when jitter=False should be > level

        Returns
        -------
        np.array
            shape (N, 2)
        """
        # bypasses grid when initialized with points
        shape = self.shape(level)
        idx = np.arange(np.product(shape))
        uv = self.idx2uv(idx, shape, jitter)
        if level == 0 and not jitter and self._candidates is not None:
            uvup = self.xyz2uv(self._candidates)
            uv[:] = self.bbox[1] * 2
            uv[self.uv2idx(uvup, shape)] = uvup
        elif not jitter and snap is not None and snap > level:
            s2 = self.shape(snap)
            uv -= .5 / s2
            uv += np.random.default_rng().integers(0, 2, uv.shape)/s2
        if masked:
            return uv[self.in_view_uv(uv, False)]
        else:
            return uv

    def _auto_rotate(self):
        v = np.vstack(self.borders())
        left = np.average(v[np.isclose(self.bbox[0, 0], v[:, 0])], 0)
        right = np.average(v[np.isclose(self.bbox[1, 0], v[:, 0])], 0)
        top = np.average(v[np.isclose(self.bbox[1, 1], v[:, 1])], 0)
        bottom = np.average(v[np.isclose(self.bbox[0, 1], v[:, 1])], 0)
        vaxis = np.average((top - right, left - bottom), 0)
        haxis = np.average((right - bottom, top - left), 0)
        hv = np.linalg.norm((haxis, vaxis), axis=1)
        if hv[0] > hv[1]:
            rotation = np.arccos(np.dot(haxis/hv[0], (1, 0, 0)))*180/np.pi
        else:
            rotation = np.arccos(np.dot(vaxis/hv[1], (0, 1, 0)))*180/np.pi
        return rotation

    def _rad_scene_to_paths(self, plane):
        """reads a radiance scene for polygons, and sets bounding paths
        zheight of plan (unless PlanMapper initialized with a zheight) will be
        set to the maximum height of all polygons in scene. Ignores all objects
        in scene that are not polygons.

        Parameters
        ----------
        plane: str
            path to radiance scene file with polygons defined as::

                void polygon a
                0
                0
                12 0 0 0 0 1 0
                   1 1 0 1 0 0

                void polygon b 0 0 9 0 0 0 0 1 0 1 0 0

        Returns
        -------

        """
        with open(plane, 'r') as f:
            dl = [i for i in re.split(r'[\n\r]+', f.read().strip())
                  if not bool(re.match(r'#', i))]
            rad = " ".join(dl).split()
        pgs = [i for i, x in enumerate(rad) if x == "polygon"]
        paths = []
        zs = []
        for pg in pgs:
            npt = int(int(rad[pg + 4])/3)
            pt = np.reshape([i for i in rad[pg + 5:pg + 5 + npt*3]],
                            (npt, 3)).astype(float)
            zs.append(pt[:, 2])
            pt2 = self.world2view(pt)
            paths.append(pt2[:, 0:2])
        z = np.median(zs)
        return np.array(paths), z

    def _calc_border(self, points, level=0):
        """generate a border from convex hull of points"""
        self._candidates = points
        o = self.ptres/2**(level + 1)
        try:
            hull = ConvexHull(points[:, 0:2])
        except RuntimeError:
            offset = np.array([[o, o], [o, -o], [-o, -o], [-o, o]])
            pts = (points[:, 0:2] + offset[:, None]).reshape(-1, 2)
        else:
            p = Polygon(hull.points[hull.vertices])
            b = p.boundary.parallel_offset(o, join_style=2)
            pts = np.array(b.xy).T
        z = np.max(points[:, 2])
        hull = ConvexHull(pts)
        pt = hull.points[hull.vertices]
        pt2 = np.full((pt.shape[0], 3), z)
        pt2[:, 0:2] = pt
        pt2 = self.world2view(pt2)
        return pt2[None, :, 0:2], z
