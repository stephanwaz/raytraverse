# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import functools

import numpy as np
from raytraverse.craytraverse import interpolate_kdquery
from scipy.spatial import cKDTree, Voronoi
from scipy.interpolate import LinearNDInterpolator, SmoothBivariateSpline
from shapely.geometry import Polygon

from raytraverse import io
from raytraverse.evaluate import MetricSet
from raytraverse.lightpoint import LightPointKD


class LightPointSet(object):

    def __init__(self, scene, points, idx, src, parent):
        self.scene = scene
        self.points = points
        self.idx = idx
        self.src = src
        self.parent = parent

    @functools.lru_cache(5)
    def __getitem__(self, item):
        return LightPointKD(self.scene, pt=self.points[item],
                            posidx=self.idx[item], src=self.src,
                            parent=self.parent)

    def __len__(self):
        return len(self.points)


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
        self.lp = idx
        # self.lp = [f"{self._datadir}/{i:06d}.rytpt" for i in idx]

    @property
    def lp(self):
        """luminance (N,srcn)"""
        return self._lp

    @ lp.setter
    def lp(self, idx):
        self._lp = LightPointSet(self.scene, self.points, idx, self.src,
                                 self.pm.name)

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

    def get_applied_metrics(self, skyvec, points=None, vm=None,
                            metricclass=MetricSet, metrics=None, mask=True,
                            **kwargs):
        # qidx are the unique query indices and midx are the mapping indices
        # to restore full results from the qidx results
        if points is None:
            qidx = midx = np.arange(len(self.lp))
        else:
            ridx, d = self.query_pt(points, clip=mask)
            if mask:
                ridx = ridx[d < 1e6]
            qidx, midx = np.unique(ridx, return_inverse=True)
        results = []
        for qi in qidx:
            lp = self.lp[qi]
            vol = lp.get_applied_rays(skyvec, vm=vm)
            if vm is None:
                vm = lp.vm
            results.append(metricclass(*vol, lp.vm, metricset=metrics,
                                       **kwargs)())
        return np.array(results)[midx]

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

    def interpolate_query(self, vals, dest_vec, k=8,
                          err=0.1, up=2):
        """query a kd_tree and interpolate corresponding values. used to
        merge to kd_trees with vector and luminance
        Parameters
        ----------
        pi:
            key
        dest_vec: np.array
            destination vectors to interpolate to, shape (N, 3)
        k: int
            initial query size
        err: float
            chord length under which value is taken without interpolation
            default is .25 degrees = translate.theta2chord(.25*pi/180)
        up: float
            chord length of maximum search radius for neighbors
            default is 10 degrees  = translate.theta2chord(20*pi/180)
        Returns
        -------
        np.array
            shape of (dest_vec.shape[0], src_lum.shape[1])
        """
        errs, idxs = self.pt_kd.query(dest_vec, k=k, distance_upper_bound=up)
        arrout = interpolate_kdquery(dest_vec, errs, idxs, self.points, vals, err=err)
        return arrout

    def make_image(self, outf, vals, res=1024, interp=False, showsample=False):
        img, vecs, mask, _, header = self.pm.init_img(res)
        if interp:
            xyp = vecs[mask]
            # interp = SmoothBivariateSpline(self.points[:, 0], self.points[:, 1], vals,
            #                                bbox=[min(xyp[:, 0]), max(xyp[:, 0]),
            #                                      min(xyp[:, 1]), max(xyp[:, 1])],
            #                                kx=2, ky=2)
            # lum = interp(xyp[:, 0], xyp[:, 1], grid=False)
            # lum = self.interpolate_query(vals, xyp).ravel()
            interp = LinearNDInterpolator(self.points[:, 0:2], vals, fill_value=-1)
            lum = interp(xyp[:, 0], xyp[:, 1])
            neg = lum < 0
            # i, d = self.query_pt(xyp[neg], False)
            d, i = self.pt_kd.query(xyp[neg], 2)
            # print(d2.shape, np.asarray((vals[i2[:, 0]], vals[i2[:, 1]])).shape)
            lum[neg] = np.average((vals[i[:, 0]], vals[i[:, 1]]), axis=0, weights=1/d.T)
            lum[neg] = vals[i]
            img[mask] = lum
        else:
            i, d = self.query_pt(vecs[mask], False)
            img[mask] = vals[i]

        if showsample:
            img = np.repeat(img[None, ...], 3, 0)
            img = self.pm.add_vecs_to_img(img, self.points,
                                          channels=(1, 0, 0))
            io.carray2hdr(img, outf, header)
        else:
            io.array2hdr(img, outf, header)

    def direct_view(self, res=512, showsample=True, vm=None, area=False,
                    metricclass=MetricSet, metrics=('avglum',), interp=False):
        """create a summary image of lightplane showing samples and areas"""
        if area:
            outf = self._datadir.replace("/", "_") + "_area.hdr"
            self.make_image(outf, self.omega, res=res, showsample=showsample,
                            interp=interp)
        if metrics is not None:
            result = self.get_applied_metrics(1, vm=vm,
                                              metricclass=metricclass,
                                              metrics=metrics, scale=1).T
            for r, m in zip(result, metrics):
                outf = self._datadir.replace("/", "_") + f"_{m}.hdr"
                self.make_image(outf, r, res=res, showsample=showsample,
                                interp=interp)

    def add(self, lf2, src=None, calcomega=True, write=False):
        """add light planes of distinct sources together
        """
        pass
        # return type(self)()
