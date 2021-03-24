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
from sklearn.cluster import Birch


class LightPointKD(object):
    """light distribution from a point with KDtree structure for directional
    query

    Parameters
    ----------
    scene: raytraverse.scene.BaseScene
    vec: np.array, optional
        shape (N, >=3) where last three columns are normalized direction vectors
        of samples. If not given, tries to load from scene.outdir
    lum: np.array, optional
        reshapeable to (N, srcn). sample values for each source corresponding
        to vec. If not given, tries to load from scene.outdir
    vm: raytraverse.mapper.ViewMapper, optional
        a default viewmapper for image and metric calculations, should match
        viewmapper of sampler.run() if possible.
    pt: tuple list np.array
        3 item point location of light distribution
    posidx: int, optional
        index position of point, will govern file naming so must be set to
        avoid clobbering writes. also used by spacemapper for planar sampling
    src: str, optional
        name of source group. will govern file naming so must be set to
        avoid clobbering writes.
    srcn: int, optional
        must match lum, does not need to be set if reloading from scene.outdir
    calcomega: bool, optional
        if True (default) calculate solid angle of rays. This  is
        unnecessary if point will be combined before calculating any metrics.
        setting to False will save some computation time.
    write: bool, optional
        whether to write to disk.
    omega: np.array, optional
        provide precomputed omega values, if given, overrides calcomega
    """

    def __init__(self, scene, vec=None, lum=None, vm=None, pt=(0, 0, 0),
                 posidx=0, src='sky', srcn=1, calcomega=True,
                 write=True, omega=None):
        if vm is None:
            vm = ViewMapper()
        #: raytraverse.mapper.ViewMapper
        self.vm = vm
        #: raytraverse.scene.Scene
        self.scene = scene
        #: int: index for point
        self.posidx = posidx
        #: np.array: point location
        self.pt = np.asarray(pt).flatten()[0:3]
        #: str: source key
        self.src = src
        #: str: relative path to disk storage
        self.file = f"{self.scene.outdir}/{self.src}/{self.posidx:06d}.rytpt"
        if vec is not None and lum is not None:
            scene.log(self, f"building {src} at {posidx}")
            self._d_kd, self._vec, self._lum = self._build(vec, lum, srcn)
            self._omega = omega
            if calcomega and self._omega is None:
                self.calc_omega(False)
            if write:
                self.dump()
            scene.log(self, f"build complete")
        elif os.path.isfile(self.file):
            self.load()
        else:
            raise ValueError(f"Cannot initialize {type(self).__name__} without"
                             f" file: {self.file} or parameters 'vec' "
                             f"and 'lum'")
        self.srcn = self.lum.shape[1]

    def load(self):
        f = open(self.file, 'rb')
        self._d_kd, self._vec, self._omega, self._lum = pickle.load(f)
        f.close()

    def dump(self):
        try_mkdir(f"{self.scene.outdir}/{self.src}")
        f = open(self.file, 'wb')
        pickle.dump((self._d_kd, self._vec, self._omega, self._lum), f,
                    protocol=4)
        f.close()

    @property
    def srcdir(self):
        return np.zeros((self.srcn, 3))

    @property
    def vec(self):
        """direction vector (N,3)"""
        return self._vec

    @property
    def lum(self):
        """luminance (N,srcn)"""
        return self._lum

    @property
    def d_kd(self):
        """kd tree for spatial query

        :getter: Returns kd tree structure
        :type: scipy.spatial.cKDTree
        """
        return self._d_kd

    @property
    def omega(self):
        """solid angle (N)

        :getter: Returns array of solid angles
        :setter: sets soolid angles with viewmapper
        :type: np.array
        """
        return self._omega

    def calc_omega(self, write=True):
        """calculate solid angle

        Parameters
        ----------
        write: bool, optional
            update/write kdtree data to file
        """
        vm = self.vm
        if self.vec.shape[0] < 100:
            omega = None
        elif vm.aspect == 1:
            # in case of 180 view, cannot use spherical voronoi, instead
            # the method estimates area in square coordinates by intersecting
            # 2D voronoi with border square.
            # so none of our points have infinite edges.
            uv = vm.xyz2uv(self.vec)
            bordered = np.concatenate((uv, np.array([[-10, -10], [-10, 10],
                                                     [10, 10], [10, -10]])))
            vor = Voronoi(bordered)
            # the border of our 180 degree region
            lb = .5 - vm.viewangle / 360
            ub = .5 + vm.viewangle / 360
            mask = Polygon(np.array([[lb, lb], [lb, ub],
                                     [ub, ub], [ub, lb], [lb, lb]]))
            omega = []
            for i in range(len(uv)):
                region = vor.regions[vor.point_region[i]]
                p = Polygon(np.concatenate((vor.vertices[region],
                                            [vor.vertices[region][
                                                 0]]))).intersection(mask)
                # scaled from unit square -> hemisphere
                omega.append(p.area*2*np.pi)
            omega = np.asarray(omega)
        else:
            try:
                omega = SphericalVoronoi(self.vec).calculate_areas()
            except ValueError:
                # spherical voronoi raises a ValueError when points are
                # too close, in this case we cull duplicates before calculating
                # area, leaving the duplicates with omega=0 it would be more
                # efficient downstream to filter these points, but that would
                # require culling the vecs and lum and rebuilding to kdtree
                flt = np.full(len(self.vec), True)
                omega = np.zeros(self.vec.shape[0])
                tol = 2*np.pi/2**10
                pairs = self.d_kd.query_ball_tree(self.d_kd, tol)
                flagged = set()
                for j, p in enumerate(pairs):
                    if j not in flagged and len(p) > 1:
                        flt[p[1:]] = False
                        flagged.update(p[1:])
                omega[flt] = SphericalVoronoi(self.vec[flt]).calculate_areas()
        self._omega = omega
        if write:
            self.dump()

    def apply_coef(self, coefs):
        """apply coefficient vector to self.lum

        Parameters
        ----------
        coefs: np.array int float list
            shape (N, self.srcn) or broadcastable

        Returns
        -------
        alum: np.array
            shape (N, self.vec.shape[0])
        """
        try:
            c = np.asarray(coefs).reshape(-1, self.srcn)
        except ValueError:
            c = np.broadcast_to(coefs, (1, self.srcn))
        return np.einsum('ij,kj->ik', c, self.lum)

    def add_to_img(self, img, vecs, mask=None, skyvec=1, interp=False,
                   omega=False, vm=None, rnd=False):
        """add luminance contributions to image array (updates in place)

        Parameters
        ----------
        img: np.array
            2D image array to add to (either zeros or with other source)
        vecs: np.array
            vectors corresponding to img pixels shape (N, 3)
        mask: np.array, optional
            indices to img that correspond to vec (in case where whole image
            is not being updated, such as corners of fisheye)
        skyvec: int float np.array, optional
            source coefficients, shape is (1,) or (srcn,)
        interp: bool, optional
            for linear interpolation (falls back to nearest outside of
            convexhull
        omega: bool
            if true, add value of ray solid angle instead of luminance
        vm: raytraverse.mapper.ViewMapper, optional
        """
        if rnd:
            val = np.random.rand(1, self.omega.size)
        elif omega:
            val = self.omega.reshape(1, -1)
        else:
            val = self.apply_coef(skyvec)
        if interp:
            if vm is None:
                vm = self.vm
            xyp = vm.xyz2xy(vecs)
            xys = vm.xyz2xy(self.vec)
            interp = LinearNDInterpolator(xys, val[0], fill_value=-1)
            lum = interp(xyp[:, 0], xyp[:, 1])
            neg = lum < 0
            i, d = self.query_ray(vecs[neg])
            lum[neg] = val[0, i]
        else:
            i, d = self.query_ray(vecs)
            lum = val[:, i]
        img[mask] += np.squeeze(lum)

    def get_applied_rays(self, skyvec, vm=None):
        """return rays within view with skyvec applied. this is the
        analog to add_to_img for metric calculations

        Parameters
        ----------
        skyvec: int float np.array, optional
            source coefficients, shape is (1,) or (srcn,)
        vm: raytraverse.mapper.ViewMapper, optional

        Returns
        -------
        rays: np.array
            shape (N, 3) rays falling within view
        omega: np.array
            shape (N,) associated solid angles
        lum: np.array
            shape (N,) associated luminances
        """
        if vm is None:
            vm = self.vm
        idx = self.query_ball(vm.dxyz, vm.viewangle * vm.aspect)[0]
        omega = np.squeeze(self.omega[idx])
        rays = self.vec[idx]
        lum = np.squeeze(self.apply_coef(skyvec))[idx]
        return rays, omega, lum

    def query_ray(self, vecs):
        """return the index and distance of the nearest ray to each of vecs

        Parameters
        ----------
        vecs: np.array
            shape (N, 3) normalized vectors to query, could represent image
            pixels for example.

        Returns
        -------
        i: np.array
            integer indices of closest ray to each query
        d: np.array
            distance (corresponds to chord length on unit sphere) from query to
            ray in lightpoint. use translate.chord2theta to convert to angle.
        """
        d, i = self.d_kd.query(vecs)
        return i, d

    def query_ball(self, vecs, viewangle=180):
        """return set of rays within a view cone

        Parameters
        ----------
        vecs: np.array
            shape (N, 3) vectors to query.
        viewangle: int float
            opening angle of view cone

        Returns
        -------
        i: list np.array
            if vecs is a single point, a list of vector indices of rays
            within view cone. if vecs is a set of point an array of lists, one
            for each vec is returned.
        """
        vs = translate.theta2chord(viewangle/360*np.pi)
        return self.d_kd.query_ball_point(translate.norm(vecs), vs)

    def direct_view(self, res=512, showsample=False, showweight=True, rnd=False,
                    srcidx=None, interp=False, omega=False, scalefactor=1, vm=None):
        """create an unweighted summary image of lightpoint"""
        if vm is None:
            vm = self.vm
        img, pdirs, mask, mask2, header = vm.init_img(res, self.pt)
        if showweight:
            if srcidx is not None:
                skyvec = np.zeros(self.srcn)
                skyvec[srcidx] = scalefactor
            else:
                skyvec = np.full(self.srcn, scalefactor / self.srcn)
            self.add_to_img(img, pdirs[mask], mask,
                            interp=interp, skyvec=skyvec, omega=omega, rnd=rnd)
            if vm.aspect == 2:
                self.add_to_img(img[res:], pdirs[res:][mask2], mask2, vm=vm.ivm,
                                interp=interp, skyvec=skyvec, omega=omega, rnd=rnd)
            channels = (1, 0, 0)
        else:
            channels = (1, 1, 1)
        if omega:
            outf = self.file.replace("/", "_").replace(".rytpt", "_omega.hdr")
        else:
            outf = self.file.replace("/", "_").replace(".rytpt", ".hdr")
        if srcidx is not None:
            outf = outf.replace(f"_{self.src}_", f"_{self.src}_{srcidx:04d}_")
        if showsample:
            img = np.repeat(img[None, ...], 3, 0)
            vi = self.query_ball(vm.dxyz, vm.viewangle * vm.aspect)
            v = self.vec[vi[0]]
            img = io.add_vecs_to_img(vm, img, v, channels=channels)
            io.carray2hdr(img, outf, [header])
        else:
            io.array2hdr(img, outf, [header])
        return outf

    @staticmethod
    def _build(vec, lum, srcn):
        """load samples and build data structure"""
        vec = vec[:, -3:]
        d_kd = cKDTree(vec)
        lum = lum.reshape(-1, srcn)
        return d_kd, vec, lum

    def source_weighted_vector(self, weight=10):
        """generate vector for clustering

        Parameters
        ----------
        weight: float, optional
            coefficient for min-max normalized luminances, bigger values weight
            luminance more strongly compared to vector direction, meaning with
            higher numbers clusters will have less variance in luminance.

        Returns
        -------
        weighted_vector: np.array
            (N, 4) ray direction, and source brightness
        """
        lum = self.apply_coef(1).T
        # min-max normalize luminances
        bound = np.percentile(lum, (0, 100))
        scale = bound[1] - bound[0]
        ldist = (lum - bound[0])/scale
        return np.hstack((self.vec, ldist * weight))

    def reduce(self, slices):
        """group vector/omega/lum data according to slices generated by cluster

        Parameters
        ----------
        slices: list
            length = len(np.unique(labels))
            list of index arrays for each cluster label

        Returns
        -------
        vec: np.array
            shape (n_clust, 3), reduced vectors
        oga: np.array
            shape (n_clust,), reduced omeega
        lum: np.array
            shape (n_clust, nsrcs), reduced lum
        """
        v2 = []
        o2 = []
        l2 = []
        lum = np.reshape(self.lum, (self.vec.shape[0], -1))
        for s in slices:
            v = np.average(self.vec[s], 0, weights=self.omega[s])
            mag = np.linalg.norm(v)
            v2.append(v/mag)
            o2.append(np.sum(self.omega[s]))
            l2.append(np.average(lum[s], 0, weights=self.omega[s]))
        return np.array(v2), np.array(o2), np.array(l2)

    def compress(self, src=None, write=False, dist=0.19603428, lweight=10,
                 **kwargs):
        """A lossy compression based on clustering. Rays are clustered using
        the birch algoritm on a 4D vector  (x,y,z,lum) where lum is the sum
        of contributions from all sources in the LightPoint. In the optional
        second stage (activated with secondary=True) sources are further
        grouped through agglomerative cluster using an average linkage. this
        is to help with source indentification/matching between LightPoints,
        but can introduce significant errors to computing non energy
        conserving metrics in cases where the applied sky vectors have large
        relative differences between adjacent patches (> 1.5:1) or if the
        variance in peak luminance above the lthreshold parameter is
        significant. These include cases where nearby transmitting materials
        is varied (example: a trans upper above a clear lower), or lthreshold
        is set too low. For this reason, it is better to use single stage
        compression for metric computation and only do glare source grouping
        for interpolation between LightPoints.

        Parameters
        ----------
        src: str, optional
            new name for src passed to LightPointKD constructor
        write: bool, optional
            save lightpoint to disk
        dist: float, optional
            translate.theta2chord(np.pi/32), primary clustering distance
            using the birch algorithm, for lossy compression of lf. this is the
            maximum radius of a cluster, preserving important directional
            information. clustering acts on ray direction and luminance, with
            weight of luminance dimension controlled by the lweight parameter.
        lweight: float, optional
            coefficient for min-max normalized luminances used in primary
            compression clustering, bigger values weight luminance more strongly
            compared to vector direction, meaning with higher numbers clusters
            will have less variance in luminance.
        kwargs:
            passed to constructor of returned class

        Returns
        -------
        raytraverse.lightpoint.LightPointKD

        """
        wv = self.source_weighted_vector(lweight)
        clust = Birch(threshold=dist, n_clusters=None)
        clust.fit(wv)
        lsort = np.argsort(clust.labels_)
        ul, sidx = np.unique(clust.labels_[lsort], return_index=True)
        slices = np.array_split(lsort, sidx[1:])
        ovec, ooga, olum = self.reduce(slices)
        if src is None:
            src = f"{self.src}_compressed"
        return type(self)(self.scene, ovec, olum, vm=self.vm, pt=self.pt,
                          posidx=self.posidx, src=src, srcn=self.srcn,
                          omega=ooga, write=write, **kwargs)
