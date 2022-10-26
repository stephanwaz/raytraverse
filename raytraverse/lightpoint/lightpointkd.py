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
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import correlate1d

from scipy.spatial import cKDTree, SphericalVoronoi
from clasp.script_tools import try_mkdir

from raytraverse import io, translate
from raytraverse.mapper import ViewMapper, PlanMapper
from raytraverse.lightpoint.srcviewpoint import SrcViewPoint


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
        whether to save ray data to disk.
    omega: np.array, optional
        provide precomputed omega values, if given, overrides calcomega
    """

    def __init__(self, scene, vec=None, lum=None, vm=None, pt=(0, 0, 0),
                 posidx=0, src='sky', srcn=1, srcdir=(0, 0, 1), calcomega=True,
                 write=True, omega=None, filterviews=True, srcviews=None,
                 parent=None, srcviewidxs=None, features=1):
        self.srcviews = []
        self.srcviewidxs = []
        self.set_srcviews(srcviews, srcviewidxs)
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
        #: direction to source(s)
        self.srcdir = translate.norm(np.asarray(srcdir).reshape(-1, 3))
        if parent is not None:
            outdir = f"{self.scene.outdir}/{parent}"
        else:
            outdir = self.scene.outdir
        self.parent = parent
        #: str: relative path to disk storage
        self.file = f"{outdir}/{self.src}/{self.posidx:06d}.rytpt"
        self._vec = np.empty((0, 3))
        if features > 1:
            self._lum = np.empty((0, srcn, features))
        else:
            self._lum = np.empty((0, srcn))
        self._d_kd = None
        self._omega = None
        if vec is not None and lum is not None:
            self.srcn = srcn
            self.features = features
            self.update(vec, lum, omega=omega, calcomega=calcomega,
                        write=write, filterviews=filterviews)
        elif os.path.isfile(self.file):
            self.load()
        else:
            raise ValueError(f"Cannot initialize {type(self).__name__} without"
                             f" file: {self.file} or parameters 'vec' "
                             f"and 'lum'")
        self.srcdir = np.broadcast_to(self.srcdir, (self.srcn, 3))

    def load(self):
        f = open(self.file, 'rb')
        loads = pickle.load(f)
        self._d_kd, self._vec, self._omega, self._lum = loads[0:4]
        self.srcviews = loads[4]
        self.srcdir = loads[5]
        try:
            self.srcviewidxs = loads[6]
        except IndexError:
            self.srcviewidxs = [-1] * max(len(self.srcviews), 1)
        self.srcn = self.lum.shape[1]
        try:
            self.features = self.lum.shape[2]
        except IndexError:
            self.features = 1
        f.close()

    def dump(self):
        if self.parent is not None:
            try_mkdir(f"{self.scene.outdir}/{self.parent}")
            try_mkdir(f"{self.scene.outdir}/{self.parent}/{self.src}")
        else:
            try_mkdir(f"{self.scene.outdir}/{self.src}")
        f = open(self.file, 'wb')
        pickle.dump((self._d_kd, self._vec, self._omega, self._lum,
                     self.srcviews, self.srcdir, self.srcviewidxs), f,
                    protocol=4)
        f.close()

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

    def set_srcviews(self, srcviews, idxs=None):
        if srcviews is None:
            srcviews = []
        self.srcviews = [i for i in srcviews
                         if issubclass(type(i), SrcViewPoint) or
                            issubclass(type(i), LightPointKD)]
        if idxs is None or len(idxs) == 0:
            self.srcviewidxs = [-1] * max(len(self.srcviews), 1)
        else:
            self.srcviewidxs = idxs

    def calc_omega(self, write=True):
        """calculate solid angle

        Parameters
        ----------
        write: bool, optional
            update/write kdtree data to file
        """
        vm = self.vm
        if self.vec.shape[0] < 100:
            omega = np.full(len(self.vec), 2*np.pi*vm.aspect/len(self.vec))
        elif vm.aspect == 1:
            # in case of 180 view, cannot use spherical voronoi, instead
            # the method estimates area in square coordinates
            b = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
            pm = PlanMapper(b)
            uv = np.hstack((vm.xyz2uv(self.vec), np.zeros((len(self.vec), 1))))
            # scale unit square back to view area
            omega = translate.calc_omega(uv, pm)*vm.area
        else:
            try:
                omega = SphericalVoronoi(self.vec,
                                         threshold=1e-10).calculate_areas()
            except ValueError:
                # spherical voronoi raises a ValueError when points are
                # too close, in this case we cull duplicates before calculating
                # area, leaving the duplicates with omega=0 it would be more
                # efficient downstream to filter these points, but that would
                # require culling the vecs and lum and rebuilding to kdtree
                omega = np.zeros(self.vec.shape[0])
                tol = 2*np.pi/2**10
                flt = translate.cull_vectors(self.d_kd, tol)
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
        if self.features > 1:
            try:
                c = np.asarray(coefs).reshape(-1, self.srcn, self.features)
            except ValueError:
                c = np.broadcast_to(coefs, (1, self.srcn, self.features))
            return np.einsum('ijf,kjf->ikf', c, self.lum)
        try:
            c = np.asarray(coefs).reshape(-1, self.srcn)
        except ValueError:
            c = np.broadcast_to(coefs, (1, self.srcn))
        return np.einsum('ij,kj->ik', c, self.lum)

    def add_to_img(self, img, vecs, mask=None, skyvec=1, interp=False,
                   idx=None, interpweights=None, order=False, omega=False,
                   vm=None, rnd=False, engine=None, **kwargs):
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
        interp: Union[bool, str], optional

            - if "precomp", use index and interpweights
            - if True and engine is None, linearinterpolation
            - if "fastc" and engine: uses content_interp (best after sampling w/o
              detail)
            - if "highc" and engine: uses content_interp_wedge (best after
              sampling w/o detail)
            - if "fast": use interp_fast (pair with sampling w/ detail)
            - if "high": use interp_wedge (pair with sampling w/ detail)

        idx: np.array, optional
            precomputed query/interpolation result
        interpweights: np.array, optional
            precomputted interpolation weights
        omega: bool
            if true, add value of ray solid angle instead of luminance
        vm: raytraverse.mapper.ViewMapper, optional
        rnd: bool, optional
            use random values as contribution (for visualizing data shape)
        engine: raytraverse.renderer.Rtrace, optional
            engine for content aware interpolation
        kwargs: dict, optional
            passed to interpolationn functions
        """
        if order:
            val = np.arange(self.omega.size)[None]
        elif rnd:
            val = np.random.rand(1, self.omega.size)
        elif omega:
            val = self.omega.reshape(1, -1)
        else:
            val = self.apply_coef(skyvec)
        if vm is None:
            vm = self.vm
        if interp == "precomp":
            lum = self.apply_interp(idx, val[0], interpweights)
        elif interp == "fastc" and engine is not None:
            i, weights = self.interp(vecs, angle=False, lum=False, dither=True,
                                     rt=engine, **kwargs)
            lum = self.apply_interp(i, val[0], weights)
        elif interp == "highc" and engine is not None:
            i, weights = self.interp(vecs, rt=engine, **kwargs)
            lum = self.apply_interp(i, val[0], weights)
        elif interp == "fast":
            i, weights = self.interp(vecs, angle=False, lum=False, dither=True,
                                     **kwargs)
            lum = self.apply_interp(i, val[0], weights)
        elif interp == "high":
            i, weights = self.interp(vecs, **kwargs)
            lum = self.apply_interp(i, val[0], weights)
        elif interp is True:
            lum = self.linear_interp(vm, val[0], vecs)
        elif interp:
            raise ValueError(f"Bad value for interp={interp}, should be boolean"
                             f", 'precomp', 'fast', or 'high'")
        else:
            if idx is not None:
                i = idx
            else:
                i, d = self.query_ray(vecs)
            lum = val[:, i]
        if mask is None:
            a = np.transpose(lum, (0, 1, 4, 2, 3))[0, 0]
            img += a
        else:
            img[mask] += np.ravel(lum)
        if not (omega or rnd or order):
            for srcview, srcidx in zip(self.srcviews, self.srcviewidxs):
                srcview.add_to_img(img, vecs, mask, skyvec[srcidx], vm)

    def evaluate(self, skyvec, vm=None, idx=None, srconly=False,
                 blursun=False, includeviews=True):
        """return rays within view with skyvec applied. this is the
        analog to add_to_img for metric calculations

        Parameters
        ----------
        skyvec: int float np.array, optional
            source coefficients, shape is (1,) or (srcn,)
        vm: raytraverse.mapper.ViewMapper, optional
        idx: np.array, optional
            precomputed query_ball result
        srconly: bool, optional
            only evaluate direct sources (stored in self.srcviews)
        includeviews: bool, optional
            include src views in returned results

        Returns
        -------
        rays: np.array
            shape (N, 3) rays falling within view
        omega: np.array
            shape (N,) associated solid angles
        lum: np.array
            shape (N,) associated luminances
        """
        if srconly:
            rays = np.array([[0, 0, 1]])
            omega = np.atleast_1d(np.pi*2)
            lum = np.zeros((1, 1))
        elif vm is None:
            rays = self.vec
            omega = np.atleast_1d(np.squeeze(self.omega))
            lum = self.apply_coef(skyvec).T
            vm = self.vm
        else:
            if idx is None:
                idx = self.query_ball(vm.dxyz, vm.viewangle * vm.aspect)[0]
            omega = np.atleast_1d(np.squeeze(self.omega[idx]))
            rays = self.vec[idx]
            lum = self.apply_coef(skyvec)[:, idx].T
        lum = np.atleast_1d(np.squeeze(lum))
        if len(lum.shape) > 1:
            lum = io.rgb2rad(lum.T)
        if len(self.srcviews) > 0 and includeviews:
            vrs = []
            vos = []
            vls = []
            for srcview, srcidx in zip(self.srcviews, self.srcviewidxs):
                srcvec = np.atleast_2d(skyvec)[:, srcidx]
                vr, vo, vl = srcview.evaluate(srcvec, vm, blursun=blursun)
                # 4 times sun
                if vo[0] > 0.00027:
                    cost = np.argsort(-translate.ctheta(vr[0], rays))
                    comega = np.cumsum(omega[cost])
                    cutoff = np.searchsorted(comega, vo[0], 'right')
                    cutoff = max(cutoff-1, 0)
                    cost = cost[cutoff:]
                    rays = rays[cost]
                    lum = lum[cost]
                    omega = omega[cost]
                vrs.append(vr)
                vos.append(vo)
                vls.append(vl)
            rays = np.concatenate([rays] + vrs, axis=0)
            try:
                omega = np.concatenate([omega] + vos, 0)
            except ValueError:
                omega = np.concatenate(vos, axis=0)
            lum = np.concatenate([lum] + vls, axis=0)
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

    def make_image(self, outf, skyvec, vm=None, res=1024, interp=False,
                   showsample=False):
        if vm is None:
            vm = self.vm
        img, pdirs, mask, mask2, header = vm.init_img(res, self.pt, self.features)
        header = [header]
        self.add_to_img(img, pdirs[mask], mask2, vm=vm,
                        interp=interp, skyvec=skyvec)
        if showsample:
            if len(img.shape) < 3:
                img = np.repeat(img[None, ...], 3, 0)
            vi = self.query_ball(vm.dxyz, vm.viewangle * vm.aspect)
            v = self.vec[vi[0]]
            img = vm.add_vecs_to_img(img, v)
            io.carray2hdr(img, outf, header)
        elif len(img.shape) == 3:
            io.carray2hdr(img, outf, header)
        else:
            io.array2hdr(img, outf, header)
        return outf

    def direct_view(self, res=512, showsample=False, showweight=True, rnd=False,
                    order=False, srcidx=None, interp=False, omega=False,
                    scalefactor=1, vm=None, fisheye=True, grow=1):
        """create an unweighted summary image of lightpoint"""
        if omega or rnd or order:
            features = 1
            interp = False
        else:
            features = self.features
        if vm is None:
            vm = self.vm
        img, pdirs, mask, mask2, header = vm.init_img(res, self.pt,
                                                      features=features)
        if fisheye:
            header = [header]
        else:
            uv = translate.bin2uv(np.arange(res*res*vm.aspect), res)
            pdirs = vm.uv2xyz(uv).reshape(vm.aspect, res, res, 3)
            pdirs = pdirs[:, ::-1].reshape(vm.aspect*res, res, 3)
            mask = None
            mask2 = None
            header = None
        if showweight:
            grow = 0
            if srcidx is not None:
                skyvec = np.zeros(self.srcn)
                skyvec[srcidx] = scalefactor
            else:
                skyvec = np.full(self.srcn, scalefactor)
            self.add_to_img(img, pdirs[mask], mask2, vm=vm, order=order,
                            interp=interp, skyvec=skyvec, omega=omega, rnd=rnd)
            channels = (1, 0, 0)
        else:
            channels = (1, 1, 1)
        if order:
            outf = self.file.replace("/", "_").replace(".rytpt", "_order.hdr")
        elif omega:
            outf = self.file.replace("/", "_").replace(".rytpt", "_omega.hdr")
        else:
            outf = self.file.replace("/", "_").replace(".rytpt", ".hdr")
        outf = outf.strip(".").strip("_")
        if srcidx is not None:
            try:
                outf = outf.replace(f"_{self.src}_", f"_{self.src}_{srcidx:04d}_")
            except TypeError:
                outf = outf.replace(f"_{self.src}_",
                                    f"_{self.src}_filtered_")
        if showsample:
            if len(img.shape) < 3:
                img = np.repeat(img[None, ...], 3, 0)
            vi = self.query_ball(vm.dxyz, vm.viewangle * vm.aspect)
            v = self.vec[vi[0]]
            img = vm.add_vecs_to_img(img, v, channels=channels,
                                     fisheye=fisheye, mask=mask, grow=grow)
            io.carray2hdr(img, outf, header)
        elif len(img.shape) == 3:
            io.carray2hdr(img, outf, header)
        else:
            io.array2hdr(img, outf, header)
        return outf

    def add(self, lf2, src=None, calcomega=True, write=False, sumsrc=False):
        """add light points of distinct sources together
        results in a new lightpoint with srcn=self.srcn+srcn2 and
        vector size=self.vecsize+vecsize2

        Parameters
        ----------
        lf2: raytraverse.lightpoint.LightPointKD
        src: str, optional
            if None (default), src is "{lf1.src}_{lf2.src}"
        calcomega: bool, optional
            passed to LightPointKD constructor
        write: bool, optional
            passed to LightPointKD constructor
        sumsrc: bool, optional
            if True adds matching source indices together (must be same shape)
            this assumes that the two lightpoints represent the same source
            but different components (such as direct/indirect)
        Returns
        -------
        raytraverse.lightpoint.LightPointKD
            will be subtyped according to self, unless lf2 is needed to preserve
            data
        """
        i, d = self.query_ray(lf2.vec)
        notmatch = d > 1e-6
        i = i[notmatch]
        vecs = np.concatenate((self.vec, lf2.vec[notmatch]), axis=0)

        lum1 = np.concatenate((self.lum, self.lum[i]), axis=0)
        i, d = lf2.query_ray(self.vec)
        lum2 = np.concatenate((lf2.lum[i], lf2.lum[notmatch]), axis=0)
        if sumsrc:
            lums = lum1 + lum2
            srcn = self.srcn
            srcdir = self.srcdir
        else:
            lums = np.concatenate((lum1, lum2), axis=1)
            srcdir = np.concatenate((self.srcdir, lf2.srcdir), axis=0)
            srcn = self.srcn + lf2.srcn
        if src is None:
            src = f"{self.src}_{lf2.src}"
        svi = [i if i >= 0 else self.srcn + i
               for i, sv in zip(self.srcviewidxs, self.srcviews)]
        svi2 = [i + self.srcn if i >= 0 else i
                for i, sv in zip(self.srcviewidxs, self.srcviews)]
        return type(self)(self.scene, vecs, lums, vm=self.vm, pt=self.pt,
                          posidx=self.posidx, src=src, calcomega=calcomega,
                          srcn=srcn, write=write, srcdir=srcdir,
                          srcviews=self.srcviews + lf2.srcviews,
                          srcviewidxs=svi + svi2, filterviews=False,
                          parent=self.parent)

    def update(self, vec, lum, omega=None, calcomega=True, write=True,
               filterviews=False):
        """add additional rays to lightpoint in place

        Parameters
        ----------
        vec: np.array, optional
            shape (N, >=3) where last three columns are normalized direction
            vectors of samples.
        lum: np.array, optional
            reshapeable to (N, srcn). sample values for each source
            corresponding to vec.
        omega: np.array, optional
            provide precomputed omega values, if given, overrides calcomega
        calcomega: bool, optional
            if True (default) calculate solid angle of rays. This  is
            unnecessary if point will be combined before calculating any
            metrics. setting to False will save some computation time. If False,
            resets omega to None!
        write: bool, optional
            whether to save updated ray data to disk.
        filterviews: bool, optional
            delete rays near sourceviews
        """
        self._vec = np.vstack((self.vec, vec[:, -3:]))
        if self.features > 1:
            self._lum = np.vstack((self.lum, lum.reshape(-1, self.srcn,
                                                         self.features)))
        else:
            self._lum = np.vstack((self.lum, lum.reshape(-1, self.srcn)))
        self._d_kd = cKDTree(self.vec)
        if filterviews and len(self.srcviews) > 0:
            for sv in self.srcviews:
                lucky_squirel = self.d_kd.query_ball_point(sv.vec, sv.radius)
                if len(lucky_squirel) > 0:
                    self._vec = np.delete(self.vec, lucky_squirel, 0)
                    self._lum = np.delete(self.lum, lucky_squirel, 0)
                    self._d_kd = cKDTree(self.vec)
        self._omega = omega
        if calcomega and self._omega is None:
            self.calc_omega(False)
        if write:
            self.dump()

    def linear_interp(self, vm, srcvals, destvecs):
        xyp = vm.xyz2vxy(destvecs)
        xys = vm.xyz2vxy(self.vec)
        interp = LinearNDInterpolator(xys, srcvals, fill_value=-1)
        lum = interp(xyp[:, 0], xyp[:, 1])
        if len(lum.shape) > 1:
            neg = np.min(lum, 1) < 0
        else:
            neg = lum < 0
        i, d = self.query_ray(destvecs[neg])
        lum[neg] = srcvals[i]
        return lum

    @staticmethod
    def apply_interp(i, srcvals, weights=None):
        if weights is None:
            return srcvals[i]
        else:
            return np.average(srcvals[i], 1, weights)

    def _content_mask(self, rt, destvecs, i, srfnormtol=5.0, disttol=0.5):
        pts = np.hstack(np.broadcast_arrays(self.pt[None], self.vec))
        pts_o = np.hstack(np.broadcast_arrays(self.pt[None], destvecs))
        # store engine state
        targs = rt.args
        tospec = rt.ospec
        rt.set_args(rt.directargs)
        rt.update_ospec("LNM")
        sgeo = rt(pts)
        sgeo_o = rt(pts_o)
        # restore engine state
        rt.set_args(targs)
        rt.update_ospec(tospec)

        mods = sgeo[:, 4].astype(int)
        norms = np.arccos(sgeo[:, 1:4])
        dist = sgeo[:, 0]
        mods_o = sgeo_o[:, 4].astype(int)
        norms_o = np.arccos(sgeo_o[:, 1:4])
        dist_o = sgeo_o[:, 0]

        mod_match = np.equal(mods[i], mods_o[:, None])
        norm_match = np.all(
            np.isclose(norms[i], norms_o[:, None], atol=srfnormtol*np.pi/180),
            axis=2)
        dist_match = np.abs(dist[i] - dist_o[:, None]) < disttol

        all_match = np.all((mod_match, norm_match, dist_match), axis=0)
        # make sure atleast the closest match is flagged true
        all_match[np.sum(all_match, 1) == 0, 0] = True
        return all_match

    @staticmethod
    def _weight_distance(d):
        # get var on distance (from true mean zero)
        dvar = np.mean(np.square(d), axis=1)[:, None]
        # scale distance on gaussian as weight
        return np.exp(-np.square(d)/(2*dvar))

    def _weight_lum(self, i):
        # get variance on lum
        lum = np.max(self.lum.reshape(self.lum.shape[0], -1)[i], axis=-1)
        dlum = lum - np.mean(lum, axis=1)[:, None]
        lvar = np.mean(np.square(dlum), axis=1)[:, None]
        # scale distance on gaussian as weight
        return np.exp(-np.square(dlum)/(2*lvar))

    def _weight_angle(self, destvecs, i):
        # project to interpolation point and calculate theta
        ymtx, pmtx = translate.rmtx_yp(destvecs)
        nv = np.einsum("vij,vkj,vli->vkl", ymtx, self.vec[i], pmtx)
        ang = np.arctan2(nv[..., 1], nv[..., 0]) + np.pi
        # resort by theta
        asr = np.argsort(ang)
        ang = np.take_along_axis(ang, asr, axis=1)

        # pad and estimate local density
        ang2 = np.hstack((ang[:, -2:] - 2*np.pi, ang, 2*np.pi + ang[:, 0:2]))
        # weighted average of nearby less than
        lookb = correlate1d(ang2, [-1/6, -1/3, .5], origin=1)
        # weighted average of nearby greater than
        lookf = correlate1d(ang2, [-.5, 1/3, 1/6], origin=-1)
        # total local variance in nearby
        return (lookb[:, 2:-2] + lookf[:, 2:-2]), asr

    def interp(self, destvecs, bandwidth=10, rt=None, lum=True, angle=True,
               dither=False, **kwargs):
        d, i = self.d_kd.query(destvecs, bandwidth)
        w = self._weight_distance(d)
        if lum:
            w *= self._weight_lum(i)
        cf = None
        if rt is not None:
            content = self._content_mask(rt, destvecs, i)
            w[np.logical_not(content)] *= 0.05
            cf = np.all(content, 1)
        if angle:
            if cf is not None:
                asr = np.tile(np.arange(bandwidth)[:, None], len(destvecs)).T
                angw = np.ones_like(w)
                angwcf, asrcf = self._weight_angle(destvecs[cf], i[cf])
                asr[cf] = asrcf
                angw[cf] = angwcf
            else:
                angw, asr = self._weight_angle(destvecs, i)
            w = np.take_along_axis(w, asr, axis=1) * angw
            i = np.take_along_axis(i, asr, axis=1)

        w = w / np.sum(w, axis=-1)[:, None]
        if dither:
            dc = np.cumsum(w, 1)
            c = np.random.default_rng().random(len(destvecs))
            ic = np.array(list(map(np.searchsorted, dc, c)))[:, None]
            i = np.take_along_axis(i, ic, 1).ravel()
            w = None
        elif self.features > 1:
            w = np.broadcast_to(w[..., None], w.shape + (self.features,))
        return i, w

