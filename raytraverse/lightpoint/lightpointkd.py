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


class LightPointKD(object):
    """light field with KDtree structures for spatial query"""

    def __init__(self, scene, vec=None, lum=None, vm=None, pt=(0, 0, 0),
                 posidx=0, src='sky', srcn=1, calcomega=True,
                 write=True):
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
            self._d_kd, self._vec, self._lum, clr = self._build(vec, lum, srcn)
            self._omega = None
            if calcomega:
                self.calc_omega(False)
            if write:
                self.dump()
                for rf in clr:
                    try:
                        os.remove(rf)
                    except IOError:
                        pass
            scene.log(self, f"build complete")
        elif os.path.isfile(self.file):
            f = open(self.file, 'rb')
            self._d_kd, self._vec, self._omega, self._lum = pickle.load(f)
            f.close()
        else:
            raise ValueError(f"Cannot initialize {type(self).__name__} without"
                             f" file: {self.file} or parameters 'vec' "
                             f"and 'lum'")
        self.srcn = self.lum.shape[1]

    def dump(self):
        try_mkdir(f"{self.scene.outdir}/{self.src}")
        f = open(self.file, 'wb')
        pickle.dump((self._d_kd, self._vec, self._omega, self._lum), f,
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
        elif vm.viewangle <= 180:
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
        try:
            c = np.asarray(coefs).reshape(-1, self.srcn)
        except ValueError:
            c = np.broadcast_to(coefs, (1, self.srcn))
        return np.einsum('ij,kj->ik', c, self.lum)

    def add_to_img(self, img, vecs, mask=None, coefs=1, interp=False,
                   omega=False, vm=None):
        """add luminance contributions to image array (updates in place)

        Parameters
        ----------
        img: np.array
            2D image array to add to (either zeros or with other source)
        vecs: np.array
            vectors corresponding to img pixels shape (N, 3)
        mask: np.array
            indices to img that correspond to vec (in case where whole image
            is not being updated, such as corners of fisheye)
        coefs: int, float, np.array
            source coefficients, shape is (1,) or (srcn,)
        interp: bool, optional
            for linear interpolation (falls back to nearest outside of
            convexhull
        omega: bool
            if true, add value of ray solid angle instead of luminance
        vm: raytraverse.mapper.ViewMapper, optional

        Returns
        -------
        None

        """
        if omega:
            val = self.omega.reshape(1, -1)
        else:
            val = self.apply_coef(coefs)
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
        """the analog to add_to_img for metric calculations"""
        if vm is None:
            vm = self.vm
        idx = self.query_ball(vm.dxyz)[0]
        omega = np.squeeze(self.omega[idx])
        rays = self.vec[idx]
        lum = np.squeeze(self.apply_coef(skyvec))[idx]
        return rays, omega, lum

    def query_ray(self, vecs):
        d, i = self.d_kd.query(vecs)
        return i, d

    def query_ball(self, vecs, viewangle=180):
        vs = translate.theta2chord(viewangle/360*np.pi)
        return self.d_kd.query_ball_point(translate.norm(vecs), vs)

    def direct_view(self, res=512, showsample=False, showweight=True,
                    srcidx=None, interp=False, omega=False, scalefactor=1):
        """create a summary image of lightfield for each vpt"""
        vm = self.vm
        pdirs = vm.pixelrays(res)
        mask = vm.in_view(pdirs[0:res])
        img = np.zeros((res*vm.aspect, res))
        if showweight:
            if srcidx is not None:
                coefs = np.zeros(self.srcn)
                coefs[srcidx] = scalefactor
            else:
                coefs = np.full(self.srcn, scalefactor / self.srcn)
            self.add_to_img(img, pdirs[mask], mask,
                            interp=interp, coefs=coefs, omega=omega)
            if vm.aspect == 2:
                mask = vm.ivm.in_view(pdirs[res:])
                self.add_to_img(img[res:], pdirs[res:][mask], mask, vm=vm.ivm,
                                interp=interp, coefs=coefs, omega=omega)
            channels = (1, 0, 0)
        else:
            channels = (1, 1, 1)
        if omega:
            outf = self.file.replace("/", "_").replace(".rytpt", "_omega.hdr")
        else:
            outf = self.file.replace("/", "_").replace(".rytpt", ".hdr")
        vstr = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'
                ' -vp {4} {5} {6}'.format(vm.viewangle, *vm.dxyz[0], *self.pt))
        if showsample:
            img = np.repeat(img[None, ...], 3, 0)
            vi = self.query_ball(vm.dxyz, vm.viewangle)
            v = self.vec[vi[0]]
            img = io.add_vecs_to_img(vm, img, v, channels=channels)
            io.carray2hdr(img, outf, [vstr])
        else:
            io.array2hdr(img, outf, [vstr])
        return outf

    @staticmethod
    def _build(vec, lum, srcn):
        """load samples and build data structure"""
        clear = []
        try:
            f = open(vec, 'rb')
        except TypeError:
            pass
        else:
            clear.append(vec)
            vec = io.bytefile2np(f, (-1, 6))
            f.close()
        vec = vec[:, -3:]
        try:
            f = open(lum, 'rb')
        except TypeError:
            lum = lum.reshape(-1, srcn)
        else:
            clear.append(lum)
            lum = io.bytefile2np(f, (-1, srcn))
        d_kd = cKDTree(vec)
        return d_kd, vec, lum, clear
