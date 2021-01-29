# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import pickle
import sys
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi, Voronoi
from shapely.geometry import Polygon
from clasp.script_tools import try_mkdir

from raytraverse.lightfield.memarraydict import MemArrayDict
from raytraverse import io, translate
from raytraverse.mapper import ViewMapper
from raytraverse.craytraverse import interpolate_kdquery


class LightFieldKD(object):
    """light field with KDtree structures for spatial query"""

    def __init__(self, scene, rebuild=False, src='sky', position=0, srcn=1,
                 rmraw=False, fvrays=0.0, calcomega=True):
        #: float: threshold for filtering direct view rays
        self._fvrays = fvrays
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
        self.srcn = srcn
        #: str: prefix of data files from sampler (stype)
        self.src = src
        self.position = position
        self._vec = None
        self._lum = None
        self._omega = None
        self._rmraw = rmraw
        self.scene = scene
        self.calcomega = calcomega
        self._rawfiles = self.raw_files()

    def raw_files(self):
        return []

    @property
    def vec(self):
        """direction vector (3,)"""
        return self._vec

    @property
    def lum(self):
        """luminance (srcn,)"""
        return self._lum

    @property
    @functools.lru_cache(1)
    def outfile(self):
        outdir = f"{self.scene.outdir}/{self.src}"
        try_mkdir(outdir)
        return f"{outdir}/{self.position:06d}.rytree"

    @property
    def d_kd(self):
        """kd tree for spatial query

        :getter: Returns kd tree structure
        :type: scipy.spatial.cKDTree
        """
        return self._d_kd

    @d_kd.setter
    def d_kd(self, v):
        self._d_kd = cKDTree(v)

    @property
    def omega(self):
        """solid angle

        :getter: Returns array of solid angles
        :setter: sets soolid angles with viewmapper
        :type: np.array
        """
        return self._omega

    @omega.setter
    def omega(self, vm):
        """solid angle"""
        if self.vec.shape[0] < 100:
            self._omega = None
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
            self._omega = np.asarray(omega)
        else:
            try:
                self._omega = SphericalVoronoi(self.vec).calculate_areas()
            except ValueError:
                # spherical voronoi raises a value error when points are
                # too close, in this case we cull duplicates before calculating
                # area, leaving the duplicates with omega=0
                flt = np.full(len(self.vec), True)
                self._omega = np.zeros(self.vec.shape[0])
                tol = 2*np.pi/2**10
                pairs = self.d_kd.query_ball_tree(self.d_kd, tol)
                flagged = set()
                for j, p in enumerate(pairs):
                    if j not in flagged and len(p) > 1:
                        flt[p[1:]] = False
                        flagged.update(p[1:])
                self._omega[flt] = \
                    SphericalVoronoi(self.vec[flt]).calculate_areas()

    @property
    def scene(self):
        """scene information

        :getter: Returns this integrator's scene
        :setter: Set this integrator's scene
        :type: raytraverse.scene.Scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        """Set this field's scene and load samples"""
        self._scene = scene
        kdfile = f'{scene.outdir}/{self.prefix}_kd_data.pickle'
        lumfile = f'{scene.outdir}/{self.prefix}_kd_lum_map.pickle'
        lumdat = f'{self.scene.outdir}/{self.prefix}_kd_lum.dat'
        if self.rebuild and not all([os.path.isfile(p) for p in
                                     self.raw_files()]):
            raise FileNotFoundError("some data-files missing, cannot rebuild"
                                    " lightfield")
        if (os.path.isfile(kdfile) and os.path.isfile(lumfile)
                and not self.rebuild):
            # self.scene.log(self, "Reloading data")
            f = open(kdfile, 'rb')
            self._d_kd = pickle.load(f)
            self._vec = pickle.load(f)
            self._omega = pickle.load(f)
            f.close()
            f = open(lumfile, 'rb')
            self._lum = pickle.load(f)
            f.close()
        else:
            self.scene.log(self, "Building kd-tree")
            f = open(lumdat, 'wb')
            f.close()
            self._d_kd, self._vec, self._omega, self._lum = self._mk_tree()
            f = open(kdfile, 'wb')
            pickle.dump(self.d_kd, f, protocol=4)
            pickle.dump(self.vec, f, protocol=4)
            pickle.dump(self.omega, f, protocol=4)
            f.close()
            f = open(lumfile, 'wb')
            pickle.dump(self._lum, f, protocol=4)
            f.close()

    def raw_files(self):
        """get list of files used to build field"""
        dfile = f'{self.scene.outdir}/{self.prefix}_vals.out'
        vfile = f'{self.scene.outdir}/{self.prefix}_vecs.out'
        return [dfile, vfile]

    def _to_mem_map(self, ar, offset=0):
        outf = f'{self.scene.outdir}/{self.prefix}_kd_lum.dat'
        mar = np.memmap(outf, dtype='<f', mode='r+',
                        offset=offset, shape=ar.shape)
        mar[:] = ar[:]
        mar.flush()
        del mar
        return outf, '<f', 'r', offset, ar.shape

    def _get_vl(self, npts, pref='', ltype=MemArrayDict, os0=0):
        dfile = f'{self.scene.outdir}/{self.prefix}{pref}_vals.out'
        vfile = f'{self.scene.outdir}/{self.prefix}{pref}_vecs.out'
        if not (os.path.isfile(dfile) and os.path.isfile(vfile)):
            raise FileNotFoundError("No results files found, have you run"
                                    f" a Sampler of type {self.prefix} for"
                                    f" scene {self.scene.outdir}?")
        fvecs = io.bytefile2np(open(vfile, 'rb'), (-1, 4))
        alums = io.bytefile2np(open(dfile, 'rb'), (fvecs.shape[0], self.srcn))
        if self._fvrays > 0:
            blindsquirrel = (np.max(alums, 1) < self._fvrays)
            fvecs = fvecs[blindsquirrel]
            alums = alums[blindsquirrel]
        sorting = fvecs[:, 0].argsort()
        fvecs = fvecs[sorting]
        pidx = fvecs[:, 0]
        pt_div = np.searchsorted(pidx, np.arange(npts), side='right')
        pt0 = 0
        vecs = {}
        lums = []
        lummap = []
        with ThreadPoolExecutor() as exc:
            pts = []
            for i, pt in enumerate(pt_div):
                if pt != pt0:
                    pts.append(i)
                    vecs[i] = fvecs[pt0:pt, 1:4]
                    slc = sorting[pt0:pt]
                    ar = alums[slc]
                    lums.append(exc.submit(self._to_mem_map, ar,
                                           offset=os0 + 4*pt0*self.srcn))
                    pt0 = pt
            for pt, lm in zip(pts, lums):
                lummap.append((pt, lm.result()))
        return vecs, ltype(lummap)

    def _mk_tree(self, pref='', ltype=MemArrayDict):
        npts = self.scene.area.npts
        vs, lums = self._get_vl(npts, pref=pref, ltype=ltype)
        if self.scene.view.viewangle < 360:
            uv = [self.scene.view.xyz2uv(v) for v in vs.values()]
            with ProcessPoolExecutor(io.get_nproc()) as exc:
                d_kd, omega = zip(*exc.map(LightFieldKD.mk_vector_view,
                                           vs.values(), uv))
        else:
            with ProcessPoolExecutor(io.get_nproc()) as exc:
                d_kd, omega = zip(*exc.map(LightFieldKD.mk_vector_ball,
                                           vs.values()))
        return dict(zip(vs.keys(), d_kd)), vs, dict(zip(vs.keys(), omega)), lums

    def apply_coef(self, pi, coefs):
        c = np.asarray(coefs).reshape(-1, 1)
        return np.einsum('ij,kj->ik', c, self.lum[pi])

    def add_to_img(self, img, mask, pi, vecs, coefs=1, vm=None, interp=1,
                   omega=False, **kwargs):
        if omega:
            i, d = self.query_ray(pi, vecs)
            lum = self.omega[pi][i]
        elif interp > 1:
            arrout = LightFieldKD.interpolate_query(self.d_kd[pi],
                                                    self.lum[pi],
                                                    self.vec[pi],
                                                    vecs, k=interp, **kwargs)
            if np.asarray(coefs).size == 1:
                lum = arrout * coefs
            else:
                lum = np.einsum('j,kj->k', coefs, arrout)
        else:
            i, d = self.query_ray(pi, vecs)
            lum = self.apply_coef(pi, coefs)
            lum = lum[:, i]
        img[mask] += np.squeeze(lum)

    def get_applied_rays(self, pi, vm, skyvec, sunvec=None):
        """the analog to add_to_img for metric calculations"""
        idx = self.query_ball(pi, vm.dxyz)[0]
        omega = np.squeeze(self.omega[pi][idx])
        rays = self.vec[pi][idx]
        lum = np.squeeze(self.apply_coef(pi, skyvec))[idx]
        return rays, omega, lum

    def query_ray(self, pi, vecs, interp=1):
        d, i = self.d_kd[pi].query(vecs, k=interp)
        return i, d

    def query_all_pts(self, vecs, interp=1):
        futures = []
        idxs = []
        errs = []
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            for pt in self.items():
                futures.append(exc.submit(self.d_kd[pt].query, vecs, interp))
            for fu in futures:
                r = fu.result()
                idxs.append(r[1])
                errs.append(r[0])
        return np.stack(idxs), np.stack(errs)

    def query_ball(self, pi, vecs, viewangle=180):
        vs = translate.theta2chord(viewangle/360*np.pi)
        return self.d_kd[pi].query_ball_point(translate.norm(vecs), vs)

    @staticmethod
    def interpolate_query(kd, lum, vec, dest_vec, k=8,
                          err=0.00436, up=0.347296):
        """query a kd_tree and interpolate corresponding values. used to
        merge to kd_trees with vector and luminance

        Parameters
        ----------
        kd:
            kd-tree
        lum:
            lum array
        vec:
            vector array
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
        if k == 1:
            errs, idxs = kd.query(dest_vec)
            arrout = lum[idxs]
        else:
            errs, idxs = kd.query(dest_vec, k=k, distance_upper_bound=up)
            arrout = interpolate_kdquery(dest_vec, errs, idxs, vec, lum, err=err)
        return arrout

    def _dview(self, vm, idx, res=512, showsample=True,
               showweight=True, srcidx=None, interp=1, omega=False):
        pdirs = vm.pixelrays(res)
        if vm.aspect == 2:
            mask = vm.in_view(np.concatenate((pdirs[0:res],
                                              -pdirs[res:]), 0))
        else:
            mask = vm.in_view(pdirs)
        img = np.zeros((res*vm.aspect, res))
        if showweight:
            if srcidx is not None:
                coefs = np.zeros(self.srcn)
                coefs[srcidx] = 1
                self.add_to_img(img, mask, idx, pdirs[mask], vm=vm,
                                interp=interp, coefs=coefs, omega=omega)
            else:
                self.add_to_img(img, mask, idx, pdirs[mask], vm=vm,
                                interp=interp, omega=omega)
            channels = (1, 0, 0)
        else:
            channels = (1, 1, 1)
        outf = f"{self.outfile(idx)}.hdr"
        try:
            pt = self.scene.area.pts()[idx[0]]
        except (TypeError, IndexError):
            pt = self.scene.area.pts()[idx]
        vstr = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'
                ' -vp {4} {5} {6}'.format(vm.viewangle, *vm.dxyz[0], *pt))
        if showsample:
            img = np.repeat(img[None, ...], 3, 0)
            vi = self.query_ball(idx, vm.dxyz, vm.viewangle)
            v = self.vec[idx][vi[0]]
            img = io.add_vecs_to_img(vm, img, v, channels=channels)
            io.carray2hdr(img, outf, [vstr])
        else:
            io.array2hdr(img, outf, [vstr])
        return outf

    def direct_view(self, res=512, showsample=True, showweight=True,
                    dpts=None, items=None, srcidx=None, interp=1, omega=False):
        """create a summary image of lightfield for each vpt"""
        if items is None:
            items = list(self.items())
        if dpts is not None:
            vm = ViewMapper(dpts[:, 3:6], viewangle=180)
            perrs, pis = self.scene.area.pt_kd.query(dpts[:, 0:3])
            items = []
            vmi = []
            for j, pi in enumerate(pis):
                pti = list(self.ptitems(pi))
                items += pti
                vmi += [j]*len(pti)
        else:
            vm = self.scene.view
            vmi = [0] * len(items)
        fu = []
        with ThreadPoolExecutor() as exc:
            for idx, vi in zip(items, vmi):
                fu.append(exc.submit(self._dview, vm[vi], idx, res,
                                     showsample, showweight, srcidx, interp,
                                     omega))
            for f in as_completed(fu):
                print(f.result(), file=sys.stderr)
