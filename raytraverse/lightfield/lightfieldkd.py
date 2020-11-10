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
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
from scipy.spatial import cKDTree, SphericalVoronoi, _voronoi

import numpy as np
from raytraverse.lightfield.memarraydict import MemArrayDict

from raytraverse import io, translate
from raytraverse.mapper import ViewMapper
from raytraverse.lightfield.lightfield import LightField
from raytraverse.craytraverse import interpolate_kdquery


class LightFieldKD(LightField):
    """light field with KDtree structures for spatial query"""

    @staticmethod
    def mk_vector_ball(v):

        class SVoronoi(SphericalVoronoi):
            """this is a temporary fix for an apperent bug in
            SphericalVoronoi"""
            def sort_vertices_of_regions(self):
                if self._dim != 3:
                    raise TypeError(
                        "Only supported for three-dimensional point sets")
                reg = [r for r in self.regions if len(r) > 0]
                _voronoi.sort_vertices_of_regions(self._simplices, reg)

        d_kd = cKDTree(v)
        omega = SVoronoi(v).calculate_areas()[:, None]
        return d_kd, omega * (np.pi * 4/np.sum(omega))

    @property
    def d_kd(self):
        """list of direction kdtrees

        :getter: Returns kd tree structure
        :type: list of scipy.spatial.cKDTree
        """
        return self._d_kd

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
            self.scene.log(self, "Reloading data")
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
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            d_kd, omega = zip(*exc.map(LightFieldKD.mk_vector_ball,
                                       vs.values()))
        return dict(zip(vs.keys(), d_kd)), vs, dict(zip(vs.keys(), omega)), lums

    def apply_coef(self, pi, coefs):
        c = np.asarray(coefs).reshape(-1, 1)
        return np.einsum('ij,kj->ik', c, self.lum[pi])

    def add_to_img(self, img, mask, pi, vecs, coefs=1, vm=None, interp=1,
                   **kwargs):
        if interp > 1:
            arrout = self.interpolate_query(pi, vecs, k=interp, **kwargs)
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

    def interpolate_query(self, pi, dest_vec, k=8,
                          err=0.00436, up=0.347296):
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
        errs, idxs = self.d_kd[pi].query(dest_vec, k=k, distance_upper_bound=up)
        if k == 1:
            return self.lum[pi][idxs]
        arrout = interpolate_kdquery(dest_vec, errs, idxs, self.vec[pi],
                                     self.lum[pi], err=err)
        return arrout

    def _dview(self, vm, idx, pdirs, mask, res=512, showsample=True,
               showweight=True, srcidx=None):
        img = np.zeros((res*vm.aspect, res))
        if showweight:
            if srcidx is not None:
                coefs = np.zeros(self.srcn)
                coefs[srcidx] = 1
                self.add_to_img(img, mask, idx, pdirs[mask], vm=vm, coefs=coefs)
            else:
                self.add_to_img(img, mask, idx, pdirs[mask], vm=vm)
            channels = (1, 0, 0)
        else:
            channels = (1, 1, 1)
        outf = f"{self.outfile(idx)}.hdr"
        if showsample:
            img = np.repeat(img[None, ...], 3, 0)
            vi = self.query_ball(idx, vm.dxyz, vm.viewangle)
            v = self.vec[idx][vi[0]]
            img = io.add_vecs_to_img(vm, img, v, channels=channels)
            io.carray2hdr(img, outf)
        else:
            io.array2hdr(img, outf)
        return outf

    def direct_view(self, res=512, showsample=True, showweight=True,
                    dpts=None, items=None, srcidx=None):
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
        pdirs = vm.pixelrays(res)
        if vm.aspect == 2:
            mask = vm.in_view(np.concatenate((pdirs[0:res],
                                              -pdirs[res:]), 0))
        else:
            mask = vm.in_view(pdirs)
        fu = []
        with ThreadPoolExecutor() as exc:
            for idx, vi in zip(items, vmi):
                fu.append(exc.submit(self._dview, vm[vi], idx, pdirs, mask, res,
                                     showsample, showweight, srcidx))
            for f in as_completed(fu):
                print(f.result(), file=sys.stderr)
