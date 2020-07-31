# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
from scipy.spatial import cKDTree
from memory_profiler import profile

import numpy as np
from raytraverse.helpers import SVoronoi, MemArrayDict

from raytraverse import io, translate
from raytraverse.lightfield.lightfield import LightField


class LightFieldKD(LightField):
    """light field with KDtree structures for spatial query"""

    @staticmethod
    def mk_vector_ball(v):
        d_kd = cKDTree(v)
        omega = SVoronoi(v).calculate_areas()[:, None]
        return d_kd, omega

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
        if (os.path.isfile(kdfile) and os.path.isfile(lumfile)
                and not self.rebuild):
            f = open(kdfile, 'rb')
            self._d_kd = pickle.load(f)
            self._vec = pickle.load(f)
            self._omega = pickle.load(f)
            f.close()
            f = open(lumfile, 'rb')
            self._lum = pickle.load(f)
            f.close()
        else:
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
        if type(ar) == tuple and type(ar[0]) == str:
            ar = io.bytefile2rad(ar[0], ar[1], slc=ar[2], subs='ijk,k->ij')
        outf = f'{self.scene.outdir}/{self.prefix}_kd_lum.dat'
        mar = np.memmap(outf, dtype='<f', mode='r+',
                        offset=offset, shape=ar.shape)
        mar[:] = ar[:]
        mar.flush()
        del mar
        return outf, '<f', 'r', offset, ar.shape

    def _get_vl(self, npts, pref='', ltype=MemArrayDict, os0=0, fvrays=False):
        dfile = f'{self.scene.outdir}/{self.prefix}{pref}_vals.out'
        vfile = f'{self.scene.outdir}/{self.prefix}{pref}_vecs.out'
        if not (os.path.isfile(dfile) and os.path.isfile(vfile)):
            raise FileNotFoundError("No results files found, have you run"
                                    f" a Sampler of type {self.prefix} for"
                                    f" scene {self.scene.outdir}?")
        fvecs = io.bytefile2np(open(vfile, 'rb'), (-1, 4))
        ishape = (fvecs.shape[0], self.srcn, 3)
        if fvrays:
            alums = io.bytefile2rad(dfile, ishape, subs='ijk,k->ij')
            blindsquirrel = (np.max(alums, 1) < self.scene.maxspec)
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
                    try:
                        ar = alums[slc]
                    except NameError:
                        ar = (dfile, ishape, slc)
                    lums.append(exc.submit(self._to_mem_map, ar,
                                           offset=os0 + 4*pt0*self.srcn))
                    pt0 = pt
            for pt, lm in zip(pts, lums):
                lummap.append((pt, lm.result()))
        return vecs, ltype(lummap)

    def _mk_tree(self, pref='', ltype=MemArrayDict, os0=0):
        npts = np.product(self.scene.ptshape)
        vs, lums = self._get_vl(npts, pref=pref, ltype=ltype, os0=os0)
        with ProcessPoolExecutor() as exc:
            d_kd, omega = zip(*exc.map(LightFieldKD.mk_vector_ball,
                                       vs.values()))
        return dict(zip(vs.keys(), d_kd)), vs, dict(zip(vs.keys(), omega)), lums

    def apply_coef(self, pi, coefs):
        c = np.asarray(coefs).reshape(-1, 1)
        return np.einsum('ij,kj->ik', c, self.lum[pi])

    def add_to_img(self, img, mask, pi, i, d, coefs=1, vm=None, radius=3):
        lum = self.apply_coef(pi, coefs)
        if len(i.shape) > 1:
            # gaussian reconstruction filter
            y = norm(scale=translate.theta2chord(radius*np.pi/180))
            w = np.broadcast_to(y.pdf(d), (lum.shape[0],) + d.shape)
            lum = np.average(lum[:, i], weights=w, axis=-1)
        else:
            lum = lum[:, i]
        img[mask] += np.squeeze(lum)

    def get_illum(self, vm, pis, vdirs, coefs, scale=179):
        illums = []
        for pi in pis:
            lm = self.apply_coef(pi, coefs)
            idx = self.query_ball(pi, vdirs)
            illum = []
            for j, i in enumerate(idx):
                v = self.vec[pi][i]
                o = self.omega[pi][i]
                illum.append(np.einsum('j,ij,j,->i', vm.ctheta(v, j), lm[:, i],
                                       o, scale))
            illums.append(illum)
        return np.squeeze(illums)

    def query_ray(self, pi, vecs, interp=1):
        d, i = self.d_kd[pi].query(vecs, k=interp)
        return i, d

    def query_all_pts(self, vecs, interp=1):
        futures = []
        idxs = []
        errs = []
        with ProcessPoolExecutor() as exc:
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

    def _dview(self, idx, pdirs, mask, res=800, showsample=True):
        img = np.zeros((res*self.scene.view.aspect, res))
        i, d = self.query_ray(idx, pdirs[mask])
        self.add_to_img(img, mask, idx, i, d)
        outf = f"{self.outfile(idx)}.hdr"
        if showsample:
            vm = self.scene.view
            img = np.repeat(img[None, ...], 3, 0)
            vi = self.query_ball(idx, vm.dxyz, vm.viewangle)
            v = self.vec[idx][vi[0]]
            reverse = vm.degrees(v) > 90
            pa = vm.ivm.ray2pixel(v[reverse], res)
            pa[:, 0] += res
            pb = vm.ray2pixel(v[np.logical_not(reverse)], res)
            xp = np.concatenate((pa[:, 0], pb[:, 0]))
            yp = np.concatenate((pa[:, 1], pb[:, 1]))
            img[1:, xp, yp] = 0
            img[0, xp, yp] = 1
            io.carray2hdr(img, outf)
        else:
            io.array2hdr(img, outf)
        return outf

    def direct_view(self, res=800, showsample=True, items=None):
        """create a summary image of lightfield for each vpt"""
        vm = self.scene.view
        pdirs = vm.pixelrays(res)
        if items is None:
            items = self.items()
        if vm.aspect == 2:
            mask = vm.in_view(np.concatenate((pdirs[0:res],
                                              -pdirs[res:]), 0))
        else:
            mask = vm.in_view(pdirs)
        fu = []
        with ThreadPoolExecutor() as exc:
            for idx in items:
                fu.append(exc.submit(self._dview, idx, pdirs, mask, res,
                                     showsample))
            for f in as_completed(fu):
                print(f.result())
