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

import numpy as np

from raytraverse import io, optic, translate
from raytraverse.lightfield.lightfield import LightField


class LightFieldKD(LightField):
    """light field with KDtree structures for spatial query"""

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
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._d_kd = pickle.load(f)
            self._vlo = pickle.load(f)
            f.close()
        else:
            self._d_kd, self._vlo = self._mk_tree()
            f = open(kdfile, 'wb')
            pickle.dump(self.d_kd, f, protocol=4)
            pickle.dump(self.vlo, f, protocol=4)
            f.close()

    def _get_vl(self, npts, pref=''):
        dfile = f'{self.scene.outdir}/{self.prefix}{pref}_vals.out'
        vfile = f'{self.scene.outdir}/{self.prefix}{pref}_vecs.out'
        if not (os.path.isfile(dfile) and os.path.isfile(vfile)):
            raise FileNotFoundError("No results files found, have you run"
                                    f" a Sampler of type {self.prefix} for"
                                    f" scene {self.scene.outdir}?")
        fvecs = io.bytefile2np(open(vfile, 'rb'), (-1, 4))
        sorting = fvecs[:, 0].argsort()
        fvals = optic.rgb2rad(io.bytefile2np(open(dfile, 'rb'), (-1, 3)))
        fvals = fvals.reshape(fvecs.shape[0], -1)[sorting]
        fvecs = fvecs[sorting]
        pidx = fvecs[:, 0]
        pt_div = np.searchsorted(pidx, np.arange(npts), side='right')
        pt0 = 0
        vl = []
        for i, pt in enumerate(pt_div):
            vl.append(np.hstack((fvecs[pt0:pt, 1:4], fvals[pt0:pt])))
            pt0 = pt
        return vl

    def _mk_tree(self):
        return None, None

    def apply_coef(self, pi, coefs):
        return np.einsum('ij,kj->ik', coefs, self.vlo[pi][:, 3:-1])

    def add_to_img(self, img, mask, pi, i, d, coefs=1):
        lum = self.apply_coef(pi, coefs)
        if len(i.shape) > 1:
            w = np.broadcast_to(1/d, (lum.shape[0],) + d.shape)
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
                v = self.vlo[pi][i, 0:3]
                o = self.vlo[pi][i, -1]
                illum.append(np.einsum('j,ij,j,->i', vm.ctheta(v, j), lm[:, i],
                                       o, scale))
            illums.append(illum)
        return np.squeeze(illums)

    def query_ray(self, pi, vecs, interp=1):
        d, i = self.d_kd[pi].query(vecs, k=interp)
        return i, d

    def query_ball(self, pi, vecs, viewangle=180):
        vs = translate.theta2chord(viewangle/360*np.pi)
        return self.d_kd[pi].query_ball_point(translate.norm(vecs), vs)

    def _dview(self, idx, pdirs, mask, res=800):
        img = np.zeros((res, res*self.scene.view.aspect))
        i, d = self.query_ray(idx, pdirs[mask])
        self.add_to_img(img, mask, idx, i, d)
        outf = f"{self.outfile(idx)}.hdr"
        io.array2hdr(img, outf)
        return outf

    def direct_view(self, res=800):
        """create a summary image of lightfield for each vpt"""
        vm = self.scene.view
        pdirs = vm.pixelrays(res)
        if vm.aspect == 2:
            mask = vm.in_view(np.concatenate((pdirs[:, 0:res],
                                              -pdirs[:, res:]), 1))
        else:
            mask = vm.in_view(pdirs)
        fu = []
        with ThreadPoolExecutor() as exc:
            for idx in self.items():
                fu.append(exc.submit(self._dview, idx, pdirs, mask))
        [print(f.result()) for f in as_completed(fu)]
