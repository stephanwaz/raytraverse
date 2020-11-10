# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from clasp.script_tools import pipeline, sglob
from raytraverse.lightfield.memarraydict import MemArrayDict
from raytraverse import io
from raytraverse.lightfield.lightfieldkd import LightFieldKD
from raytraverse.lightfield.sunviewfield import SunViewField


class SunField(LightFieldKD):
    """container for sun view data

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun vectors and SunMapper (passed to SunViewField)
    rebuild: bool, optional
        build kd-tree even if one exists
    """

    def __init__(self, scene, suns, rebuild=False, rmraw=False, **kwargs):
        #: raytraverse.sunsetter.SunSetter
        self.suns = suns
        #: raytraverse.lightfield.SunViewField
        self.view = SunViewField(scene, suns, rebuild=False, rmraw=rmraw,
                                 **kwargs)
        super().__init__(scene, rebuild=rebuild, prefix='sun', rmraw=rmraw,
                         fvrays=scene.maxspec)

    def raw_files(self):
        """get list of files used to build field"""
        rf = []
        for i in range(self.suns.suns.shape[0]):
            rf.append(f'{self.scene.outdir}/{self.prefix}_{i:04d}_vals.out')
            rf.append(f'{self.scene.outdir}/{self.prefix}_{i:04d}_vecs.out')
        return rf

    def _mk_tree(self, pref='', ltype=list):
        d_kds = {}
        vecs = {}
        omegas = {}
        lums = MemArrayDict({})
        offset = 0
        npts = self.scene.area.npts
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            futures = []
            for i in range(self.suns.suns.shape[0]):
                dfile = f'{self.scene.outdir}/{self.prefix}_{i:04d}_vals.out'
                if os.path.isfile(dfile):
                    vs, lum = self._get_vl(npts, pref=f'_{i:04d}', ltype=ltype,
                                           os0=offset)
                    lasti = lum[-1][1]
                    offset = lasti[-2] + lasti[-1][0]*lasti[-1][1]*4
                    for j, lm in lum:
                        vecs[(j, i)] = vs[j]
                        lums[(j, i)] = lm
                        futures.append(((j, i),
                                       exc.submit(LightFieldKD.mk_vector_ball,
                                       vs[j])))
            for fu in futures:
                idx = fu[0]
                d_kds[idx], omegas[idx] = fu[1].result()
        return d_kds, vecs, omegas, lums

    def items(self):
        return self.d_kd.keys()

    def ptitems(self, i):
        return [j for j in self.items() if j[0] == i]

    def keymap(self):
        npts = self.scene.area.npts
        shape = (npts, self.suns.suns.shape[0] + 1)
        idxgrid = np.unravel_index(np.arange(shape[1] * npts), shape)
        full = np.core.records.fromarrays(idxgrid)
        items = np.core.records.fromarrays(list(zip(*self.items())))
        return np.isin(full, items).reshape(shape)

    def add_to_img(self, img, mask, pi, vecs, coefs=92444.45, vm=None, interp=1,
                   **kwargs):
        if vm is None:
            vm = self.scene.view
        super().add_to_img(img, mask, pi, vecs, coefs=coefs, interp=interp,
                           **kwargs)
        sun = np.concatenate((self.suns.suns[pi[1]], [coefs, ]))
        self.view.add_to_img(img, pi, sun, vm)

    def get_applied_rays(self, pi, dxyz, skyvec, sunvec=None):
        """the analog to add_to_img for metric calculations"""
        rays, omega, lum = super().get_applied_rays(pi, dxyz, skyvec)
        svw = self.view.get_ray(pi, dxyz, sunvec)
        if svw is not None:
            rays = np.vstack((rays, svw[0][None, :]))
            lum = np.concatenate((lum, [svw[1]]))
            omega = np.concatenate((omega, [svw[2]]))
        return rays, omega, lum

    def direct_view(self, res=512, showsample=True, showweight=True,
                    dpts=None, items=None, srcidx=None):
        """create a summary image of lightfield for each vpt"""
        super().direct_view(res=res, showsample=showsample,
                            showweight=showweight, dpts=dpts, items=items)
        if items is not None:
            return None
        for i in super().items():
            flist = sglob(f"{self.scene.outdir}_{self.prefix}_{i:04d}_*.hdr")
            ssq = int(np.ceil(np.sqrt(len(flist))))
            files = ' '.join(flist)
            outf = f"{self.scene.outdir}_{self.prefix}_{i:04d}.hdr"
            pcompos = f'pcompos -a -{ssq} -s 5 -b 1 1 1 -la {files}'
            xscale = min(ssq*res*2, 2000)
            pfilt = f'pfilt -1 -e 1 -x {xscale} -p 1'
            pipeline([pcompos, pfilt], outf, close=True, writemode='wb')
            for fl in flist:
                os.remove(fl)
