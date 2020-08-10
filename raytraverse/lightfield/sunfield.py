# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm

from clasp.script_tools import pipeline, sglob
from raytraverse.lightfield.memarraydict import MemArrayDict
from raytraverse import translate
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

    def __init__(self, scene, suns, rebuild=False, rmraw=False):
        #: raytraverse.sunsetter.SunSetter
        self.suns = suns
        #: raytraverse.lightfield.SunViewField
        self.view = SunViewField(scene, suns, rebuild=False, rmraw=rmraw)
        super().__init__(scene, rebuild=rebuild, prefix='sun', rmraw=rmraw)

    def raw_files(self):
        """get list of files used to build field"""
        rf = []
        for i in range(self.suns.suns.shape[0]):
            rf.append(f'{self.scene.outdir}/{self.prefix}_{i:04d}_vals.out')
            rf.append(f'{self.scene.outdir}/{self.prefix}_{i:04d}_vecs.out')
        return rf + self.view.raw_files()

    def _mk_tree(self, pref='', ltype=list, os0=0):
        d_kds = {}
        vecs = {}
        omegas = {}
        lums = MemArrayDict({})
        offset = 0
        npts = self.scene.area.npts
        with ProcessPoolExecutor() as exc:
            futures = []
            for i in range(self.suns.suns.shape[0]):
                dfile = f'{self.scene.outdir}/{self.prefix}_{i:04d}_vals.out'
                if os.path.isfile(dfile):
                    vs, lum = self._get_vl(npts, pref=f'_{i:04d}', ltype=ltype,
                                           os0=offset, fvrays=True)
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

    def keymap(self):
        npts = self.scene.area.npts
        shape = (npts, self.suns.suns.shape[0] + 1)
        idxgrid = np.unravel_index(np.arange(shape[1] * npts), shape)
        full = np.core.records.fromarrays(idxgrid)
        items = np.core.records.fromarrays(list(zip(*self.items())))
        return np.isin(full, items).reshape(shape)

    def add_to_img(self, img, mask, pi, i, d, coefs=1, vm=None, radius=3):
        if vm is None:
            vm = self.scene.view
        lum = self.apply_coef(pi, coefs)
        if len(i.shape) > 1:
            y = norm(scale=translate.theta2chord(radius*np.pi/180))
            w = np.broadcast_to(y.pdf(d), (lum.shape[0],) + d.shape)
            lum = np.average(lum[:, i], weights=w, axis=-1)
        else:
            lum = lum[:, i]
        img[mask] += np.squeeze(lum)
        sun = np.concatenate((self.suns.suns[pi[1]], [coefs, ]))
        self.view.add_to_img(img, pi, sun, vm)

    def direct_view(self, res=200, showsample=False, items=None):
        """create a summary image of lightfield for each vpt"""
        super().direct_view(res=res, showsample=showsample, items=items)
        if items is not None:
            return None
        for i in super().items():
            flist = sglob(f"{self.scene.outdir}_{self.prefix}_{i:04d}_*.hdr")
            ssq = int(np.ceil(np.sqrt(len(flist))))
            files = ' '.join(flist)
            outf = f"{self.scene.outdir}_{self.prefix}_{i:04d}.hdr"
            pcompos = f'pcompos -a -{ssq} -s 5 -b 1 1 1 -la {files}'
            pipeline([pcompos], outf, close=True, writemode='wb')
            for fl in flist:
                os.remove(fl)
