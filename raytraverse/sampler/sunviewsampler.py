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

from raytraverse import draw, translate, io
from raytraverse.lightfield import SCBinField
from raytraverse.sampler import Sampler


class SunViewSampler(Sampler):
    """sample view rays to direct suns.

    here idres and fdres are sampled on a per sun basis for a view centered
    on each sun direction with a view angle of .533 degrees (hardcoded in
    sunmapper class).

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    loadsrc: bool
        include suns.rad in base scene initialization. if False,
        self.engine.load_source must be invoked before call.
    """
    #: deterministic sample draws
    ub = 1

    def __init__(self, scene, suns, srcdef=None, stype='sunview',
                 checkviz=True, **kwargs):
        self.suns = suns
        self._checkviz = checkviz
        if srcdef is None:
            srcdef = f"{scene.outdir}/suns.rad"
        engine_args = scene.formatter.direct_args
        super().__init__(scene, stype=stype, idres=4, fdres=6,
                         srcdef=None, engine_args=engine_args, **kwargs)
        self.engine.load_source(srcdef)

    def sample(self, vecf, vecs):
        """call rendering engine to sample direct view rays"""
        return super().sample(vecf, vecs).ravel()

    def _offset(self, shape, dim):
        """no jitter on sun view because of very fine resolution and potentially
        large number of samples bog down random number generator"""
        return 0.5/dim

    def run_callback(self):
        """post sampling, write full resolution (including interpolated values)
         non zero rays to result file."""
        shape = self.levels[self.idx, -2:]
        size = np.prod(shape)
        si = np.stack(np.unravel_index(np.arange(size), shape)).T
        uv = ((si + .5)/shape)

        def ptv_ptl(v):
            valid = v > self.suns.srct
            cnt = np.sum(valid)
            if cnt > 0:
                return uv[valid], v[valid]
            else:
                return np.arange(0), np.arange(0)

        vecs = []
        lums = []
        if self.vizkeys is None:
            vals = self.weights.reshape(-1, self.weights.shape[2], size)
            for i in range(vals.shape[0]):
                ptvs = []
                ptls = []
                for j in range(vals.shape[1]):
                    ptv, ptl = ptv_ptl(vals[i, j])
                    if ptv.size > 0:
                        print(i, j, ptv.shape)
                    ptvs.append(ptv)
                    ptls.append(ptl)
                lums.append(ptls)
                vecs.append(ptvs)
        else:
            i = -1
            for vizpoint in self.vizmap.reshape(-1,*self.vizmap.shape[2:]):
                ptvs = []
                ptls = []
                for k in vizpoint:
                    if i != k:
                        i = k
                        ptv, ptl = ptv_ptl(self.weights[i].ravel())
                    else:
                        ptv = np.arange(0)
                        ptl = np.arange(0)
                    ptvs.append(ptv)
                    ptls.append(ptl)
                lums.append(ptls)
                vecs.append(ptvs)
        outf = f'{self.scene.outdir}/{self.stype}_result.pickle'
        f = open(outf, 'wb')
        pickle.dump(vecs, f, protocol=4)
        pickle.dump(lums, f, protocol=4)
        pickle.dump(shape, f, protocol=4)
        f.close()
        os.remove(f'{self.scene.outdir}/{self.stype}_vals.out')
        [os.remove(f) for f in self._vecfiles]
