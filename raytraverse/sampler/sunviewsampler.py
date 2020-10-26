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

from raytraverse import draw, renderer
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

    def __init__(self, scene, suns, srcdef=None, stype='sunview', checkviz=True,
                 **kwargs):
        self.suns = suns
        self.engine = renderer.Rtrace()
        self._checkviz = checkviz
        if srcdef is None:
            srcdef = f"{scene.outdir}/suns.rad"
        super().__init__(scene, stype=stype, idres=4, fdres=6,
                         srcdef=None, engine_args='-oZ -ab 0', **kwargs)
        self.engine.load_source(srcdef)
        self.samplemap = self.suns.map

    @property
    def levels(self):
        """sampling scheme

        :getter: Returns the sampling scheme
        :setter: Set the sampling scheme from (ptres, fdres, skres)
        :type: np.array
        """
        return self._levels

    @levels.setter
    def levels(self, fdres):
        """calculate sampling scheme"""
        self._levels = np.array([(self.suns.suns.shape[0], 2**i, 2**i)
                                 for i in range(self.idres, fdres + 1, 1)])

    def check_viz(self):
        skyfield = SCBinField(self.scene)
        idx, errs = skyfield.query_all_pts(self.suns.suns, 4)
        isviz = np.array([np.max(skyfield.lum[i][j], (1, 2)) > self.suns.srct/2
                          for i, j in enumerate(idx)])
        isviz = isviz.reshape(*self.scene.area.ptshape,  -1)
        suns = np.broadcast_to(isviz[..., None, None].astype(int),
                               isviz.shape + (self.weights.shape[-2:]))
        return suns

    def sample(self, vecf):
        """call rendering engine to sample direct view rays

        Parameters
        ----------
        vecf: str
            path of file name with sample vectors
            shape (N, 6) vectors in binary float format

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        return super().sample(vecf).ravel()

    def _uv2xyz(self, uv, si):
        return self.samplemap.uv2xyz(uv, si[2])

    def draw(self):
        """draw first level based on sky visibility"""
        if self.idx == 0 and self._checkviz:
            p = self.check_viz().ravel()
            pdraws = draw.from_pdf(p, .5)
        else:
            pdraws = super().draw()
        return pdraws

    def run_callback(self):
        """post sampling, write full resolution (including interpolated values)
         non zero rays to result file."""
        super().run_callback()
        shape = self.levels[self.idx, -2:]
        size = np.prod(shape)
        vals = self.weights.reshape(-1, self.weights.shape[2], size)
        si = np.stack(np.unravel_index(np.arange(size), shape)).T
        uv = ((si + .5)/shape)
        vecs = []
        lums = []
        for i in range(vals.shape[0]):
            ptv = []
            ptl = []
            for j in range(vals.shape[1]):
                valid = vals[i, j] > self.suns.srct
                cnt = np.sum(valid)
                if cnt > 0:
                    ptv.append(uv[valid])
                    ptl.append(vals[i, j][valid])
                else:
                    ptv.append(np.arange(0))
                    ptl.append(np.arange(0))
            lums.append(ptl)
            vecs.append(ptv)
        outf = f'{self.scene.outdir}/{self.stype}_result.pickle'
        f = open(outf, 'wb')
        pickle.dump(vecs, f, protocol=4)
        pickle.dump(lums, f, protocol=4)
        pickle.dump(shape, f, protocol=4)
        f.close()
        os.remove(f'{self.scene.outdir}/{self.stype}_vals.out')
