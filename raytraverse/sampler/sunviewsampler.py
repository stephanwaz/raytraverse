# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import pickle

import numpy as np

from raytraverse import draw
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
    """

    def __init__(self, scene, suns, **kwargs):
        self.suns = suns
        sunfile = f"{scene.outdir}/suns.rad"
        super().__init__(scene, stype='sunview', idres=4, fdres=6,
                         srcdef=sunfile, **kwargs)
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
        sunuv = self.scene.view.xyz2uv(self.suns.suns)
        inview = self.scene.in_view(sunuv)
        idx, errs = skyfield.query_all_pts(self.suns.suns, 4)
        isviz = np.array([np.max(skyfield.lum[i][j], (1, 2)) > self.suns.srct/2
                          for i, j in enumerate(idx)])
        isviz = np.logical_and(isviz,
                               inview[None]).reshape(*self.scene.ptshape, -1)
        suns = np.broadcast_to(isviz[..., None, None].astype(int),
                               isviz.shape + (self.weights.shape[-2:]))
        return suns

    def sample(self, vecs, rcopts='-ab 0',
               nproc=12):
        """call rendering engine to sample direct view rays

        Parameters
        ----------
        vecs: np.array
            shape (N, 6) vectors to calculate contributions for
        rcopts: str, optional
            option string to send to executable
        nproc: int, optional
            number of processes executable should use

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        rc = f"rtrace -fff {rcopts} -h -n {nproc} {self.compiledscene}"
        return super().sample(vecs, call=rc).ravel()

    def _uv2xyz(self, uv, si):
        return self.samplemap.uv2xyz(uv, si[2])

    def dump_vecs(self, si, vecs):
        """save vectors to file

        Parameters
        ----------
        si: np.array
            sample indices
        vecs: np.array
            ray directions to write
        """
        pass

    def draw(self):
        """draw first level based on sky visibility"""
        if self.idx == 0:
            p = self.check_viz().ravel()
            pdraws = draw.from_pdf(p, .5)
        else:
            pdraws = super().draw()
        return pdraws

    def run_callback(self):
        """post sampling, right full resolution (including interpolated values)
         non zero rays to result file."""
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
