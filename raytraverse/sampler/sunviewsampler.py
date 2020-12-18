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

from raytraverse import draw, translate
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
        self.samplemap = self.suns.map
        self.vizkeys = None
        self.vizmap = None

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
        """to avoid massive memory usage, the weights array for sunview
        needs to be stored sparsely, this method resizes self.weights and
        stores the keys needed to map back to the dense representation for
        indexing draw samples"""
        skyfield = SCBinField(self.scene)
        # 4 rays from each point closest to each sun direction
        # shape (suns, 4, sky bins)
        idx, errs = skyfield.query_all_pts(self.suns.suns, 4)

        # sky bin(s) in direction of suns
        # shape (suns, 3) -> duplicates index to fill array if less than
        # 3 sky bins are needed.
        sbins = []
        sunuvs = translate.xyz2uv(self.suns.suns, flipu=False)
        tol = .125/self.scene.skyres
        uvi = np.linspace(-tol, tol, 3)
        for uv in sunuvs:
            uvs = np.stack(np.meshgrid(uvi, uvi)).reshape(2, 9).T + uv
            sbin = np.unique(translate.uv2bin(uvs,
                                              self.scene.skyres)).astype(int)
            sbin = sbin[sbin <= self.scene.skyres**2]
            if len(sbin) < 3:
                sbins.append(list(sbin) + [sbin[-1]] * (3 - len(sbin)))
            else:
                sbins.append(list(sbin)[0:3])
        sbins = np.array(sbins)

        # array slices, for each sun (xi) match with corresponding skybin (zi)
        xi = np.arange(sbins.shape[0], dtype=int)[:, None, None]
        yi = np.arange(4, dtype=int)[None, :, None]
        zi = sbins[:, None, :]

        isviz = np.array([np.max(skyfield.lum[i][j][(xi, yi, zi)], (1, 2))
                          > self.suns.srct/2 for i, j in enumerate(idx)])
        isviz = isviz.reshape(*self.area.ptshape,  -1)
        # for collapsing from indices to weights (compression)
        self.vizmap = (np.cumsum(isviz) - 1).reshape(isviz.shape)
        # indices of the pt/sun combos to run (decompression)
        self.vizkeys = np.indices(isviz.shape)[:, isviz].T
        self.weights = np.ones((self.vizkeys.shape[0],) +
                               (self.weights.shape[-2:]))

    def sample(self, vecf, vecs):
        """call rendering engine to sample direct view rays"""
        return super().sample(vecf, vecs).ravel()

    def sample_idx(self, pdraws):
        """generate samples vectors from flat draw indices

        Parameters
        ----------
        pdraws: np.array
            flat index positions of samples to generate

        Returns
        -------
        si: np.array
            index array of draws matching samps.shape
        vecs: np.array
            sample vectors
        """
        shape = np.concatenate((self.area.ptshape, self.levels[self.idx]))
        # index assignment
        si = np.stack(np.unravel_index(pdraws, self.weights.shape))
        # decompress weights
        if self.vizkeys is not None:
            si = np.vstack((self.vizkeys[si[0]].T, si[1:]))
        # convert to UV directions and positions
        uv = si.T[:, -2:]/shape[3]
        pos = self.area.uv2pt((si.T[:, 0:2] + .5)/shape[0:2])
        uv += self._offset(uv.shape)
        if pos.size == 0:
            xyz = pos
        else:
            xyz = self._uv2xyz(uv, si)
        vecs = np.hstack((pos, xyz))
        return si, vecs

    def _offset(self, shape):
        """no jitter on sun view because of very fine resolution and potentially
        large number of samples bog down random number generator"""
        return 0.5/self.levels[self.idx][-1]

    def _uv2xyz(self, uv, si):
        return self.samplemap.uv2xyz(uv, si[2])

    def draw(self):
        """draw first level based on sky visibility"""
        if not self._checkviz:
            return super().draw()
        if self.idx == 0:
            self.check_viz()
            p = self.weights.ravel()
        # use wavelet transform
        elif self.detailfunc == 'wavelet':
            p = draw.get_detail(self.weights, (1, 2))
        # use filter banks
        else:
            p = draw.get_detail_filter(self.weights,
                                       *self.filters[self.detailfunc])
        # draw on pdf
        pdraws = draw.from_pdf(p, self.threshold(self.idx), ub=1)
        return pdraws

    def update_weights(self, si, lum):
        """update self.weights (which holds values used to calculate pdf)

        Parameters
        ----------
        si: np.array
            multidimensional indices to update
        lum:
            values to update with

        """
        if self.vizkeys is not None:
            widx = self.vizmap[tuple(si[0:3])]
            si = np.vstack((widx, si[3:]))
        self.weights[tuple(si)] = lum

    def levelup_weights(self):
        """prepare weights for sampling at current level"""
        if self.vizkeys is None:
            shape = np.concatenate((self.area.ptshape,
                                    self.levels[self.idx]))
        else:
            shape = np.concatenate((self.weights.shape[0:1],
                                    self.levels[self.idx][1:]))
        self.weights = translate.resample(self.weights, shape)
        return shape

    def run_callback(self):
        """post sampling, write full resolution (including interpolated values)
         non zero rays to result file."""
        super().run_callback()
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
