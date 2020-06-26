# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np

from raytraverse import io, wavelet, Sampler, translate


class SunViewSampler(Sampler):
    """sample view rays to direct suns.

    here idres and fdres are sampled on a per sun basis for a view centered
    on each sun direction with a view angle of .7 degrees.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    """

    def __init__(self, scene, suns):
        self.suns = suns
        super(SunViewSampler, self).__init__(scene, stype='sunview', idres=4,
                                             fdres=6, t0=0, t1=0, minrate=.05,
                                             maxrate=1)
        self.samplemap = self.suns.map
        self.srcn = self.suns.suns.shape[0]
        self.init_weights()

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

    def init_weights(self):
        shape = np.concatenate((self.scene.ptshape,
                                (self.scene.view.aspect*1024, 1024)))
        try:
            skypdf = np.load(f"{self.scene.outdir}/sky_vis.npy")
        except FileNotFoundError:
            print('Warning! sunsampler initialized without vector weights')
            vis = np.ones(shape)
        else:
            vis = translate.resample(skypdf, shape)
        sunuv = self.scene.view.xyz2uv(self.suns.suns)
        inview = self.scene.in_view(sunuv)
        suni = translate.uv2ij(sunuv[inview], 1024).T
        s = np.s_[..., suni[0], suni[1]]
        isviz = vis[s] > self.suns.srct/2
        suns = np.zeros((*isviz.shape, 8, 8))
        suns[isviz, :, :] = 1
        self._viz = np.sum(isviz)
        self.weights = suns

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
        executable: str, optional
            rendering engine binary

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        fdr = self.scene.outdir
        octr = f"{fdr}/sun.oct"
        rc = f"rtrace -fff {rcopts} -h -n {nproc} {octr}"
        outf = f'{self.scene.outdir}/{self.stype}_vals.out'
        lum = io.call_sampler(outf, rc, vecs)
        return lum

    def _uv2xyz(self, uv, si):
        return self.samplemap.uv2xyz(uv, si[2])

    def draw(self):
        """draw samples based on detail calculated from weights
        detail is calculated across direction only as it is the most precise
        dimension

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        """
        if self.idx > 0:
            p = wavelet.get_detail(self.weights, (3, 4))
        else:
            p = self.weights.ravel()

        sq = int(np.sqrt(self.suns.suns.shape[0]))
        ss = [np.s_[i:i + sq] for i in range(0, sq*sq, sq)]
        side = self.levels[self.idx][-1]
        for i in range(self.weights.shape[1]):
            a = p.reshape(self.weights.shape)[0][i][0:sq*sq]
            b = self.weights[0][i][0:sq*sq]
            im = np.hstack([b[s].reshape(side*sq, side) for s in ss]).reshape(
                side*sq, side*sq).T
            io.imshow(im, [10, 10])

        nsampc = int(self._sample_rate*self._viz*self.levels[self.idx, 2]**2)
        nsampc = max(min(nsampc, np.sum(p > 0.0001)), 2)
        # draw on pdf
        pdraws = np.random.default_rng().choice(p.size, nsampc, replace=False,
                                                p=p/np.sum(p))
        return pdraws
