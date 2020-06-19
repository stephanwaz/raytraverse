# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import shlex
from subprocess import Popen, PIPE

import numpy as np

from scipy.ndimage import maximum_filter

from raytraverse import optic, io, wavelet, Sampler, translate


class SunViewSampler(Sampler):
    """sample view rays to direct suns.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    """

    def __init__(self, scene, suns):
        self.ndsamps = 0
        self.suns = suns
        super(SunViewSampler, self).__init__(scene, stype='sunview', idres=4,
                                             dndepth=7, t0=0, t1=0, minrate=.25,
                                             maxrate=1)
        self.srcn = self.suns.suns.shape[0]
        self.init_weights()

    @property
    def levels(self):
        """sampling scheme

        :getter: Returns the sampling scheme
        :setter: Set the sampling scheme from (ptres, dndepth, skres)
        :type: np.array
        """
        return self._levels

    @levels.setter
    def levels(self, dndepth):
        """calculate sampling scheme"""
        self._levels = np.array([(self.suns.suns.shape[0], 2**i, 2**i)
                                 for i in range(self.idres, dndepth + 1, 1)])

    def init_weights(self):
        shape = np.concatenate((self.scene.ptshape,
                                (self.scene.view.aspect*1024, 1024)))
        skypdf = f"{self.scene.outdir}/sky_pdf.npy"
        try:
            skypdf = np.load(skypdf)
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
        self._wcoords = sunuv
        self.weights = suns
        self.ndsamps = np.sum(self.weights > 0)

    # def set_rates(self, x):
    #     self._sample_rate = self.sun_sample_rate(x)
    #
    # def sun_sample_rate(self, x):
    #     # return 1/(1 + (x > 0))
    #     return 1 + x*2.5

    # def print_sample_cnt(self):
    #     an = 0
    #     print("shape\tsamples")
    #     for i, l in enumerate(self.levels):
    #         x = i/(self.levels.shape[0] - 1)
    #         a = int(self.sun_sample_rate(x)*self.ndsamps)
    #         an += a
    #         print(f"{l}\t{a}")
    #     print(f"total\t{an} samples for {self.suns.suns.shape[0]} visible suns")

    def sample(self, vecs, rcopts='-ab 0',
               nproc=12, executable='rcontrib'):
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
            array of shape (N, binnumber) with sun coefficients
        """
        fdr = self.scene.outdir
        octr = f"{fdr}/sun.oct"
        rc = (f"{executable} -fff {rcopts} -h -n {nproc} "
              f"-M {fdr}/sun_modlist.txt {octr}")
        p = Popen(shlex.split(rc), stdout=PIPE,
                  stdin=PIPE).communicate(io.np2bytes(vecs))
        lum = optic.rgb2rad(io.bytes2np(p[0], (-1, 3)))
        return lum.reshape(-1, self.srcn)

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
        shape = np.concatenate((self.scene.ptshape, self.levels[self.idx]))
        # index assignment
        si = np.stack(np.unravel_index(pdraws, shape))
        uvsize = (shape[3] * 2**7)
        # convert to UV directions and positions (offset to each sun position)
        uv = self._wcoords[si[2]] + si.T[:, 3:]/uvsize - 4/1024
        pos = self.scene.area.uv2pt((si.T[:, 0:2] + .5)/shape[0:2])
        uv += (np.random.default_rng().uniform(0, 1/uvsize, uv.shape))
        xyz = self.scene.view.uv2xyz(uv)
        vecs = np.hstack((pos, xyz))
        return si, vecs

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

        ss = [np.s_[i:i + 10] for i in range(0, 100, 10)]
        side = self.levels[self.idx][-1]
        for i in range(self.weights.shape[1]):
            a = p.reshape(self.weights.shape)[0][i][0:100]
            b = self.weights[0][i][0:100]
            im = np.hstack([a[s].reshape(side*10, side) for s in ss]).reshape(
                side*10, side*10).T
            io.imshow(im, [10, 10])
            print(np.sum(b > .6)/(25*side*side))

        nsampc = int(self._sample_rate*self._viz*self.levels[self.idx, 2]**2)
        nsampc = min(nsampc, (np.sum(p > 0)))
        # draw on pdf
        pdraws = np.random.default_rng().choice(p.size, nsampc, replace=False,
                                                p=p/np.sum(p))
        return pdraws
