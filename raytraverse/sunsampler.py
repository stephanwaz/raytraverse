# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import shlex
from subprocess import Popen, PIPE

import numpy as np

from clasp import script_tools as cst
from scipy.ndimage import maximum_filter

from raytraverse import optic, io, wavelet, Sampler, translate

from memory_profiler import profile


class SunSampler(Sampler):
    """sample contributions from direct suns.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    maxspec: float
        maximum expected value of non reflected and/or scattered light
    """

    def __init__(self, scene, suns, maxspec=.3, idres=10,
                 dndepth=12, maxrate=.01, minrate=.005, wpow=.5,
                 **kwargs):
        self.suns = suns
        super(SunSampler, self).__init__(scene, stype='sunreflect',
                                         dndepth=dndepth,
                                         t0=0, t1=0, minrate=minrate,
                                         maxrate=maxrate, idres=idres, **kwargs)
        self.srcn = self.suns.suns.shape[0]
        self.wpow = wpow
        self.init_weights(maxspec)

    def init_weights(self, maxspec):
        shape = np.concatenate((self.scene.ptshape, self.levels[self.idx]))
        skypdf = f"{self.scene.outdir}/sky_pdf.npy"
        try:
            skypdf = np.load(skypdf)
        except FileNotFoundError:
            print('Warning! sunsampler initialized without vector weights')
        else:
            skypdf = np.where(skypdf > maxspec, 0, np.power(skypdf, self.wpow))
            self.weights = translate.resample(skypdf, shape)

    def sample(self, vecs, rcopts='-ab 7 -ad 60000 -as 30000 -lw 1e-7',
               nproc=12, executable='rcontrib'):
        """call rendering engine to sample sky contribution

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

    # @profile
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
            p = translate.gaussian_filter(self.weights, 1).ravel()
        else:
            p = self.weights.ravel()

        # io.imshow(self.weights[0,0], [10, 10])
        io.imshow(p.reshape(self.weights.shape)[0,0])
        print(np.sum(p > .01))
        nsampc = int(self._sample_rate*self.weights.size)
        # draw on pdf
        pdraws = np.random.default_rng().choice(p.size, nsampc, replace=False,
                                                p=p/np.sum(p))
        return pdraws

    def update_pdf(self, si, lum):
        """update self.weights (which holds values used to calculate pdf)

        Parameters
        ----------
        si: np.array
            multidimensional indices to update
        lum:
            values to update with

        """
        self.weights[tuple(si)] = np.power(np.max(lum, 1), self.wpow)
