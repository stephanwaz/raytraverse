# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate
from raytraverse.sampler.sampler import Sampler


class SingleSunSampler(Sampler):
    """sample contributions from direct suns.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    """

    def __init__(self, scene, suns, sidx, sb, idres=10, fdres=12,
                 maxrate=.01, minrate=.005, **kwargs):
        self.sunpos = suns.suns[sidx]
        self.sidx = sidx
        dec, mod = suns.write_sun(sidx)
        super().__init__(scene, stype=f"sun_{sidx:04d}", fdres=fdres, t0=0,
                         t1=0, minrate=minrate, maxrate=maxrate,
                         idres=idres, srcdef=dec, **kwargs)
        self.init_weights(sb)

    def init_weights(self, sb):
        shape = np.concatenate((self.scene.ptshape, self.levels[self.idx]))
        fi = f"{self.scene.outdir}/sunpdfs/{self.sidx:04d}_{sb:04d}.npy"
        try:
            skypdf = np.load(fi)
        except FileNotFoundError:
            raise FileNotFoundError(f'cannot initialize without pdf, {fi} '
                                    'does not exist')
        else:
            self.weights = np.clip(translate.resample(skypdf, shape), 0, 1)

    def sample(self, vecs, rcopts='-ab 7 -ad 60000 -as 30000 -lw 1e-7',
               nproc=12):
        """call rendering engine to sample sky contribution

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
        return super().sample(vecs, call=rc)

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
        p = self.weights.ravel()
        nsampc = int(min(self._sample_rate*self.weights.size, np.sum(p > .001)))
        nsampc = max(nsampc, 2)
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
        self.weights[tuple(si)] += lum
