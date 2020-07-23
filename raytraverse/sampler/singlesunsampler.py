# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate, quickplot
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

    def __init__(self, scene, suns, sidx, sb, speclevel=9, fdres=10,
                 accuracy=.01, **kwargs):
        #: np.array: sun position x,y,z
        self.sunpos = suns.suns[sidx]
        #: int: sun index of sunpos from associated SunSetter (for naming)
        self.sidx = sidx
        #: float: controls sampling limit in case of limited contribution
        self.slimit = suns.srct * .03
        dec, mod = suns.write_sun(sidx)
        anorm = accuracy * np.pi * (1 - np.cos(.533*np.pi/360))
        super().__init__(scene, stype=f"sun_{sidx:04d}", fdres=fdres,
                         accuracy=anorm, srcdef=dec, **kwargs)
        self.specidx = speclevel - self.idres
        self.specularpdf = sb

    @property
    def specularpdf(self):
        return self._specularpdf

    @specularpdf.setter
    def specularpdf(self, sb):
        shape = np.concatenate((self.scene.ptshape, self.levels[self.specidx]))
        fi = f"{self.scene.outdir}/sunpdfs/{self.sidx:04d}_{sb:04d}.npy"
        try:
            skypdf = np.load(fi)
        except FileNotFoundError:
            raise FileNotFoundError(f'cannot initialize without pdf, {fi} '
                                    'does not exist')
        else:
            print(skypdf.shape, shape)
            self._specularpdf = np.clip(translate.resample(skypdf, shape), 0, 1)

    def sample(self, vecs, rcopts='-ab 7 -ad 60000 -as 30000 -lw 1e-7',
               nproc=12, **kwargs):
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
        if self.idx == self.specidx:
            p = self.specularpdf.ravel()
            nsampc = max(np.sum(p > self.slimit), 2)
            if self.plotp:
                quickplot.imshow(p.reshape(self.weights.shape)[0, 0], [20, 10])
            pdraws = np.random.default_rng().choice(p.size, nsampc,
                                                    replace=False,
                                                    p=p/np.sum(p))
        else:
            pdraws = super().draw()
        return pdraws

