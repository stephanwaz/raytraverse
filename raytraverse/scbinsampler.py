# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os

import numpy as np

from clasp import script_tools as cst
from raytraverse import io, wavelet, Sampler
from raytraverse.sampler import scbinscal


class SCBinSampler(Sampler):
    """sample contributions from the sky hemisphere according to a square grid
    transformed by shirley-chiu mapping using rcontrib.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    srcn: int, optional
        side of square sky resolution
    """

    def __init__(self, scene, srcn=20, **kwargs):
        #: int: side of square sky resolution
        self.skres = srcn
        f = open(f'{scene.outdir}/scbins.cal', 'w')
        f.write(scbinscal)
        f.close()
        skydeg = ("void glow skyglow 0 0 4 1 1 1 0 skyglow source sky 0 0 4"
                  " 0 0 1 180")
        super().__init__(scene, srcn=srcn**2, stype='sky', srcdef=skydeg,
                         **kwargs)

    def __del__(self):
        super().__del__()
        try:
            os.remove(f'{self.scene.outdir}/scbins.cal')
        except (IOError, TypeError):
            pass

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
            array of shape (N,) to update weights
        """
        octr = f"{self.scene.outdir}/sky.oct"
        rc = (f"{executable} -V+ -fff {rcopts} -h -n {nproc} -e "
              f"'side:{self.skres}' -f "
              f"{self.scene.outdir}/scbins.cal -b bin -bn {self.srcn} "
              f"-m skyglow {octr}")
        outf = f'{self.scene.outdir}/{self.stype}_vals.out'
        lum = io.call_sampler(outf, rc, vecs)
        return np.max(lum.reshape(-1, self.srcn), 1)

    def draw(self):
        """draw samples based on detail calculated from weights
        detail is calculated across direction only as it is the most precise
        dimension

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        """
        dres = self.levels[self.idx]
        pres = self.scene.ptshape
        if self._sample_rate == 1:
            pdraws = np.arange(np.prod(dres) * np.prod(pres))
        else:
            # direction detail
            daxes = tuple(range(len(pres), len(pres) + len(dres)))
            p = wavelet.get_detail(self.weights, daxes)
            p = p*(1 - self._sample_t) + np.median(p)*self._sample_t
            # draw on pdf
            nsampc = int(self._sample_rate*self.weights.size)
            pdraws = np.random.default_rng().choice(p.size, nsampc,
                                                    replace=False,
                                                    p=p/np.sum(p))
        return pdraws

    def run_callback(self):
        outf = f'{self.scene.outdir}/{self.stype}_vis'
        np.save(outf, self.weights)
        outf = f'{self.scene.outdir}/{self.stype}_scheme'
        np.save(outf, self.get_scheme())
