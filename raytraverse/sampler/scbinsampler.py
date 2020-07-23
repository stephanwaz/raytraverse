# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os

import numpy as np

from raytraverse import translate, wavelet
from raytraverse.sampler.sampler import Sampler


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

    def __init__(self, scene, accuracy=.01, **kwargs):
        f = open(f'{scene.outdir}/scbins.cal', 'w')
        f.write(translate.scbinscal)
        f.close()
        skydeg = ("void glow skyglow 0 0 4 1 1 1 0 skyglow source sky 0 0 4"
                  " 0 0 1 180")
        anorm = accuracy*np.pi
        super().__init__(scene, srcn=scene.skyres**2, stype='sky', srcdef=skydeg,
                         accuracy=anorm, **kwargs)

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
        rc = (f"{executable} -V+ -fff {rcopts} -h -n {nproc} -e "
              f"'side:{self.scene.skyres}' -f "
              f"{self.scene.outdir}/scbins.cal -b bin -bn {self.srcn} "
              f"-m skyglow {self.compiledscene}")
        lum = super().sample(vecs, call=rc)
        return np.max(lum.reshape(-1, self.srcn), 1)

    def run_callback(self):
        outf = f'{self.scene.outdir}/{self.stype}_vis'
        np.save(outf, self.weights)
        outf = f'{self.scene.outdir}/{self.stype}_scheme'
        np.save(outf, self.get_scheme())
