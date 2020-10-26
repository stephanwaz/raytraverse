# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os

import numpy as np

from raytraverse import translate, renderer
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

    def __init__(self, scene, accuracy=1,
                 rcopts='-ab 7 -ad 60000 -as 30000 -lw 1e-7', **kwargs):
        skydeg = ("void glow skyglow 0 0 4 1 1 1 0 skyglow source sky 0 0 4"
                  " 0 0 1 180\nskyglow source ground  0 0 4 0 0 -1 180")
        mods = "-m skyglow"
        self.engine = renderer.Rcontrib()
        self.engine.reset()
        srcn = scene.skyres**2 + 1
        engine_args = (f"-V+ {rcopts} -Z+ -e 'side:{scene.skyres}' -f "
                       f"scbins.cal -b bin -bn {srcn} {mods}")
        super().__init__(scene, srcn=srcn, stype='sky',  srcdef=skydeg,
                         accuracy=accuracy, engine_args=engine_args, **kwargs)

    def sample(self, vecf):
        """call rendering engine to sample sky contribution

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
        lum = super().sample(vecf)
        return np.max(lum[:, :self.srcn-1], 1)
