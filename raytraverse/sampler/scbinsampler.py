# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np

from raytraverse import renderer
from raytraverse.sampler.sampler import Sampler


class SCBinSampler(Sampler):
    """sample contributions from the sky hemisphere according to a square grid
    transformed by shirley-chiu mapping using rcontrib.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    """

    def __init__(self, scene, engine=renderer.Rcontrib,
                 rcopts='-ab 7 -ad 60000 -as 30000 -lw 1e-7', **kwargs):
        skydeg = scene.formatter.get_skydef((1, 1, 1), ground=True,
                                            name='skyglow')
        engine_args, srcn = scene.formatter.get_contribution_args(rcopts,
                                                                  scene.skyres,
                                                                  'skyglow')
        engine().reset()
        super().__init__(scene, srcn=srcn, stype='sky',  srcdef=skydeg,
                         engine=engine, engine_args=engine_args, **kwargs)

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
