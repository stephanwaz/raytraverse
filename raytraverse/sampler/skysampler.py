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


class SkySampler(Sampler):
    """sample contributions from the sky hemisphere according to a square grid
    transformed by shirley-chiu mapping using rcontrib.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
        scene: str, optional (required if not reload)
        space separated list of radiance scene files (no sky) or octree
    skyres: float, optional
        approximate square patch size in degrees
    ropts: str, optional
        arguments for engine
    """

    def __init__(self, scene, engine=renderer.Rcontrib, skyres=10.0,
                 ropts='-ab 7 -ad 60000 -as 30000 -lw 1e-7', **kwargs):
        skydeg = scene.formatter.get_skydef((1, 1, 1), ground=True,
                                            name='skyglow')
        skyres = int(np.floor(90/skyres)*2)
        engine_args, srcn = scene.formatter.get_contribution_args(ropts,
                                                                  skyres,
                                                                  'skyglow')
        engine().reset()
        super().__init__(scene, engine=engine, srcn=srcn, stype='sky',
                         srcdef=skydeg, engine_args=engine_args, **kwargs)

    def sample(self, vecf, vecs):
        """call rendering engine to sample sky contribution"""
        lum = super().sample(vecf, vecs)
        return np.max(lum[:, :self.srcn-1], 1)
