# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
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
    engine: raytraverse.renderer.Rcontrib
        initialized rendering instance
    """

    def __init__(self, scene, engine, **kwargs):
        super().__init__(scene, engine, srcn=engine.srcn, stype='sky',
                         **kwargs)

    def sample(self, vecs):
        """call rendering engine to sample rays

        Parameters
        ----------
        vecs: np.array
            sample vectors (subclasses can choose which to use)

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        lum = self.engine(vecs)
        if len(self.lum) == 0:
            self.lum = lum
        else:
            self.lum = np.concatenate((self.lum, lum), 0)
        return np.max(lum, 1)
