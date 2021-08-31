# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.sampler.samplerpt import SamplerPt
from raytraverse.lightpoint import LightPointKD
from raytraverse import translate


class SkySamplerPt(SamplerPt):
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
        self._dump_vecs(vecs)
        lum = self.engine.run(np.copy(vecs, 'C'))
        if len(self.lum) == 0:
            self.lum = lum
        else:
            self.lum = np.concatenate((self.lum, lum), 0)
        return np.max(lum, 1)

    def _run_callback(self, point, posidx, vm, write=True, **kwargs):
        """include sky patch source dirs"""
        srcdirs = translate.skybin2xyz(np.arange(self.srcn), self.engine.side)
        lightpoint = LightPointKD(self.scene, self.vecs, self.lum, vm=vm,
                                  src=self.stype, srcdir=srcdirs, pt=point,
                                  write=write, srcn=self.srcn, posidx=posidx,
                                  **kwargs)
        return lightpoint
