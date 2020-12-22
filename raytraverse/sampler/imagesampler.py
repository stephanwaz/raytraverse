# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import sys

import numpy as np

from raytraverse import io
from raytraverse.sampler.sampler import Sampler
from raytraverse.renderer import ImageRenderer


class ImageSampler(Sampler):
    """sample image (for testing algorithms).

    Parameters
    ----------
    scene: raytraverse.scene.ImageScene
        scene class containing image file information
    """
    t0 = .5
    t1 = 8
    lb = .25
    ub = 8

    def __init__(self, scene, scalefac=None, **kwargs):
        super().__init__(scene, stype="image", engine=ImageRenderer,  **kwargs)
        if scalefac is None:
            scalefac = np.average(self.engine.scene[self.engine.scene > 0])
        self.accuracy *= scalefac

    def sample(self, vecf, vecs):
        """sample an ImageRenderer"""
        lum = self.engine.call(vecs)
        outf = f'{self.scene.outdir}/{self.stype}_vals.out'
        f = open(outf, 'a+b')
        f.write(io.np2bytes(lum))
        f.close()
        return lum.ravel()


class DeterministicImageSampler(ImageSampler):
    def _offset(self, shape):
        """for modifying jitter behavior of UV direction samples"""
        return 0.5/self.levels[self.idx][-1]
