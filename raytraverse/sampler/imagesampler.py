# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse import io
from raytraverse.lightpoint import LightPointKD
from raytraverse.sampler.sampler import Sampler
from raytraverse.renderer import ImageRenderer


class ImageSampler(Sampler):
    """sample image (for testing algorithms).

    Parameters
    ----------
    scene: raytraverse.scene.ImageScene
        scene class containing image file information
    scalefac: float, optional
        by default set to the average of non-zero pixels in the image used to
        establish sampling thresholds similar to contribution based samplers
    """

    def __init__(self, scene, scalefac=None, **kwargs):
        super().__init__(scene, stype="image", engine=ImageRenderer,  **kwargs)
        if scalefac is None:
            img = io.hdr2array(scene.scene)
            scalefac = np.average(img[img > 0])
        self.accuracy *= scalefac
        self.vecs = None
        self.lum = []

    def sample(self, vecf, vecs, outf=None):
        """sample an ImageRenderer"""
        lum = self.engine.call(vecs).ravel()
        self.lum = np.concatenate((self.lum, lum))
        return lum

    def _dump_vecs(self, vecs, vecf):
        if self.vecs is None:
            self.vecs = vecs
        else:
            self.vecs = np.concatenate((self.vecs, vecs))

    def run_callback(self, vecfs, name, point, posidx, vm):
        return LightPointKD(self.scene, self.vecs, self.lum, vm=vm, pt=point,
                            posidx=posidx, src=self.stype, write=False)


class DeterministicImageSampler(ImageSampler):

    ub = 1

    def _offset(self, shape, dim):
        """for modifying jitter behavior of UV direction samples"""
        return 0.5/dim
