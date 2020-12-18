# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import sys

import numpy as np

from raytraverse import io, draw, translate
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

    detailfunc = 'wavelet'

    filters = {'prewitt': (np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])/3,
                           np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])/3),
               'sobel': (np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/4,
                         np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/3),
               'sobelswap': (np.array([[1, 2, -1], [0, 0, 0], [1, -2, -1]])/4,
                             np.array([[1, 0, 1], [-2, 0, 2], [-1, 0, -1]])/4),
               'cross': (np.array([[1, 0], [0, -1]])/2,
                         np.array([[0, 1], [-1, 0]])/2),
               'point': (np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/3,
                         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])),
               'wav': (np.array([[-1, 0, 0], [-1, 4, -1], [0, 0, -1]])/3,
                       np.array([[0, 0, -1], [-1, 4, -1], [-1, 0, 0]])/3),
               }

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
        pres = self.area.ptshape
        if self.idx == 0 and np.var(self.weights) < 1e-9:
            pdraws = np.arange(np.prod(dres)*np.prod(pres))
        else:
            # direction detail
            if self.detailfunc == 'wavelet':
                daxes = (len(pres) + len(dres) - 2, len(pres) + len(dres) - 1)
                p = draw.get_detail(self.weights, daxes)
            else:
                p = draw.get_detail_filter(self.weights,
                                           *self.filters[self.detailfunc])
            if self.plotp:
                self._plot_p(p, fisheye=True)
            # draw on pdf
            pdraws = draw.from_pdf(p, self.threshold(self.idx),
                                   lb=self.lb, ub=self.ub)
        return pdraws


class DeterministicImageSampler(ImageSampler):
    def _offset(self, shape):
        """for modifying jitter behavior of UV direction samples"""
        return 0.5/self.levels[self.idx][-1]
