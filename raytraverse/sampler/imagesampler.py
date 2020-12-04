# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
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

    def __init__(self, scene, **kwargs):
        super().__init__(scene, stype="image", engine=ImageRenderer,  **kwargs)
        self.accuracy *= np.average(self.engine.scene[self.engine.scene > 0])
        self.t0 = .5
        self.t1 = 4

    def sample(self, vecf, vecs):
        """sample an ImageRenderer"""
        lum = self.engine.call(vecs)
        outf = f'{self.scene.outdir}/{self.stype}_vals.out'
        f = open(outf, 'a+b')
        f.write(io.np2bytes(lum))
        f.close()
        return lum.ravel()

    detailfunc = 'prewitt'

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
        pres = self.scene.area.ptshape
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
            # a cooling parameter towards deterministic sampling at final level
            bound = self._linear(self.idx, .5, 0)
            # bound = 0
            # draw on pdf
            pdraws = draw.from_pdf(p, self.threshold(self.idx),
                                   lb=1 - bound, ub=1 + bound)
        return pdraws


class DeterministicImageSampler(ImageSampler):

    r1 = False

    def _offset(self, shape):
        """for modifying jitter behavior of UV direction samples"""
        # return 0.5/self.levels[self.idx][-1]
        return 0.5/self.levels[self.idx][-1]

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
        pres = self.scene.area.ptshape
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
            # a cooling parameter towards deterministic sampling at final level
            bound = 0
            if self.r1:
                bound = self._linear(self.idx, .5, 0)
            # draw on pdf
            pdraws = draw.from_pdf(p, self.threshold(self.idx),
                                   lb=.125, ub=8)
        return pdraws
