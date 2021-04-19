# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate
from raytraverse.sampler import draw


filterdict = {'prewitt': (np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])/3,
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
              'wav3': (np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])/2,
                        np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]])/2,
                        np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]])/2),
              'haar': (np.array([[1, -1]])/2, np.array([[1], [-1]])/2,
                        np.array([[1, 0], [0, -1]])/2),
              }


class BaseSampler(object):
    """wavelet based sampling class

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry and formatter compatible with engine
    engine:
        has a run() method
    accuracy: float, optional
        parameter to set threshold at sampling level relative to final level
        threshold (smaller number will increase sampling, default is 1.0)
    stype: str, optional
        sampler type (prefixes output files)
    """

    #: initial sampling threshold coefficient
    t0 = 2**-8
    #: final sampling threshold coefficient
    t1 = .0625

    #: lower bound for drawing from pdf
    lb = .25
    #: upper bound for drawing from pdf
    ub = 8

    _includeorigin = False

    def __init__(self, scene, engine, accuracy=1.0, stype='generic'):
        self.engine = engine
        #: raytraverse.scene.Scene: scene information
        self.scene = scene
        #: float: accuracy parameter
        self.accuracy = accuracy
        #: str: sampler type
        self.stype = stype
        self._levels = None
        #: np.array: holds weights for self.draw
        self.weights = np.empty(0)
        self.features = 1
        self.vecs = None
        self.lum = []

    @property
    def levels(self):
        """sampling scheme

        :getter: Returns the sampling scheme
        :setter: Set the sampling scheme
        :type: np.array
        """
        return self._levels

    def sampling_scheme(self, *args):
        """calculate sampling scheme"""
        return np.arange(*args, dtype=int)

    def run(self, mapper, name, levels, plotp=False, log=False, pfish=True,
            **kwargs):
        """sample a single point, poisition index handles file naming

        Parameters
        ----------
        mapper: raytraverse.mapper.Mapper
            mapper to sample
        name: str
            output name
        levels: np.array
            the sampling scheme
        plotp: bool, optional
            plot weights, detail and vectors for each level
        log: str, optional
            whether to log level sampling rates
            can be 'scene', 'err' or None
            'scene' - logs to Scene log file
            'err' - logs to stderr
            anything else - does not log incremental progress
        pfish: bool, optional
            if True and plotp, use fisheye projection for detail/weight/vector
            images.
        kwargs:
            unused
        """
        self.vecs = None
        self.lum = []
        detaillog = True
        logerr = False
        if log == 'scene':
            logerr = False
        elif log == 'err':
            logerr = True
        else:
            detaillog = False
        allc = 0
        self.scene.log(self, f"Started sampling {self.scene.outdir} at {name} "
                             f"with {self.stype}", logerr)
        if detaillog:
            hdr = ['level ', '      shape', 'samples', '   rate']
            self.scene.log(self, '\t'.join(hdr), logerr)
        self._levels = levels
        # reset weights
        self.weights = np.full(self._wshape(0), 1, dtype=np.float32)
        for i in range(self.levels.shape[0]):
            shape = self.levels[i]
            self._lift_weights(i)
            draws, p = self.draw(i)
            si, uv = self.sample_to_uv(draws, shape)
            if si.size > 0:
                vecs = mapper.uv2xyz(uv, stackorigin=self._includeorigin)
                srate = si.shape[1]/np.prod(shape)
                self._dump_vecs(vecs)
                if detaillog:
                    row = (f"{i + 1} of {self.levels.shape[0]}\t"
                           f"{str(shape): >11}\t{si.shape[1]: >7}\t"
                           f"{srate: >7.02%}")
                    self.scene.log(self, row, logerr)
                if plotp:
                    self._plot_p(p, i, mapper, name, fisheye=pfish)
                    self._plot_vecs(vecs, i, mapper, name, fisheye=pfish)
                lum = self.sample(vecs)
                self._update_weights(si, lum)
                if plotp:
                    self._plot_weights(i, mapper, name, fisheye=pfish)
                a = lum.shape[0]
                allc += a
        srate = (allc * self.features /
                 np.prod(self._wshape(self.levels.shape[0] - 1)))
        row = ['total sampling:', '- ', f"{allc: >7}", f"{srate: >7.02%}"]
        self.scene.log(self, '\t'.join(row), logerr)

    def draw(self, level):
        """draw samples based on detail calculated from weights

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        p: np.array
            computed probabilities
        """
        dres = self.levels[level]
        # sample all if weights is not set or all even
        if level == 0 and np.var(self.weights) < 1e-9:
            pdraws = np.arange(int(np.prod(dres)))
            p = np.ones(self.weights.shape)
        else:
            # use weights directly on first pass
            if level == 0:
                p = self.weights.ravel()
            else:
                p = draw.get_detail(self.weights,
                                    *filterdict[self.detailfunc])
            # draw on pdf
            pdraws = draw.from_pdf(p, self._threshold(level),
                                   lb=self.lb, ub=self.ub)
        return pdraws, p

    def sample_to_uv(self, pdraws, shape):
        """generate samples vectors from flat draw indices

        Parameters
        ----------
        pdraws: np.array
            flat index positions of samples to generate
        shape: tuple
            shape of level samples

        Returns
        -------
        si: np.array
            index array of draws matching samps.shape
        vecs: np.array
            sample vectors
        """
        if len(pdraws) == 0:
            return np.empty(0), np.empty(0)
        # index assignment
        si = np.stack(np.unravel_index(pdraws, shape))
        # convert to UV directions and positions
        uv = si.T/shape[1]
        uv += self._offset(uv.shape, shape[1])
        return si, uv

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
        lum = self.engine.run(np.copy(vecs, 'C')).ravel()
        self.lum = np.concatenate((self.lum, lum))
        return lum

    #: filter banks for calculating detail choices:
    #:
    #: 'prewitt': [[1 1 1] [0 0 0] [-1 -1 -1]]/3, [[1 0 -1] [1 0 -1] [1 0 -1]]/3
    #:
    #: 'sobel': [[1 2 1] [0 0 0] [-1 -2 -1]]/4, [[1 0 -1] [2 0 -2] [1 0 -1]]/3
    #:
    #: 'sobelswap': [[1 2 -1] [0 0 0] [1 -2 -1]]/4,
    #: [[1 0 1] [-2 0 2] [-1 0 -1]]/4
    #:
    #: 'cross': [[1 0] [0 -1]]/2, [[0 1] [-1 0]]/2
    #:
    #: 'haar': [[1 -1]]/2, [[1] [-1]]/2, [[1, 0] [0, -1]]/2
    #:
    #: 'point': [[-1 -1 -1] [-1 8 -1] [-1 -1 -1]]/3
    #:
    #: 'wav': [[-1 0 0] [-1 4 -1] [0 0 -1]]/3, [[0 0 -1] [-1 4 -1] [-1 0 0]]/3
    #:
    #: 'wav3': [[0 0 0] [-1 2 -1] [0 0 0]] / 2, [[0 -1 0] [0 2 0] [0 -1 0]] / 2,
    #: [[-1 0 0] [0 2 0] [0 0 -1]] / 2
    detailfunc = 'wav3'

    def _update_weights(self, si, lum):
        """update self.weights (which holds values used to calculate pdf)

        Parameters
        ----------
        si: np.array
            multidimensional indices to update
        lum:
            values to update with

        """
        self.weights[tuple(si)] = lum

    def _lift_weights(self, level):
        self.weights = translate.resample(self.weights, self._wshape(level))

    def _wshape(self, level):
        return self.levels[level]

    def _threshold(self, idx):
        """threshold for determining sample count"""
        return self.accuracy * self._linear(idx, self.t0, self.t1)

    def _linear(self, x, x1, x2):
        if len(self.levels) <= 2:
            return (x1, x2)[x]
        else:
            return (x2 - x1)/(len(self.levels) - 1) * x + x1

    def _offset(self, shape, dim):
        """for modifying jitter behavior of UV direction samples

        Parameters
        ----------
        shape: tuple
            shape of samples to jitter/offset
        dim: int
            number of divisions in square side
        """
        return np.random.default_rng().random(shape)/dim

    def _dump_vecs(self, vecs):
        if self.vecs is None:
            self.vecs = vecs
        else:
            self.vecs = np.concatenate((self.vecs, vecs))

    def _plot_p(self, p, level, vm, name, suffix=".hdr", **kwargs):
        pass

    def _plot_weights(self, level, vm, name, suffix=".hdr", **kwargs):
        pass

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr", **kwargs):
        pass
