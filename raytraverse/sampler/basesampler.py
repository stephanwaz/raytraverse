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


filterdict = {
              'wav': (np.array([[-1, 2, -1]])/2, np.array([[-1], [2], [-1]])/2,
                      np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]])/2),
              'haar': (np.array([[1, -1]])/2, np.array([[1], [-1]])/2,
                       np.array([[1, 0], [0, -1]])/2)
              }


class BaseSampler(object):
    """wavelet based sampling class
    this is a virutal class that holds the shared sampling methods across
    directional, area, and sunposition samplers. subclasses are named as:
    {Source}Sampler{SamplingRange}, for instance:

        - SamplerPt: virtual base class for sampling directions from a point
            - SkySamplerPt: sampling directions from a point with a sky patch
              source.
            - SunSamplerPt: sampling directions from a point with a single sun
              source
            - SunSamplerPtView: sampling the view from a point of the sun
            - ImageSampler: (re)sampling a fisheye image, useful for testing
        - SamplerArea: sampling points on a horizontal planar area with any
          source type
        - SamplerSuns: sampling sun positions (with nested area sampler)

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
    #: this value times the accuracy parameter is passed to
    #: raytraverse.sampler.draw.from_pdf() at level 0 (usually not used)
    t0 = 2**-8
    #: final sampling threshold coefficient
    #: this value times the accuracy parameter is passed to
    #: raytraverse.sampler.draw.from_pdf() at final level, intermediate
    #: sampling levels are thresholded by a linearly interpolated between t0
    #: and t1
    t1 = .0625

    #: lower bound for drawing from pdf
    #: passed to raytraverse.sampler.draw.from_pdf()
    lb = .25
    #: upper bound for drawing from pdf
    #: passed to raytraverse.sampler.draw.from_pdf()
    ub = 8

    _includeorigin = False

    def __init__(self, scene, engine, accuracy=1.0, stype='generic',
                 samplerlevel=0):
        self.engine = engine
        #: raytraverse.scene.Scene: scene information
        self.scene = scene
        #: float: accuracy parameter
        #: some subclassed samplers may apply a scale factor to normalize
        #: threshold values depending on source brightness (see for instance
        #: ImageSampler and SunSamplerPt)
        self.accuracy = accuracy
        #: str: sampler type
        self.stype = stype
        self._levels = None
        #: np.array: holds weights for self.draw
        self.weights = np.empty(0)
        self.features = 1
        self.vecs = None
        self.lum = []
        self._slevel = samplerlevel

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

    def run(self, mapper, name, levels, plotp=False, log='err', pfish=True,
            **kwargs):
        """trigger a sampling run. subclasses should return a
        LightPoint/LightField from the executed object state (first call this
        method with super().run(...)

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
        detaillog = self._slevel == 0
        logerr = False
        if log == 'scene':
            logerr = False
        elif log == 'err':
            logerr = True
        else:
            detaillog = False
        if detaillog:
            self.scene.log(self,
                           f"Started sampling {self.scene.outdir} at {name} "
                           f"with {self.stype}", logerr, level=self._slevel)
            hdr = ['level ', '      shape', 'samples', '   rate']
            self.scene.log(self, '\t'.join(hdr), logerr, level=self._slevel)
        allc = 0
        leveliter = self._init4run(levels, plotp=plotp, pfish=pfish)
        for i in leveliter:
            if hasattr(leveliter, "set_description"):
                leveliter.set_description(f"Level {i+1} of {len(self.levels)}")
            allc += self._run_level(mapper, name, i, plotp, detaillog, logerr,
                                    pfish)
        srate = (allc * self.features /
                 np.prod(self._wshape(self.levels.shape[0] - 1)))
        if detaillog:
            row = ['total sampling:', '- ', f"{allc: >7}", f"{srate: >7.02%}"]
            self.scene.log(self, '\t'.join(row), logerr, level=self._slevel)

    def _init4run(self, levels, **kwargs):
        """(re)initialize object for new run, ensuring properties are cleared
        prior to executing sampling loop"""
        self.vecs = None
        self.lum = []
        self._levels = levels
        # reset weights
        self.weights = np.full(self._wshape(0), 1, dtype=np.float32)
        leveliter = range(self.levels.shape[0])
        return leveliter

    def _run_level(self, mapper, name, i, plotp=False, detaillog=True,
                   logerr=False, pfish=True):
        """the main execution at a sampling level"""
        shape = self.levels[i]
        self._lift_weights(i)
        draws, p = self.draw(i)
        si, uv = self.sample_to_uv(draws, shape)
        if si.size > 0:
            vecs = mapper.uv2xyz(uv, stackorigin=self._includeorigin)
            srate = si.shape[1]/np.prod(shape)
            if detaillog:
                row = (f"{i + 1} of {self.levels.shape[0]}\t"
                       f"{str(shape): >11}\t{si.shape[1]: >7}\t"
                       f"{srate: >7.02%}")
                self.scene.log(self, row, logerr, level=self._slevel)
            if plotp:
                self._plot_p(p, i, mapper, name, fisheye=pfish)
                self._plot_vecs(vecs, i, mapper, name, fisheye=pfish)
            lum = self.sample(vecs)
            self._update_weights(si, lum)
            if plotp:
                self._plot_weights(i, mapper, name, fisheye=pfish)
            a = lum.shape[0]
        else:
            a = 0
        return a

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
                p = draw.get_detail(self.weights, *filterdict[self.detailfunc])
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
        self._dump_vecs(vecs)
        lum = self.engine.run(np.copy(vecs, 'C')).ravel()
        self.lum = np.concatenate((self.lum, lum))
        return lum

    #: filter banks for calculating detail choices:
    #:
    #: 'haar': [[1 -1]]/2, [[1] [-1]]/2, [[1, 0] [0, -1]]/2
    #:
    #: 'wav': [[-1 2 -1]] / 2, [[-1] [2] [-1]] / 2,
    #: [[-1 0 0] [0 2 0] [0 0 -1]] / 2
    detailfunc = 'wav'

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
