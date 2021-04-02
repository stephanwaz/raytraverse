# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.sampler import draw
from raytraverse.sampler.basesampler import BaseSampler
from raytraverse.evaluate import BaseMetricSet


class SamplerArea(BaseSampler):
    """wavelet based sampling class

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry and formatter compatible with engine
    engine: raytraverse.sampler.SamplerPt
        should inherit from raytraverse.renderer.Renderer
    accuracy: float, optional
        parameter to set threshold at sampling level relative to final level
        threshold (smaller number will increase sampling, default is 1.0)
    nlev: int, optional
        number of levels to sample
    jitter: bool, optional
        jitter samples
    """

    def __init__(self, scene, engine, accuracy=1.0, nlev=3, jitter=True,
                 metricclass=BaseMetricSet,
                 metricset=('avglum', 'density', 'gcr')):
        super().__init__(scene, engine, accuracy, engine.stype)
        self.nlev = nlev
        self.jitter = jitter
        #: raytraverse.mapper.PlanMapper
        self._mapper = None
        self.slices = []
        self.metricclass = metricclass
        self.metricset = metricset
        self.features = len(metricset)

    def sampling_scheme(self, mapper):
        """calculate sampling scheme"""
        return np.array([mapper.shape(i) for i in range(self.nlev)])

    def run(self, mapper, **kwargs):
        """adapively sample an areaa defined by mapper

        Parameters
        ----------
        mapper: raytraverse.mapper.PlanMapper
            the pointset to build/run if initialized with points runs a static
            sampler
        kwargs:
            passed to self.run()

        Returns
        -------
        raytraverse.lightlplane.LightPlaneKD
        """
        name = mapper.name
        self._mapper = mapper
        levels = self.sampling_scheme(mapper)
        super().run(mapper, name, levels, **kwargs)

    def draw(self, level):
        """draw samples based on detail calculated from weights
        detail is calculated across direction only as it is the most precise
        dimension

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
            p = np.ones(self.weights.shape[1:])
        else:
            # use weights directly on first pass
            if level == 0:
                p = np.sum(self.weights, axis=0).ravel()
            else:
                p = draw.get_detail(self.weights,
                                    *self.filters[self.detailfunc])
                p = np.sum(p.reshape(self.weights.shape), axis=0).ravel()
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
            return [], []
        # index assignment
        si = np.stack(np.unravel_index(pdraws, shape))
        if self.jitter:
            offset = np.random.default_rng().random(si.shape).T
        else:
            offset = 0.5
        # convert to UV directions and positions
        uv = (si.T + offset)/np.asarray(shape)
        valid = self._mapper.in_view_uv(uv, False)
        return si[:, valid], uv[valid]

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
        idx = self.slices[-1].indices(self.slices[-1].stop)
        lums = []
        for posidx, point in zip(range(*idx), vecs):
            lp = self.engine.run(point, posidx)
            vol = lp.get_applied_rays(1)
            metric = self.metricclass(*vol, lp.vm,  metricset=self.metricset)
            lums.append(metric())
        return np.array(lums)

    def update_weights(self, si, lum):
        """update self.weights (which holds values used to calculate pdf)

        Parameters
        ----------
        si: np.array
            multidimensional indices to update
        lum:
            values to update with

        """
        wv = np.moveaxis(self.weights, 0, 2)
        wv[tuple(si)] = lum

    def _dump_vecs(self, vecs):
        if self.vecs is None:
            self.vecs = vecs
            v0 = 0
        else:
            self.vecs = np.concatenate((self.vecs, vecs))
            v0 = self.slices[-1].stop
        self.slices.append(slice(v0, v0 + len(vecs)))

    def _wshape(self, level):
        return np.concatenate(([self.features], self.levels[level]))

    def _plot_p(self, p, level, vm, name, suffix=".hdr", **kwargs):
        pass

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr", **kwargs):
        pass
