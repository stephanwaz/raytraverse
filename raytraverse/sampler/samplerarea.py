# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.ckdtree import cKDTree

from raytraverse import io
from raytraverse.sampler import draw
from raytraverse.sampler.basesampler import BaseSampler
from raytraverse.evaluate import SamplingMetrics


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

    #: initial sampling threshold coefficient
    t0 = .25
    #: final sampling threshold coefficient
    t1 = .25
    #: upper bound for drawing from pdf
    ub = 100

    def __init__(self, scene, engine, accuracy=1.0, nlev=3, jitter=True,
                 metricclass=SamplingMetrics,
                 metricset=('loggcr', 'xpeak', 'ypeak')):
        super().__init__(scene, engine, accuracy, engine.stype)
        self.nlev = nlev
        self.jitter = jitter
        #: raytraverse.mapper.PlanMapper
        self._mapper = None
        self._mask = slice(None)
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
        np.savetxt("points.txt", self.vecs)

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
            pdraws = np.arange(int(np.prod(dres)))[self._mask]
            p = np.ones(dres).ravel()
            p[np.logical_not(self._mask)] = 0
        else:
            p = draw.get_detail(self.weights, *self.filters[self.detailfunc])
            p = np.sum(p.reshape(self.weights.shape), axis=0)/self.features
            # zero out cells of previous samples
            if self.vecs is not None:
                pxy = self._mapper.ray2pixel(self.vecs, self.weights.shape[1:])
                x = self.weights.shape[1] - 1 - pxy[:, 0]
                y = pxy[:, 1]
                p[x, y] = 0
            # zero out oob
            p = p.ravel()
            p[np.logical_not(self._mask)] = 0
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
        return self._mapper.idx2uv(pdraws, shape, self.jitter)

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
        if len(self.lum) == 0:
            self.lum = np.array(lums)
        else:
            self.lum = np.concatenate((self.lum, lums), 0)
        return np.array(lums)

    def _update_weights(self, si, lum):
        """unused, weights are recomputed from spatial query"""
        pass

    def _lift_weights(self, i):
        wuv = self._mapper.point_grid_uv(jitter=False, level=i, masked=False)
        self._mask = self._mapper.in_view_uv(wuv, False)
        if self.vecs is not None:
            wvecs = self._mapper.uv2xyz(wuv)
            d, idx = cKDTree(self.vecs).query(wvecs)
            weights = self.lum[idx].reshape(*self.levels[i], self.features)
            self.weights = np.moveaxis(weights, 2, 0)

    def _normed_weights(self):
        nmin = np.amin(self.weights, (1, 2), keepdims=True)
        norm = np.amax(self.weights, (1, 2), keepdims=True) - nmin
        nmin[norm == 0] = 0
        norm[norm == 0] = 1
        return (self.weights - nmin)/norm

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
        shp = self.weights.shape[1:]
        ps = p.reshape(shp)
        pixels = self._mapper.pixels(512)
        x = (np.arange(shp[0]) + .5) * pixels.shape[0]/shp[0]
        y = (np.arange(shp[1]) + .5) * pixels.shape[1]/shp[1]
        pinterp = RegularGridInterpolator((x, y), ps, bounds_error=False,
                                          method='nearest', fill_value=None)
        outpar = pinterp(pixels.reshape(-1, 2)).reshape(pixels.shape[:-1])
        outp = (f"{self.scene.outdir}_{name}_{self.stype}_detail_"
                f"{level:02d}{suffix}")
        io.array2hdr(outpar[-1::-1], outp)
        for i, w in zip(self.metricset, self.weights):
            pinterp = RegularGridInterpolator((x, y), w, bounds_error=False,
                                              method='nearest', fill_value=None)
            outwar = pinterp(pixels.reshape(-1, 2)).reshape(pixels.shape[:-1])
            outw = (f"{self.scene.outdir}_{name}_{self.stype}_weight_{i}_"
                    f"{level:02d}{suffix}")
            io.array2hdr(outwar[-1::-1], outw)

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr", **kwargs):
        img = np.zeros((3, *self._mapper.framesize(512)))
        img = self._mapper.add_vecs_to_img(img, vecs, grow=2)
        outv = (f"{self.scene.outdir}_{name}_{self.stype}_samples_"
                f"{level:02d}{suffix}")
        io.carray2hdr(img, outv)
