# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from raytraverse import io
from raytraverse.sampler import draw
from raytraverse.sampler.basesampler import BaseSampler, filterdict
from raytraverse.sampler.sunsamplerpt import SunSamplerPt
from raytraverse.evaluate import SamplingMetrics
from raytraverse.lightfield import LightPlaneKD


class SamplerArea(BaseSampler):
    """wavelet based sun position sampling class

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry and formatter compatible with engine
    engine: raytraverse.renderer.Rtrace
        initialized renderer instance (with scene loaded, no sources)
    accuracy: float, optional
        parameter to set threshold at sampling level relative to final level
        threshold (smaller number will increase sampling, default is 1.0)
    nlev: int, optional
        number of levels to sample
    jitter: bool, optional
        jitter samples
    metricclass: raytraverse.evaluate.BaseMetricSet, optional
        the metric calculator used to compute weights
    metricset: iterable, optional
        list of metrics (must be recognized by metricclass. metrics containing
        "lum" will be normalized to 0-1)
    metricfunc: func, optional
        takes detail array as an argument, shape: (len(metricset),N, M) and an
        axis=0 keyword argument, returns shape (N, M). could be np.max, np.sum
        np.average or us custom function following the same pattern.
    ptkwargs: dict, optional
        kwargs for raytraveerse.sampler.SunSamplerPt initialization
    areakwargs: dict, optional
        kwargs for raytravrse.sampler.SamplerArea initialization
    """

    #: initial sampling threshold coefficient
    t0 = .1
    #: final sampling threshold coefficient
    t1 = .9
    #: upper bound for drawing from pdf
    ub = 100

    def __init__(self, scene, engine, accuracy=1.0, nlev=3, jitter=True,
                 metricclass=SamplingMetrics,
                 metricset=('avglum', 'loggcr'),
                 metricfunc=np.max, ptkwargs=None, areakwargs=None):
        super().__init__(scene, engine, accuracy, stype='sunpositions')
        if areakwargs is None:
            areakwargs = {}
        if ptkwargs is None:
            ptkwargs = {}
        self._areakwargs = areakwargs
        self._ptkwargs = ptkwargs
        self.nlev = nlev
        self.jitter = jitter
        #: raytraverse.mapper.SkyMapper
        self._skymapper = None
        self._areamapper = None
        self._mask = slice(None)
        self._candidates = None
        self.slices = []
        #: raytraverse.evaluate.BaseMetricSet
        self.metricclass = metricclass
        #: iterable
        self.metricset = metricset
        #: numpy func takes weights and axis=0 argument to reduce metric set
        self._metricfunc = metricfunc
        #: int:
        self.features = len(metricset)

    def sampling_scheme(self, mapper):
        """calculate sampling scheme"""
        return np.array([mapper.shape(i) for i in range(self.nlev)])

    def run(self, skymapper, areamapper, **kwargs):
        """adapively sample an area defined by mapper

        Parameters
        ----------
        skymapper: raytraverse.mapper.SkyMapper
            the mapping for drawing suns
        areamapper: raytraverse.mapper.PlanMapper
            the mapping for drawing points
        kwargs:
            passed to self.run()

        Returns
        -------
        raytraverse.lightlplane.LightPlaneKD
        """
        self._skymapper = skymapper
        self._areamapper = areamapper
        levels = self.sampling_scheme(skymapper)
        super().run(skymapper, "suns", levels, **kwargs)
        # return MultiSourcePlaneKD(self.scene, self.vecs, self._areamapper, self.stype)

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
        pdraws = np.arange(int(np.prod(dres)))[self._mask]
        p = np.ones(dres).ravel()
        p[np.logical_not(self._mask)] = 0
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
        si = np.stack(np.unravel_index(pdraws, shape))
        return si, self._candidates[pdraws]

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
        # idx = self.slices[-1].indices(self.slices[-1].stop)
        # lums = []
        # for suni, sunpos in zip(range(*idx), vecs):
        #     print(i)
        #     sunsamp = SunSamplerPt(self.scene, self.engine, sunpos, suni, **self._ptkwargs)
        #     sunsampler = SamplerArea(self.scene, sunsamp, **self._areakwargs)
        #     sunsampler.run(pm, log='err')
        #
        #
        #
        #
        # lpargs = dict(parent=self._mapper.name)
        # kwargs = dict(metricset=self.metricset, lmin=1e-8)
        # if hasattr(self.engine, "slimit"):
        #     kwargs.update(peakthreshold=self.engine.slimit)
        # for posidx, point in zip(range(*idx), vecs):
        #     lp = self.engine.run(point, posidx, lpargs=lpargs)
        #     vol = lp.get_applied_rays(1)
        #     metric = self.metricclass(*vol, lp.vm,  **kwargs)
        #     lums.append(metric())
        # if len(self.lum) == 0:
        #     self.lum = np.array(lums)
        # else:
        #     self.lum = np.concatenate((self.lum, lums), 0)
        return np.ones(len(vecs))

    def _update_weights(self, si, lum):
        pass

    def _lift_weights(self, i):
        self._candidates = self._skymapper.point_grid_uv(jitter=self.jitter, level=i, masked=False)
        self._mask = self._skymapper.in_view_uv(self._candidates, False)

    def _dump_vecs(self, vecs):
        if self.vecs is None:
            self.vecs = vecs
            v0 = 0
        else:
            self.vecs = np.concatenate((self.vecs, vecs))
            v0 = self.slices[-1].stop
        self.slices.append(slice(v0, v0 + len(vecs)))
        vfile = f"{self.scene.outdir}/{self._mapper.name}_{self.stype}.tsv"
        idxvecs = np.hstack((np.arange(len(self.vecs))[:, None], self.vecs))
        np.savetxt(vfile, idxvecs, ("%d", "%.4f", "%.4f", "%.4f"))

    def _wshape(self, level):
        return np.concatenate(([self.features], self.levels[level]))

    def _plot_p(self, p, level, vm, name, suffix=".hdr", **kwargs):
        pass

    def _plot_weights(self, level, vm, name, suffix=".hdr", **kwargs):
        pass

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr", **kwargs):
        img = np.zeros((3, *self._skymapper.framesize(512)))
        img = self._skymapper.add_vecs_to_img(img, vecs, grow=2)
        outv = (f"{self.scene.outdir}_{name}_{self.stype}_samples_"
                f"{level:02d}{suffix}")
        io.carray2hdr(img, outv)
