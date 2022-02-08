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
from clasp.script_tools import try_mkdir

from raytraverse import io
from raytraverse.sampler import draw
from raytraverse.sampler.basesampler import BaseSampler, filterdict
from raytraverse.evaluate import SamplingMetrics
from raytraverse.lightfield import LightPlaneKD


class SamplerArea(BaseSampler):
    """wavelet based area sampling class

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry and formatter compatible with engine
    engine: raytraverse.sampler.SamplerPt
        point sampler
    accuracy: float, optional
        parameter to set threshold at sampling level relative to final level
        threshold (smaller number will increase sampling, default is 1.0)
    nlev: int, optional
        number of levels to sample
    jitter: bool, optional
        jitter samples
    edgemode: {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
        default: 'constant', if 'constant' value is set to -self.t1, so edge is
        always seen as detail. Internal edges (resulting from PlanMapper
        borders) will behave like 'nearest' for all options except 'constant'
    metricclass: raytraverse.evaluate.BaseMetricSet, optional
        the metric calculator used to compute weights
    metricset: iterable, optional
        list of metrics (must be recognized by metricclass. metrics containing
        "lum" will be normalized to 0-1)
    metricfunc: func, optional
        takes detail array as an argument, shape: (len(metricset),N, M) and an
        axis=0 keyword argument, returns shape (N, M). could be np.max, np.sum
        np.average or us custom function following the same pattern.
    """

    #: initial sampling threshold coefficient
    t0 = .1
    #: final sampling threshold coefficient
    t1 = .9
    #: upper bound for drawing from pdf
    ub = 100

    def __init__(self, scene, engine, accuracy=1.0, nlev=3, jitter=True,
                 edgemode='constant', metricclass=SamplingMetrics,
                 metricset=('avglum', 'loggcr', 'xpeak', 'ypeak'),
                 metricfunc=np.max, **kwargs):
        super().__init__(scene, engine, accuracy, engine.stype, **kwargs)
        self.engine._slevel = self._slevel + 1
        self.nlev = nlev
        self.jitter = jitter
        #: raytraverse.mapper.PlanMapper
        self._mapper = None
        self._name = None
        modes = ('reflect', 'constant', 'nearest', 'mirror', 'wrap')
        if edgemode not in modes:
            raise ValueError(f"edgemode={edgemode} not a valid option"
                             " must be one of: {modes}")
        self._specguide = None
        self._edgemode = edgemode
        self._mask = slice(None)
        self._candidates = None
        self._plotpchild = False
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

    def run(self, mapper, name=None, specguide=None, plotp=False, **kwargs):
        """adapively sample an area defined by mapper

        Parameters
        ----------
        mapper: raytraverse.mapper.PlanMapper
            the pointset to build/run
        name: str, optional
        specguide: raytraverse.lightfield.LightPlaneKD
            sky source lightfield to use as specular guide for sampling (used by
            engine of type raytraverse.sampler.SunSamplerPt)
        kwargs:
            passed to self.run()

        Returns
        -------
        raytraverse.lightlplane.LightPlaneKD
        """
        if name is None:
            name = mapper.name
        try_mkdir(f"{self.scene.outdir}/{name}")
        self._mapper = mapper
        self._name = name
        self._specguide = specguide
        levels = self.sampling_scheme(mapper)
        plotpthis = plotp and len(levels) > 1
        self._plotpchild = plotp and not plotpthis
        super().run(mapper, name, levels, plotp=plotpthis, **kwargs)
        return LightPlaneKD(self.scene, self.vecs, self._mapper, self.stype)

    def repeat(self, guide, stype):
        """repeat the sampling of a guide LightPlane (to match all rays)

        Parameters
        ----------
        guide: LightPlaneKD
        stype: str, optional
            alternate stype name for samplerpt. raises a ValueError if it
            matches the guide.
        Returns
        -------

        """
        self._mapper = guide.pm
        self._name = self._mapper.name
        if stype == guide.src:
            raise ValueError("stype cannot match guide.src, as it would "
                             "overwrite data")
        ostype = self.stype
        self.stype = stype

        self.vecs = None
        self.lum = []

        gvecs = guide.vecs
        self._dump_vecs(gvecs)
        pbar = self.scene.progress_bar(self, list(enumerate(gvecs)),
                                       level=self._slevel, message="resampling")
        for posidx, point in pbar:
            gpt = guide.data[posidx]
            self.engine.repeat(gpt, stype)
        lp = LightPlaneKD(self.scene, self.vecs, self._mapper, self.stype)
        self.stype = ostype
        return lp

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
            pdraws = np.arange(int(np.prod(dres)))[self._mask]
            p = np.ones(dres).ravel()
            p[np.logical_not(self._mask)] = 0
        else:
            nweights = self._normed_weights(mask=self._edgemode == 'constant')
            p = draw.get_detail(nweights, *filterdict[self.detailfunc],
                                mode=self._edgemode, cval=-self.t1)
            p = self._metricfunc(p.reshape(self.weights.shape), axis=0)
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
        self._dump_vecs(vecs)
        idx = self.slices[-1].indices(self.slices[-1].stop)
        lums = []
        lpargs = dict(parent=self._name)
        kwargs = dict(metricset=self.metricset, lmin=1e-8)
        specguide = None
        sgpt = None
        if hasattr(self.engine, "slimit"):
            kwargs.update(peakthreshold=self.engine.slimit)
            specguide = self._specguide
        level_desc = f"Level {len(self.slices)} of {len(self.levels)}"
        if self.nlev == 1 and len(vecs) == 1:
            logpoint = 'err'
        else:
            logpoint = None
        if logpoint is not None:
            self.engine._slevel -= 1
            pbar = list(zip(range(*idx), vecs))
        else:
            pbar = self.scene.progress_bar(self, list(zip(range(*idx), vecs)),
                                           level=self._slevel, message=level_desc)
        for posidx, point in pbar:
            if specguide is not None:
                sgpt = self._load_specguide(point)
            lp = self.engine.run(point, posidx, specguide=sgpt, lpargs=lpargs,
                                 log=logpoint, plotp=self._plotpchild)
            vol = lp.evaluate(1)
            metric = self.metricclass(*vol, lp.vm,  **kwargs)
            lums.append(metric())
        if logpoint is not None:
            self.engine._slevel += 1
        if len(self.lum) == 0:
            self.lum = np.array(lums)
        else:
            self.lum = np.concatenate((self.lum, lums), 0)
        return np.array(lums)

    def _load_specguide(self, point):
        """find the 3 nearest lightpoints in the specular sampling guide"""
        if self._specguide is not None:
            d, i = self._specguide.kd.query(point, 3)
            idxs = i[d <= self._mapper.ptres * np.sqrt(2)]
            specguide = [self._specguide.data[j] for j in idxs]
            if len(specguide) > 0:
                return specguide
        return None

    def _update_weights(self, si, lum):
        """only used by _plot_weights, weights are recomputed from spatial
        query"""
        wv = np.moveaxis(self.weights, 0, 2)
        wv[tuple(si)] = lum

    def _lift_weights(self, i):
        """because areas can have an arbitrary border, upsamppling is not
        effective. If we mask the weights, than the UV edges (which are padded)
        will be treated differently from the inside edges. by remapping each
        level with the nearest sample value and then masking after calculating
        the detail we avoid these issues."""
        self._candidates = self._mapper.point_grid_uv(jitter=self.jitter,
                                                      level=i, masked=False,
                                                      snap=self.nlev-1)
        self._mask = self._mapper.in_view_uv(self._candidates, False)
        if self.vecs is not None:
            wvecs = self._mapper.uv2xyz(self._candidates)
            d, idx = cKDTree(self.vecs).query(wvecs)
            weights = self.lum[idx].reshape(*self.levels[i], self.features)
            self.weights = np.moveaxis(weights, 2, 0)

    def _normed_weights(self, mask=False):
        normi = [i for i, j in enumerate(self.metricset) if 'lum' in j]
        nweights = np.copy(self.weights)
        for i in normi:
            nmin = np.min(self.weights[i])
            norm = np.max(self.weights[i]) - nmin
            if norm > 0:
                nweights[i] = (nweights[i] - nmin)/norm
        if mask:
            # when using a maskedplanmapper, we don't want to mask excluded
            # points before calculating detail
            mk = self._mapper.in_view_uv(self._candidates, False, usemask=False)
            for nw in nweights:
                nw.flat[np.logical_not(mk)] = -self.t1
        return nweights

    def _dump_vecs(self, vecs):
        if self.vecs is None:
            self.vecs = vecs
            v0 = 0
        else:
            self.vecs = np.concatenate((self.vecs, vecs))
            v0 = self.slices[-1].stop
        self.slices.append(slice(v0, v0 + len(vecs)))
        vfile = (f"{self.scene.outdir}/{self._name}/{self.stype}"
                 f"_points.tsv")
        idx = np.arange(len(self.vecs))[:, None]
        level = np.zeros_like(idx, dtype=int)
        for sl in self.slices[1:]:
            level[sl.start:] += 1
        idxvecs = np.hstack((level, idx, self.vecs))
        np.savetxt(vfile, idxvecs, ("%d", "%d", "%.4f", "%.4f", "%.4f"))

    def _wshape(self, level):
        return np.concatenate(([self.features], self.levels[level]))

    def _plot_dist(self, ps, outf):
        shp = ps.shape
        pixels = self._mapper.pixels(512)
        x = (np.arange(shp[0]) + .5)*pixels.shape[0]/shp[0]
        y = (np.arange(shp[1]) + .5)*pixels.shape[1]/shp[1]
        pinterp = RegularGridInterpolator((x, y), ps, bounds_error=False,
                                          method='nearest', fill_value=None)
        outpar = pinterp(pixels.reshape(-1, 2)).reshape(pixels.shape[:-1])
        io.array2hdr(outpar[-1::-1], outf)

    def _plot_p(self, p, level, vm, name, suffix=".hdr", **kwargs):
        outp = (f"{self.scene.outdir}_{name}_{self.stype}_detail_"
                f"{level:02d}{suffix}")
        ps = p.reshape(self.weights.shape[1:])
        self._plot_dist(ps, outp)

    def _plot_weights(self, level, vm, name, suffix=".hdr", **kwargs):
        normw = self._normed_weights()
        for i, w in zip(self.metricset, normw):
            outw = (f"{self.scene.outdir}_{name}_{self.stype}_weight_{i}_"
                    f"{level:02d}{suffix}")
            w.flat[np.logical_not(self._mask)] = 0
            self._plot_dist(w, outw)

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr", **kwargs):
        img = np.zeros((3, *self._mapper.framesize(512)))
        img = self._mapper.add_vecs_to_img(img, vecs, grow=2)
        outv = (f"{self.scene.outdir}_{name}_{self.stype}_samples_"
                f"{level:02d}{suffix}")
        io.carray2hdr(img, outv)
