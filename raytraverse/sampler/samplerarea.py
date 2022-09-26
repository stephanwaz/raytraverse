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
    """

    #: upper bound for drawing from pdf
    ub = 100

    def __init__(self, scene, engine, accuracy=1.0, nlev=3, jitter=True,
                 edgemode='constant', metricclass=SamplingMetrics,
                 metricset=('avglum', 'loggcr', 'xpeak', 'ypeak'),
                 t0=.1, t1=.9, **kwargs):
        super().__init__(scene, engine, accuracy, engine.stype, t0=t0, t1=t1,
                         **kwargs)
        if "sun" in self.stype:
            self._gcrnorm = 8
        else:
            self._gcrnorm = 2
        self.engine._slevel = self._slevel + 1
        self.nlev = nlev
        self.jitter = jitter
        #: raytraverse.mapper.PlanMapper
        self._mapper = None
        self._specguide = None
        self.edgemode = edgemode
        self._mask = slice(None)
        self._candidates = None
        self._plotpchild = False
        self.slices = []
        #: raytraverse.evaluate.BaseMetricSet
        self.metricclass = metricclass
        #: iterable
        self.metricset = metricset
        #: int:
        self.features = len(metricset)

    @property
    def edgemode(self):
        return self._edgemode

    @edgemode.setter
    def edgemode(self, e):
        modes = ('reflect', 'constant', 'nearest', 'mirror', 'wrap')
        self._cval = -self.t1
        self._edgemode = e
        if e not in modes:
            try:
                self._cval = np.asarray(e, dtype=float).ravel()[0]
            except ValueError:
                raise ValueError(f"edgemode={e} not a valid option"
                                 f" must be one of: {modes}")
            else:
                self._edgemode = 'constant'

    def sampling_scheme(self, mapper):
        """calculate sampling scheme"""
        return np.array([mapper.shape(i) for i in range(self.nlev)])

    def run(self, mapper, specguide=None, plotp=False,
            **kwargs):
        """adapively sample an area defined by mapper

        Parameters
        ----------
        mapper: raytraverse.mapper.PlanMapper
            the pointset to build/run
        specguide: Union[None, bool, str]
        plotp: bool, optional
            plot weights, detail and vectors for each level
        kwargs:
            passed to self.run()

        Returns
        -------
        raytraverse.lightlplane.LightPlaneKD
        """
        name = mapper.name
        try_mkdir(f"{self.scene.outdir}/{name}")
        self._mapper = mapper
        reflf = f"{self.scene.outdir}/{name}/reflection_normals.txt"
        if specguide is True:
            self._specguide = reflf
        else:
            self._specguide = specguide
        levels = self.sampling_scheme(mapper)
        plotpthis = plotp and len(levels) > 1
        self._plotpchild = plotp and not plotpthis
        super().run(mapper, name, levels, plotp=plotpthis, **kwargs)
        return self._run_callback()

    def _init4run(self, levels, **kwargs):
        self.slices = []
        return super()._init4run(levels, **kwargs)

    def _run_callback(self):
        return LightPlaneKD(self.scene, self.idxvecs(), self._mapper,
                            self.stype)

    def repeat(self, guide, stype):
        """repeat the sampling of a guide LightPlane (to match all rays)

        Parameters
        ----------
        guide: LightPlaneKD
        stype: str
            alternate stype name for samplerpt. raises a ValueError if it
            matches the guide.
        Returns
        -------

        """
        self._mapper = guide.pm
        if stype == guide.src:
            raise ValueError("stype cannot match guide.src, as it would "
                             "overwrite data")
        ostype = self.stype
        self.stype = stype

        self.vecs = None
        self.lum = []
        self.slices = []

        gvecs = guide.vecs
        self._dump_vecs(gvecs)
        pbar = self.scene.progress_bar(self, list(enumerate(gvecs)),
                                       level=self._slevel, message="resampling")
        for posidx, point in pbar:
            gpt = guide.data[posidx]
            self.engine.repeat(gpt, stype)
        lp = self._run_callback()
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
            nweights = self._normed_weights(mask=self.edgemode == 'constant')
            p = draw.get_detail(nweights, *filterdict[self.detailfunc],
                                mode=self.edgemode, cval=self._cval)
            p = self.featurefunc(p.reshape(self.weights.shape), axis=0)
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
        lpargs = dict(parent=self._mapper.name)
        kwargs = dict(metricset=self.metricset, lmin=1e-8)
        level_desc = f"Level {len(self.slices)} of {len(self.levels)}"
        if self.nlev == 1 and len(vecs) == 1:
            logpoint = 'err'
        else:
            logpoint = None
        if logpoint is not None:
            self.engine._slevel -= 1
            pbar = list(zip(range(*idx), vecs))
        elif self._slevel > 0:
            pbar = list(zip(range(*idx), vecs))
        else:
            pbar = self.scene.progress_bar(self, list(zip(range(*idx), vecs)),
                                           level=self._slevel, message=level_desc)
        for posidx, point in pbar:
            lp = self.engine.run(point, posidx, specguide=self._specguide, lpargs=lpargs,
                                 log=logpoint, plotp=self._plotpchild, pfish=False)
            vol = lp.evaluate(1, includeviews=True)
            metric = self.metricclass(*vol, lp.vm, gcrnorm=self._gcrnorm,
                                      **kwargs)
            lums.append(metric())
        if logpoint is not None:
            self.engine._slevel += 1
        if len(self.lum) == 0:
            self.lum = np.array(lums)
        else:
            self.lum = np.concatenate((self.lum, lums), 0)
        return np.array(lums)

    def _update_weights(self, si, lum):
        """only used by _plot_weights, weights are recomputed from spatial
        query"""
        wv = np.moveaxis(self.weights, 0, 2)
        wv[tuple(si)] = lum

    def _lift_weights(self, i):
        """because areas can have an arbitrary border, upsamppling is not
        effective. If we mask the weights, then the UV edges (which are padded)
        will be treated differently from the inside edges. by remapping each
        level with the nearest sample value and then masking after calculating
        the detail we avoid these issues."""
        self._candidates = self._mapper.point_grid_uv(jitter=self.jitter,
                                                      level=i, masked=False,
                                                      snap=self.nlev-1)
        self._mask = self._mapper.in_view_uv(self._candidates, False)
        # avoid error by not sampling anything, assume that iif samplerarea is
        # called we atleast want to sample 1 point
        if not np.any(self._mask):
            # this assumes a MaskedPlanMapper, otherwise we are just repeating
            # 2 lines up, so next line will be false
            m2 = self._mapper.in_view_uv(self._candidates, False, usemask=False)
            if np.any(m2):
                one = np.random.default_rng().choice(np.arange(m2.size)[m2], 1)
            else:
                # fall back to the first candidate (should never happen)
                one = 0
            self._mask[one] = True
        if self.vecs is not None:
            wvecs = self._mapper.uv2xyz(self._candidates)
            d, idx = cKDTree(self.vecs).query(wvecs)
            if self.lum.ndim == 4:
                slum = self.weightfunc(self.lum, axis=2)
            else:
                slum = self.lum
            weights = slum[idx].reshape(*self.levels[i], self.features)
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
                nw.flat[np.logical_not(mk)] = self._cval
        return nweights

    def _dump_vecs(self, vecs):
        if self.vecs is None:
            self.vecs = vecs
            v0 = 0
        else:
            self.vecs = np.concatenate((self.vecs, vecs))
            v0 = self.slices[-1].stop
        self.slices.append(slice(v0, v0 + len(vecs)))
        vfile = (f"{self.scene.outdir}/{self._mapper.name}/{self.stype}"
                 f"_points.tsv")
        np.savetxt(vfile, self.idxvecs(), ("%d", "%d", "%.4f", "%.4f", "%.4f"))

    def idxvecs(self):
        idx = np.arange(len(self.vecs))[:, None]
        level = np.zeros_like(idx, dtype=int)
        for sl in self.slices[1:]:
            level[sl.start:] += 1
        return np.hstack((level, idx, self.vecs))

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
