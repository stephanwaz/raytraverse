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

from raytraverse import io, translate
from raytraverse.sampler import draw
from raytraverse.sampler.basesampler import BaseSampler, filterdict
from raytraverse.sampler.sunsamplerpt import SunSamplerPt
from raytraverse.evaluate import SamplingMetrics
from raytraverse.lightfield import LightPlaneKD


class SamplerSuns(BaseSampler):
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
                 ptkwargs=None, areakwargs=None):
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
        #: raytraverse.mapper.PlanMapper
        self._areamapper = None
        self._name = None
        self._mask = slice(None)
        self._candidates = None
        self.slices = []

    def sampling_scheme(self, mapper):
        """calculate sampling scheme"""
        return np.array([mapper.shape(i) for i in range(self.nlev)])

    def run(self, skymapper, areamapper, name=None, **kwargs):
        """adapively sample sun positions for an area (also adaptively sampled)

        Parameters
        ----------
        skymapper: raytraverse.mapper.SkyMapper
            the mapping for drawing suns
        areamapper: raytraverse.mapper.PlanMapper
            the mapping for drawing points
        name: str, optional
        kwargs:
            passed to self.run()

        Returns
        -------
        raytraverse.lightlplane.LightPlaneKD
        """
        if name is None:
            name = f"{skymapper.name}_{areamapper.name}"
        try_mkdir(f"{self.scene.outdir}/{name}")
        self._name = name
        self._skymapper = skymapper
        self._areamapper = areamapper
        levels = self.sampling_scheme(skymapper)
        super().run(skymapper, name, levels, **kwargs)
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
        # lpargs = dict(parent=self._name)
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
        self._candidates = self._skymapper.solar_grid_uv(jitter=self.jitter,
                                                         level=i, masked=False)
        shape = self._skymapper.shape(i)
        mask = self._skymapper.in_solarbounds_uv(self._candidates,
                                                 False).reshape(shape)
        if self.vecs is not None:
            ij = translate.uv2ij(self._skymapper.xyz2uv(self.vecs),
                                 shape[0], 1).T
            mask[ij[0], ij[1]] = False
        self._mask = mask.ravel()
        self.weights = np.ones(self.levels[i])

    def _dump_vecs(self, vecs):
        if self.vecs is None:
            self.vecs = vecs
            v0 = 0
        else:
            self.vecs = np.concatenate((self.vecs, vecs))
            v0 = self.slices[-1].stop
        self.slices.append(slice(v0, v0 + len(vecs)))
        vfile = f"{self.scene.outdir}/{self._name}/{self.stype}.tsv"
        idxvecs = np.hstack((np.arange(len(self.vecs))[:, None], self.vecs))
        np.savetxt(vfile, idxvecs, ("%d", "%.8f", "%.8f", "%.8f"))

    def _wshape(self, level):
        return np.concatenate(([self.features], self.levels[level]))

    @staticmethod
    def _plot_dist(ps, vm, outf, fisheye=True):
        outshape = (512*vm.aspect, 512)
        res = outshape[-1]
        if fisheye:
            pixelxyz = vm.pixelrays(res)
            uv = vm.xyz2uv(pixelxyz.reshape(-1, 3))
            pdirs = np.concatenate((pixelxyz[0:res], -pixelxyz[res:]), 0)
            mask = vm.in_view(pdirs, indices=False).reshape(outshape)
            ij = translate.uv2ij(uv, ps.shape[-1], aspect=vm.aspect)
            img = ps[ij[:, 0], ij[:, 1]].reshape(outshape)
            io.array2hdr(np.where(mask, img, 0), outf)
        else:
            detail = translate.resample(ps[-1::-1], outshape, radius=0,
                                        gauss=False)
            if vm.aspect == 2:
                detail = np.concatenate((detail[res:], detail[0:res]), 0)
            io.array2hdr(detail, outf)

    def _plot_p(self, p, level, vm, name, suffix=".hdr", fisheye=True,
                **kwargs):
        ps = p.reshape(self.weights.shape)
        outp = (f"{self.scene.outdir}_{name}_{self.stype}_detail_"
                f"{level:02d}{suffix}")
        self._plot_dist(ps, vm, outp, fisheye)

    def _plot_weights(self, level, vm, name, suffix=".hdr", fisheye=True,
                      **kwargs):
        pass

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr", fisheye=True,
                   **kwargs):
        outv = (f"{self.scene.outdir}_{name}_{self.stype}_samples_"
                f"{level:02d}{suffix}")
        self._skymapper.plot(vecs, outv, res=512, grow=2, fisheye=fisheye)
