# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from glob import glob

import numpy as np
from clasp.script_tools import try_mkdir
from scipy.spatial import cKDTree

from raytraverse import io, translate
from raytraverse.sampler import draw
from raytraverse.sampler.samplerarea import SamplerArea
from raytraverse.sampler.basesampler import BaseSampler, filterdict
from raytraverse.sampler.sunsamplerpt import SunSamplerPt


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
        areakwargs.update(samplerlevel=self._slevel + 1)
        self._areakwargs = dict(metricset=('avglum', 'loggcr', 'xpeak',
                                           'ypeak'), metricfunc=np.max)
        self._areakwargs.update(areakwargs)
        self._ptkwargs = ptkwargs
        self.nlev = nlev
        self.jitter = jitter
        self._areaweights = None
        self._specguide = None
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

    def get_existing_run(self, skymapper, areamapper, name=None, log=True):
        if name is None:
            name = areamapper.name
        vfile = f"{self.scene.outdir}/{name}/{skymapper.name}_{self.stype}.tsv"
        try:
            suns = np.loadtxt(vfile)
        except OSError:
            suns = None
        except ValueError:
            suns = None
            vfile = None
        ambfiles = glob(f"{self.scene.outdir}/{skymapper.name}_sun*.amb")
        ptfiles = glob(f"{self.scene.outdir}/{name}/{skymapper.name}_sun*"
                       f"points.tsv")
        conflict = suns is not None or len(ambfiles) > 0 or len(ptfiles) > 0
        if log:
            if vfile is not None:
                if suns is None:
                    s = 0
                else:
                    s = suns.shape[0]
                self.scene.log(self, f"{vfile} exists and has {s} sun "
                                     f"positions", err=True, level=-1)
            if len(ambfiles) > 0:
                self.scene.log(self, f"there are potentially {len(ambfiles)}"
                               f"ambfiles conflicting with this run,  starting "
                               f"with {ambfiles[0]}", err=True, level=-1)
            if len(ptfiles) > 0:
                self.scene.log(self, f"there are potentially {len(ptfiles)} "
                               f"area samples that would be overwritten, "
                               f"starting with {ptfiles[0]}",
                               err=True, level=-1)
            if not conflict:
                self.scene.log(self, f"directory clean, no file conflict found"
                               f" with run: {self.scene.outdir}, "
                               f"{skymapper.name}, {name}", err=True, level=-1)
        return conflict, (vfile, suns, ambfiles, ptfiles)

    def run(self, skymapper, areamapper, name=None, specguide=None, **kwargs):
        """adapively sample sun positions for an area (also adaptively sampled)

        Parameters
        ----------
        skymapper: raytraverse.mapper.SkyMapper
            the mapping for drawing suns
        areamapper: raytraverse.mapper.PlanMapper
            the mapping for drawing points
        name: str, optional
        specguide: raytraverse.lightfield.LightPlaneKD
            sky source lightfield to use as specular guide for sampling
        kwargs:
            passed to self.run()

        Returns
        -------
        raytraverse.lightlplane.LightPlaneKD
        """
        conflict, data = self.get_existing_run(skymapper, areamapper, name)
        # if conflict:
        #     raise ValueError("run exists!")
        if name is None:
            name = areamapper.name
        try_mkdir(f"{self.scene.outdir}/{name}")
        self._name = name
        self._skymapper = skymapper
        self._areamapper = areamapper
        self._specguide = specguide
        self._areaweights = None
        levels = self.sampling_scheme(skymapper)
        super().run(skymapper, name, levels, **kwargs)
        # return MultiSourcePlaneKD(self.scene, self.vecs, self._areamapper, self.stype)

    def draw(self, level):
        """draw on condition of in_solarbounds from skymapper.
        In this way all solar positions are presented to the area sampler, but
        the area sampler is initialized with a weighting to sample only where
        there is variancee between sun position. this keeps the subsampling of
        area and solar position independent, escaping dimensional curses.

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        p: np.array
            computed probabilities
        """
        dres = self.levels[level]
        pdraws = np.arange(int(np.prod(dres)))[self._mask]
        if level > 0:
            wvecs = self._skymapper.uv2xyz(self._candidates)
            d, idx = cKDTree(self.vecs).query(wvecs)
            weights = self.lum[idx].reshape(*self._wshape(level),
                                            *self.lum.shape[1:])
            for i, j in enumerate(self._areakwargs['metricset']):
                if 'lum' in j:
                    nmin = np.min(weights[:, :, i, ...])
                    norm = np.max(weights[:, :, i, ...]) - nmin
                    if norm > 0:
                        weights[:, :, i, ...] = (weights[:, :, i,
                                                 ...] - nmin)/norm
            det = draw.get_detail(weights.T, *filterdict[self.detailfunc])
            # mfunc = self._areakwargs['metricfunc']
            # det = mfunc(det.reshape(weights.T.shape).T, axis=2)
            det = det.reshape(weights.T.shape).T[:, :, 0, ...]
            self._areaweights = det.reshape(-1, *self.lum.shape[-2:])[pdraws]
            np.save("areaweights.npy", self._areaweights)
            np.save("candidates.npy", self._candidates[pdraws])
            raise ValueError
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

        idx = self.slices[-1].indices(self.slices[-1].stop)
        lums = []
        for suni, sunpos in zip(range(*idx), vecs):
            sunsamp = SunSamplerPt(self.scene, self.engine, sunpos, suni,
                                   stype=f"{self._skymapper.name}_sun",
                                   **self._ptkwargs)
            areasampler = SamplerArea(self.scene, sunsamp, **self._areakwargs)
            # iweight = self._areaweights[suni - idx[0]]
            lf = areasampler.run(self._areamapper, name=self._name,
                                 specguide=self._specguide)
            lf.direct_view()
            lums.append(areasampler.weights)
        if len(self.lum) == 0:
            self.lum = np.array(lums)
        else:
            self.lum = np.concatenate((self.lum, lums), 0)
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
        self.weights = np.ones(self._wshape(i))

    def _dump_vecs(self, vecs):
        if self.vecs is None:
            self.vecs = vecs
            v0 = 0
        else:
            self.vecs = np.concatenate((self.vecs, vecs))
            v0 = self.slices[-1].stop
        self.slices.append(slice(v0, v0 + len(vecs)))
        level = len(self.slices)
        self.slices.append(slice(v0, v0 + len(vecs)))
        vfile = f"{self.scene.outdir}/{self._name}/{self._skymapper.name}_{self.stype}.tsv"
        idx = np.arange(len(self.vecs))[:, None]
        idxvecs = np.hstack((np.full_like(idx, level), idx, self.vecs))
        np.savetxt(vfile, idxvecs, ("%d", "%d", "%.4f", "%.4f", "%.4f"))

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
