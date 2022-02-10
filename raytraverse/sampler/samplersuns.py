# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from clasp.script_tools import try_mkdir, sglob

from raytraverse import io, translate
from raytraverse.mapper import MaskedPlanMapper
from raytraverse.sampler import draw
from raytraverse.sampler.samplerarea import SamplerArea
from raytraverse.sampler.basesampler import BaseSampler, filterdict
from raytraverse.sampler.sunsamplerpt import SunSamplerPt
from raytraverse.lightfield import SunsPlaneKD, LightPlaneKD
from raytraverse.evaluate import SamplingMetrics
from raytraverse.utility import pool_call


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
    metricset: iterable, optional
        subset of samplerarea.metric set to use for sun detail calculation.
    """

    #: initial sampling threshold coefficient
    t0 = .05
    #: final sampling threshold coefficient
    t1 = .5
    #: upper bound for drawing from pdf
    ub = 8

    def __init__(self, scene, engine, accuracy=1.0, nlev=3, jitter=True,
                 ptkwargs=None, areakwargs=None,
                 metricset=('avglum', 'loggcr')):
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
        self._metidx = [i for i, j in enumerate(self._areakwargs['metricset'])
                        if j in metricset]
        if len(self._metidx) != len(metricset):
            raise ValueError(f"bad argument metricset={metricset},all items "
                             f"must be in {self._areakwargs['metricset']}")
        # initialize runtime variables:
        #: extra variables since sampler also needs to track each areasampler
        self._areadraws = None
        self._areadetail = None
        self._areaweights = None
        #: the LightPlaneKD of the sky sourcee used as a specular sampling guide
        #: in each sunsamplerpt (set in self.run())
        self._specguide = None
        #: raytraverse.mapper.SkyMapper (set in self.run())
        self._skymapper = None
        #: raytraverse.mapper.PlanMapper (set in self.run())
        self._areamapper = None
        self._mask = slice(None)
        self._candidates = None
        self.slices = []
        self._recovery_data = None

    def sampling_scheme(self, mapper):
        """calculate sampling scheme"""
        return np.array([mapper.shape(i) for i in range(self.nlev)])

    def get_existing_run(self, skymapper, areamapper):
        """check for file conflicts before running/overwriting parameters
        match call to run

        Parameters
        ----------
        skymapper: raytraverse.mapper.SkyMapper
            the mapping for drawing suns
        areamapper: raytraverse.mapper.PlanMapper
            the mapping for drawing points

        Returns
        -------
        conflicts: tuple
            a tuple of found conflicts (None for each if no conflicts:

                - suns: np.array of sun positions in vfile
                - ptfiles: existing point files

        """
        vfile = (f"{self.scene.outdir}/{areamapper.name}/{skymapper.name}_"
                 f"{self.stype}.tsv")
        try:
            suns = np.loadtxt(vfile)
        except OSError:
            suns = None
        except ValueError:
            suns = None
        ptfiles = sglob(f"{self.scene.outdir}/{areamapper.name}/"
                        f"{skymapper.name}_sun*points.tsv")
        return suns, ptfiles

    def run(self, skymapper, areamapper, specguide=None,
            recover=True, **kwargs):
        """adaptively sample sun positions for an area (also adaptively sampled)

        Parameters
        ----------
        skymapper: raytraverse.mapper.SkyMapper
            the mapping for drawing suns
        areamapper: raytraverse.mapper.PlanMapper
            the mapping for drawing points
        specguide: raytraverse.lightfield.LightPlaneKD
            sky source lightfield to use as specular guide for sampling
        recover: continue run on top of existing files, if false, overwrites
            previous run.
        kwargs:
            passed to self.run()

        Returns
        -------
        raytraverse.lightlplane.LightPlaneKD
        """
        sun_conflict, pt_conflict = self.get_existing_run(skymapper, areamapper)
        if sun_conflict is not None and recover:
            self._recovery_data = sun_conflict, pt_conflict
        else:
            self._recovery_data = None
        try_mkdir(f"{self.scene.outdir}/{areamapper.name}")
        # reset/initialize runtime properties
        self._skymapper = skymapper
        self._areamapper = areamapper
        self._specguide = specguide
        self._areaweights = None
        levels = self.sampling_scheme(skymapper)
        super().run(skymapper, areamapper.name, levels, **kwargs)
        return SunsPlaneKD(self.scene, self.vecs, self._areamapper,
                           f"{self._skymapper.name}_sun")

    def _init4run(self, levels, **kwargs):
        """(re)initialize object for new run, ensuring properties are cleared
        prior to executing sampling loop"""
        leveliter = super()._init4run(levels)
        if self._recovery_data is None:
            return leveliter
        levels2run = []
        suns = self._recovery_data[0]
        pts = self._recovery_data[1]
        pt_cnt = len(pts)
        for i in leveliter:
            levelsuns = suns[suns[:, 0] == i]
            if pt_cnt > 0:
                self._recover_level(levelsuns[:, 2:], i, **kwargs)
                pt_cnt -= len(levelsuns)
            else:
                levels2run.append(i)
        return levels2run

    def _recover_level(self, vecs, i, plotp=False, pfish=True):
        """use existing drawn sun positions to rerun, trying to load LightFields
        before running"""
        mapper = self._skymapper
        name = self._areamapper.name
        shape = self.levels[i]
        uv = self._skymapper.xyz2uv(vecs)
        idx = self._skymapper.uv2idx(uv, shape)
        candidates = self._skymapper.solar_grid_uv(jitter=self.jitter,
                                                   level=i, masked=False)
        candidates[idx] = uv
        self._candidates = candidates
        self._lift_weights(i)
        draws, p = self.draw(i)
        si, uv = self.sample_to_uv(draws, shape)
        if si.size > 0:
            srate = si.shape[1]/np.prod(shape)
            row = (f"{i + 1} of {self.levels.shape[0]}\t"
                   f"{str(shape): >11}\t{si.shape[1]: >7}\t"
                   f"{srate: >7.02%}")
            self.scene.log(self, row, True, level=self._slevel)
            if plotp:
                self._plot_p(p, i, mapper, name, fisheye=pfish)
                self._plot_vecs(vecs, i, mapper, name, fisheye=pfish)
            lum = self.sample(vecs)
            self._update_weights(si, lum)
            if plotp:
                self._plot_weights(i, mapper, name, fisheye=pfish)

    def draw(self, level):
        """draw on condition of in_solarbounds from skymapper.
        In this way all solar positions are presented to the area sampler, but
        the area sampler is initialized with a weighting to sample only where
        there is variance between sun position. this keeps the subsampling of
        area and solar position independent.

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        p: np.array
            computed probabilities
        """
        dres = self.levels[level]
        # draw all sun positions within solar bounds
        pdraws = np.arange(int(np.prod(dres)))[self._mask]
        p = np.ones(dres).ravel()
        p[np.logical_not(self._mask)] = 0
        if level > 0:
            # calculate detail over points across sun positions
            nweights = self._normed_weights()
            emode = 'reflect'
            if level < 2:
                emode = 'constant'
            det = draw.get_detail(nweights,
                                  *filterdict[self.detailfunc], mode=emode)
            mfunc = self._areakwargs['metricfunc']
            det = np.transpose(det.reshape(self._areaweights.shape),
                               (3, 4, 0, 1, 2))
            det = mfunc(det, axis=2)
            self._areadetail = det
            adetail = det.reshape(-1, *self.lum.shape[-2:])[pdraws]
            adraws = []
            drawmask = []
            # draw on each plan and store plan candidates in areadraws for
            # MaskedPlanMapper construction (in self.sample())
            for ad in adetail:
                adr = draw.from_pdf(ad.ravel(), self._threshold(level),
                                    lb=self.lb, ub=self.ub)
                if len(adr) > 0:
                    auv = self._areamapper.idx2uv(adr, self.lum.shape[-2:],
                                                  jitter=False)
                    adraws.append(self._areamapper.uv2xyz(auv))
                    drawmask.append(True)
                else:
                    drawmask.append(False)
            self._areadraws = adraws
            # filter no detail plans
            pdraws = pdraws[drawmask]
        else:
            self._areadraws = [None] * len(pdraws)
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
        uv = self._candidates[pdraws]
        self._candidates = None
        return si, uv

    def _sample_sun(self, suni, sunpos, adraws):
        """this function is for calling with a process pool, by declaring the
        sun sampler point after the child process is forked/spawned the
        call to engine.load_source happens on an isolated memory instance,
        allowing for concurrency on different scenes, despite the singleton/
        global namespace issues of the cRtrace instance."""
        sunsamp = SunSamplerPt(self.scene, self.engine, sunpos, suni,
                               stype=f"{self._skymapper.name}_sun",
                               **self._ptkwargs)
        areasampler = SamplerArea(self.scene, sunsamp, **self._areakwargs)
        if adraws is not None:
            amapper = MaskedPlanMapper(self._areamapper, adraws, 0)
        else:
            amapper = self._areamapper
        needs_sampling = True
        if self._recovery_data is not None:
            try:
                src = f"{self._skymapper.name}_sun_{suni:04d}"
                pts = (f"{self.scene.outdir}/{self._areamapper.name}/"
                       f"{src}_points.tsv")
                lf = LightPlaneKD(self.scene, pts, self._areamapper, src)
                metrics = self._areakwargs['metricset']
                try:
                    mc = self._areakwargs['metricclass']
                except KeyError:
                    mc = SamplingMetrics
                areasampler.lum = lf.evaluate(1, metrics=metrics,
                                              metricclass=mc)
                shp = (*areasampler.sampling_scheme(amapper)[-1],
                       areasampler.features)
            except (ValueError, OSError):
                pass
            else:
                needs_sampling = False
        if needs_sampling:
            lf = areasampler.run(amapper, specguide=self._specguide,
                                 log=False)
            shp = (*areasampler.levels[-1], areasampler.features)
        # build weights based on final sampling
        candidates = amapper.point_grid_uv(jitter=False,
                                           level=areasampler.nlev - 1,
                                           masked=False)
        mask = amapper.in_view_uv(candidates, False)
        wvecs = amapper.uv2xyz(candidates)
        idx, d = lf.query(wvecs)
        weights = areasampler.lum[idx].reshape(shp)
        weights = np.moveaxis(weights, 2, 0)
        weights = weights[self._metidx]
        for w in weights:
            w.flat[np.logical_not(mask)] = -1
        return weights

    def sample(self, vecs):
        """call rendering engine to sample rays

        Parameters
        ----------
        vecs: np.array
            sample vectors

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        self._dump_vecs(vecs)
        idx = self.slices[-1].indices(self.slices[-1].stop)
        level_desc = f"Level {len(self.slices)} of {len(self.levels)}"
        # if engine is configured for internal multiprocessing run on one
        # process, else use environment cap. Generally, if -ab > 0 then run
        # with rtrace -n X for better memory efficiency, else, use processpool
        # cap
        if self.engine.nproc is None or self.engine.nproc > 1:
            cap = 1
        else:
            cap = None
        lums = pool_call(self._sample_sun, list(zip(range(*idx), vecs,
                         self._areadraws)), desc=level_desc, cap=cap)
        lums = np.array(lums)
        # initialize areaweights here now that we know the shape
        if len(self.lum) == 0:
            self.lum = lums
            self._areaweights = np.zeros((*self.lum.shape[1:],
                                          *self._wshape(0)))
        else:
            self.lum = np.concatenate((self.lum, lums), 0)
        return lums

    def _update_weights(self, si, lum):
        """only need to update areaweights, base weights are only masked"""
        lum = np.transpose(lum, (1, 2, 3, 0))
        update = self._areaweights[..., si[0], si[1]]
        self._areaweights[..., si[0], si[1]] = np.where(lum > 0, lum, update)

    def _lift_weights(self, i):
        if self._candidates is None:
            self._candidates = self._skymapper.solar_grid_uv(jitter=self.jitter,
                                                             level=i, masked=False)
        shape = self._skymapper.shape(i)
        mask = self._skymapper.in_solarbounds_uv(self._candidates,
                                                 level=i).reshape(shape)
        if self.vecs is not None:
            ij = translate.uv2ij(self._skymapper.xyz2uv(self.vecs),
                                 shape[0], 1).T
            mask[ij[0], ij[1]] = False
        self._mask = mask.ravel()
        # at the level of the skysampler, draw all suns, detail happens
        # at the level of the area sampler
        self.weights = np.ones(self._wshape(i))
        if i > 0:
            shp = (*self.lum.shape[1:], *self.weights.shape)
            self._areaweights = translate.resample(self._areaweights, shp)

    def _normed_weights(self):
        normi = [i for i, j in
                 enumerate(self._areakwargs['metricset']) if 'lum' in j]
        nweights = np.copy(self._areaweights)
        for i in normi:
            nmin = np.min(nweights[i])
            norm = np.max(nweights[i]) - nmin
            if norm > 0:
                nweights[i] = (nweights[i] - nmin)/norm
        return nweights

    def _dump_vecs(self, vecs):
        if self.vecs is None:
            self.vecs = vecs
            v0 = 0
        else:
            self.vecs = np.concatenate((self.vecs, vecs))
            v0 = self.slices[-1].stop
        self.slices.append(slice(v0, v0 + len(vecs)))
        vfile = (f"{self.scene.outdir}/{self._areamapper.name}/"
                 f"{self._skymapper.name}_{self.stype}.tsv")
        idx = np.arange(len(self.vecs))[:, None]
        level = np.zeros_like(idx, dtype=int)
        for sl in self.slices[1:]:
            level[sl.start:] += 1
        idxvecs = np.hstack((level, idx, self.vecs))
        # file format: level idx sx sy sz
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
        if self._areadetail is not None:
            outp = (f"{self.scene.outdir}_{name}_{self.stype}_areadetail_"
                    f"{level:02d}{suffix}")
            self._plot_dist(np.max(self._areadetail, axis=(2, 3)), vm, outp,
                            fisheye)

    def _plot_weights(self, level, vm, name, suffix=".hdr", fisheye=True,
                      **kwargs):
        if self._areaweights is not None:
            normw = self._normed_weights()
            for m, w in zip(self._areakwargs['metricset'], normw):
                outp = (f"{self.scene.outdir}_{name}_{self.stype}_{m}"
                        f"_{level:02d}{suffix}")
                self._plot_dist(np.max(w, axis=(0, 1)), vm, outp, fisheye)

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr", fisheye=True,
                   **kwargs):
        outv = (f"{self.scene.outdir}_{name}_{self.stype}_samples_"
                f"{level:02d}{suffix}")
        self._skymapper.plot(vecs, outv, res=512, grow=2, fisheye=fisheye)
