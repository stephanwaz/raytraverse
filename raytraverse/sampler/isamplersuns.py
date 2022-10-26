# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from clasp.script_tools import try_mkdir

from raytraverse.mapper import MaskedPlanMapper
from raytraverse.sampler import ISamplerArea, Sensor
from raytraverse.sampler.samplersuns import SamplerSuns
from raytraverse.lightfield import SunSensorPlaneKD
from raytraverse.utility import pool_call


class ISamplerSuns(SamplerSuns):
    """wavelet based sun position sampling class

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry and formatter compatible with engine
    engine: raytraverse.sampler.Sensor
        with initialized renderer instance (with scene loaded, no sources)
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

    def __init__(self, scene, engine, accuracy=1.0, nlev=3, jitter=True,
                 areakwargs=None, t0=.05, t1=.125):
        super(SamplerSuns, self).__init__(scene, engine, accuracy,
                                          stype=f'sunpositions', t0=t0, t1=t1)
        if areakwargs is None:
            areakwargs = {}
        areakwargs.update(samplerlevel=self._slevel + 1)
        self._areakwargs = dict(featurefunc=np.max, edgemode=0.0)
        self._areakwargs.update(areakwargs)
        self.nlev = nlev
        self.jitter = jitter
        # initialize runtime variables:
        #: extra variables since sampler also needs to track each areasampler
        self._areadraws = None
        self._areadetail = None
        self._areaweights = None
        #: raytraverse.mapper.SkyMapper (set in self.run())
        self._skymapper = None
        #: raytraverse.mapper.PlanMapper (set in self.run())
        self._areamapper = None
        self._mask = slice(None)
        self._candidates = None
        self.slices = []
        self._recovery_data = None

    def get_existing_run(self, skymapper, areamapper):
        raise ValueError("get_existing_run not supported for ISamplerArea")

    def run(self, skymapper, areamapper, **kwargs):
        """adaptively sample sun positions for an area (also adaptively sampled)

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
        # reset/initialize runtime properties
        self._skymapper = skymapper
        self._areamapper = areamapper
        self._areaweights = None
        try_mkdir(f"{self.scene.outdir}/{areamapper.name}")
        levels = self.sampling_scheme(skymapper)
        super(SamplerSuns, self).run(skymapper, areamapper.name, levels,
                                     **kwargs)
        src = f"{self.engine.name}_{self._skymapper.name}_sun"

        return SunSensorPlaneKD(self.scene, self.idxvecs(), self._areamapper,
                                src)

    def _run_sample(self, idx, vecs, level_desc):
        if self.engine.nproc is None or self.engine.nproc > 1:
            cap = 1
        else:
            cap = None

        sk = dict(dirs=self.engine.dirs, offsets=self.engine.offsets,
                  name=self.engine.name, sunview=self.engine.sunview)
        stype = f"{self._skymapper.name}_sun"
        lums = pool_call(_sample_sun, list(zip(range(*idx), vecs,
                                               self._areadraws)), self.scene,
                         self.engine.engine,
                         self._areamapper, stype=stype, desc=level_desc,
                         cap=cap, pbar=self.scene.dolog, sensorkwargs=sk,
                         areakwargs=self._areakwargs)
        return np.array(lums)

    def _normed_weights(self):
        nmin = np.min(self._areaweights)
        norm = np.max(self._areaweights) - nmin
        if norm > 0:
            nweights = (self._areaweights - nmin)/norm
        else:
            nweights = self._areaweights
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
                 f"{self.engine.name}_{self._skymapper.name}_{self.stype}.tsv")
        # file format: level idx sx sy sz
        np.savetxt(vfile, self.idxvecs(), ("%d", "%d", "%.4f", "%.4f", "%.4f"))

    def _plot_weights(self, level, vm, name, suffix=".hdr", fisheye=True,
                      **kwargs):
        if self._areaweights is not None:
            normw = self._normed_weights()
            for m, w in enumerate(normw):
                outp = (f"{self.scene.outdir}_{name}_{self.stype}_{m}"
                        f"_{level:02d}{suffix}")
                self._plot_dist(np.max(w, axis=(0, 1)), vm, outp, fisheye)


def _sample_sun(suni, sunpos, adraws, scene, engine, mapper, stype="sun",
               sensorkwargs=None, areakwargs=None):
    """this function is for calling with a process pool, by declaring the
    sun sampler point after the child process is forked the
    call to engine.load_source happens on an isolated memory instance,
    allowing for concurrency on different scenes, despite the singleton/
    global namespace issues of the cRtrace instance."""
    if sensorkwargs is None:
        sensorkwargs = {}
    if areakwargs is None:
        areakwargs = None
    ambfile = f"{scene.outdir}/{stype}_{suni:04d}.amb"
    engine.load_solar_source(scene, sunpos, ambfile)
    sensor = Sensor(engine, **sensorkwargs)
    areasampler = ISamplerArea(scene, sensor, stype=f"{stype}_{suni:04d}",
                               **areakwargs)
    if adraws is not None:
        amapper = MaskedPlanMapper(mapper, adraws, 0)
    else:
        amapper = mapper
    # amapper = mapper
    lf = areasampler.run(amapper, log=False)
    shp = (*areasampler.levels[-1], areasampler.features)
    # build weights based on final sampling
    candidates = amapper.point_grid_uv(jitter=False,
                                       level=areasampler.nlev - 1,
                                       masked=False)
    mask = amapper.in_view_uv(candidates, False)
    wvecs = amapper.uv2xyz(candidates)
    idx, d = lf.query(wvecs)
    if areasampler.lum.ndim == 4:
        slum = areasampler.weightfunc(areasampler.lum, axis=2)
    else:
        slum = areasampler.lum
    weights = slum[idx].reshape(shp)
    weights = np.moveaxis(weights, 2, 0)
    for w in weights:
        w.flat[np.logical_not(mask)] = -1
    return weights
