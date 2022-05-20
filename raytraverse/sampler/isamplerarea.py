# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from clasp.script_tools import try_mkdir

from raytraverse.lightfield import LightField
from raytraverse.sampler.samplerarea import SamplerArea


class ISamplerArea(SamplerArea):
    """wavelet based area sampling class using Sensor as engine

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry and formatter compatible with engine
    engine: raytraverse.sampler.Sensor
        renderer
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
    """

    def __init__(self, scene, engine, accuracy=1.0, nlev=3, jitter=True,
                 edgemode='constant', **kwargs):
        super(SamplerArea, self).__init__(scene, engine, accuracy, **kwargs)
        self.features = engine.features * engine.engine.features
        #: int: number of sources return per vector by run
        self.srcn = engine.engine.srcn
        self.nlev = nlev
        self.jitter = jitter
        #: raytraverse.mapper.PlanMapper
        self._mapper = None
        modes = ('reflect', 'constant', 'nearest', 'mirror', 'wrap')
        if edgemode not in modes:
            raise ValueError(f"edgemode={edgemode} not a valid option"
                             " must be one of: {modes}")
        self._edgemode = edgemode
        self._mask = slice(None)
        self._candidates = None
        self.slices = []

    def run(self, mapper, plotp=False, **kwargs):
        """adapively sample an area defined by mapper

        Parameters
        ----------
        mapper: raytraverse.mapper.PlanMapper
            the pointset to build/run
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
        levels = self.sampling_scheme(mapper)
        super(SamplerArea, self).run(mapper, name, levels, plotp=plotp,
                                     **kwargs)

        idx = np.arange(len(self.vecs))[:, None]
        level = np.zeros_like(idx, dtype=int)
        for sl in self.slices[1:]:
            level[sl.start:] += 1
        idxvecs = np.hstack((level, idx, self.vecs))
        file = f"{self.scene.outdir}/{name}/{self.stype}_{self.engine.name}.npz"
        np.savez_compressed(file, vecs=idxvecs, lum=self.lum)
        return LightField(self.scene, self.vecs, self._mapper, self.stype)

    def sample(self, vecs):
        lum = super(SamplerArea, self).sample(vecs)
        return lum

    def repeat(self, guide, stype):
        raise ValueError("repeat not supported for ISamplerArea")

    def _process_features(self, lum):
        if self.srcn > 1:
            slum = self.weightfunc(lum, axis=2)
        else:
            slum = lum
        return lum, slum.reshape(-1, self.features)

    def _normed_weights(self, mask=False):
        nmin = np.min(self.weights)
        norm = np.max(self.weights) - nmin
        if norm > 0:
            nweights = (self.weights - nmin)/norm
        else:
            nweights = self.weights
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

    def _plot_weights(self, level, vm, name, suffix=".hdr", **kwargs):
        normw = self._normed_weights()
        for i, w in enumerate(normw):
            outw = (f"{self.scene.outdir}_{name}_{self.stype}_weight_{i}_"
                    f"{level:02d}{suffix}")
            w.flat[np.logical_not(self._mask)] = 0
            self._plot_dist(w, outw)
