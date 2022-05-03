# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate, io
from raytraverse.lightpoint import LightPointKD
from raytraverse.mapper import ViewMapper
from raytraverse.sampler.samplerpt import SamplerPt
from raytraverse.sampler.srcsamplerptview import SrcSamplerPtView


class SrcSamplerPt(SamplerPt):
    """sample contributions from fixed sources.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    engine: raytraverse.renderer.Rtrace
        initialized renderer instance (with scene loaded, no sources)
    source: str
        single scene file containing sources (including sky, lights, sun, etc)
    sunbin: int
        sun bin
    """

    def __init__(self, scene, engine, source, stype="source", **kwargs):
        super().__init__(scene, engine, stype=stype, **kwargs)
        # update parameters post init
        ambfile = f"{scene.outdir}/{stype}.amb"
        self.engine.load_source(source, ambfile=ambfile)
        srcs, distant = self.engine.get_sources()
        #: non distant sources, pos, radius, area
        self.lights = srcs[np.logical_not(distant)]
        #: amount of circle filled by each light
        self._fillratio = (self.lights[:, 4] /
                           (np.square(self.lights[:, 3]) * np.pi))
        #: distant sources (includes those with large solid angles
        self.sources = srcs[distant]
        #: gets initialized for each point, as apparent light size will change
        self._viewdirections = None
        #: gets initialized for each point using direct illuminance from sources
        #: should not be less than 1.08173E-05 (accuracy of sunsampler pt)
        self._normaccuracy = self.accuracy
        #: set sampling level/strategy for lights based on source solid angle,
        #: fill ratio and sampling resolution
        self._samplelevels = None
        print(self.lights)
        print(self._fillratio)
        print(self.sources)
        print(self._viewdirections)
        print(self._normaccuracy)

    def run(self, point, posidx, specguide=None, **kwargs):
        self._load_specguide(specguide)
        self._set_normalization()
        return super().run(point, posidx, **kwargs)

    def _load_specguide(self, specguide):
        """
        Parameters
        ----------
        specguide: str
            file with reflection normals.
        """
        sunr = []
        if hasattr(specguide, "lower"):
            try:
                refl = translate.norm(io.load_txt(specguide).reshape(-1, 3))
                sunr = translate.reflect(self.sunpos.reshape(1, 3), refl, True)
            except (ValueError, FileNotFoundError):
                pass
        if len(sunr) > 0:
            reflsize = np.full((len(sunr), 1), 1.066)
            refl = np.hstack((sunr, reflsize))
            self._viewdirections = np.concatenate((self._viewdirections, refl))

    def _run_callback(self, point, posidx, vm, write=True, **kwargs):
        if self._viewdirections is None or len(self._viewdirections) == 0:
            srcview = None
        else:
            viewsampler = SrcSamplerPtView(self.scene, self.engine,
                                           samplerlevel=self._slevel + 1)
            vms = [ViewMapper(j[0:3], j[3], name=f"{self.stype}_{i}",
                              jitterrate=0)
                   for i, j in enumerate(self._viewdirections)]
            srcview = viewsampler.run(point, posidx, vm=vms)
        lightpoint = LightPointKD(self.scene, self.vecs, self.lum,
                                  src=self.stype, pt=point, write=write,
                                  srcn=self.srcn, posidx=posidx,
                                  vm=vm, srcviews=srcview, **kwargs)
        self.accuracy = self._normaccuracy
        self._viewdirections = None
        self._samplelevels = None
        return lightpoint


