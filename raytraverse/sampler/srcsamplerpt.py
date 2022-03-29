# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import tempfile

import numpy as np

from raytraverse import translate, io
from raytraverse.lightpoint import LightPointKD
from raytraverse.mapper import ViewMapper
from raytraverse.sampler.samplerpt import SamplerPt
from raytraverse.sampler.sunsamplerptview import SunSamplerPtView


class SrcSamplerPt(SamplerPt):
    """sample contributions from direct suns.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    engine: raytraverse.renderer.Rtrace
        initialized renderer instance (with scene loaded, no sources)
    sun: np.array
        shape 3, sun position
    sunbin: int
        sun bin
    """

    def __init__(self, scene, engine, source, stype="source", **kwargs):
        super().__init__(scene, engine, stype=stype, **kwargs)
        # update parameters post init
        # normalize accuracy for source intensity
        self.accuracy = self.accuracy * (1 - np.cos(.533*np.pi/360))
        #: np.array: sun position x,y,z
        self.sunpos = np.asarray(sun).flatten()[0:3]
        self._viewdirections = np.concatenate((self.sunpos, [.533])).reshape(1, 4)
        ambfile = f"{scene.outdir}/{stype}.amb"
        self.engine.load_source(source, ambfile=ambfile)

    def run(self, point, posidx, vm=None, plotp=False, specguide=None,
            **kwargs):
        if vm is None:
            vm = ViewMapper()
        self._levels = self.sampling_scheme(vm.aspect)
        self._load_specguide(specguide)
        return super().run(point, posidx, vm, plotp=plotp, **kwargs)

    def _load_specguide(self, specguide):
        """
        Parameters
        ----------
        specguide: str, raytraverse.lightpoint.LightPointKD
        """
        sunr = []
        if isinstance(specguide, str):
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
        viewsampler = SunSamplerPtView(self.scene, self.engine, self.sunpos,
                                       self.sunbin,
                                       samplerlevel=self._slevel + 1)
        vms = [ViewMapper(j[0:3], j[3], name=f"sunview_{i}", jitterrate=0)
               for i,j in enumerate(self._viewdirections)]
        sunview = viewsampler.run(point, posidx, vm=vms)
        self._viewdirections = np.concatenate((self.sunpos, [.533])).reshape(1, 4)
        lightpoint = LightPointKD(self.scene, self.vecs, self.lum,
                                  srcdir=self.sunpos, src=self.stype, pt=point,
                                  write=write, srcn=self.srcn, posidx=posidx,
                                  vm=vm, srcviews=sunview, **kwargs)
        return lightpoint


