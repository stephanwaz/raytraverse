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

from raytraverse import translate
from raytraverse.lightpoint import LightPointKD
from raytraverse.mapper import ViewMapper
from raytraverse.sampler.samplerpt import SamplerPt
from raytraverse.sampler.sunsamplerptview import SunSamplerPtView


class SunSamplerPt(SamplerPt):
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

    def __init__(self, scene, engine, sun, sunbin, nlev=6, stype='sun',
                 **kwargs):
        super().__init__(scene, engine, stype=f"{stype}_{sunbin:04d}",
                         nlev=nlev, **kwargs)
        # update parameters post init
        # normalize accuracy for sun source
        self.accuracy = self.accuracy * (1 - np.cos(.533*np.pi/360))
        #: np.array: sun position x,y,z
        self.sunpos = np.asarray(sun).flatten()[0:3]
        self._viewdirections = np.concatenate((self.sunpos, [.533])).reshape(1, 4)
        self.sunbin = sunbin
        # load new source
        f, srcdef = tempfile.mkstemp(dir=f"./{scene.outdir}/", prefix='tmp_src')
        # srcdef = f'{scene.outdir}/tmp_srcdef_{sunbin}.rad'
        f = open(srcdef, 'w')
        f.write(scene.formatter.get_sundef(sun, (1, 1, 1)))
        f.close()
        ambfile = f"{scene.outdir}/{stype}_{sunbin:04d}.amb"
        self.engine.load_source(srcdef, ambfile=ambfile)
        os.remove(srcdef)

    def run(self, point, posidx, specguide=None, **kwargs):
        self._load_specguide(specguide)
        return super().run(point, posidx, **kwargs)

    def _load_specguide(self, specguide):
        """
        Parameters
        ----------
        specguide: str, raytraverse.lightpoint.LightPointKD
        """
        sunr = []
        if hasattr(specguide, "lower"):
            try:
                refl = translate.norm(np.loadtxt(specguide).reshape(-1, 3))
                sunr = translate.reflect(self.sunpos.reshape(1, 3), refl, True)
            except (OSError, ValueError, FileNotFoundError):
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


