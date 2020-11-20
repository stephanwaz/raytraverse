# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse.sampler.sampler import Sampler
from raytraverse.sampler.sunviewsampler import SunViewSampler
from raytraverse.scene import SunSetterBase


class SkySampler(Sampler):
    """sample predefined sky definition.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    srcdef: str
        path to sky definition
    skyname: str
        unique name for saving results in scene directory
    keepamb: bool, optional
        whether to keep ambient files after run, if kept, a successive call
        will load these ambient files, so care must be taken to not change
        any parameters
    ambcache: bool, optional
        whether the rcopts indicate that the calculation will use ambient
        caching (and thus should write an -af file argument to the engine)
    """

    def __init__(self, scene, srcdef, skyname, fdres=10,
                 rcopts='-ab 6 -ad 3000 -as 1500 -st 0 -ss 16 -aa .1',
                 keepamb=False, ambcache=True, **kwargs):
        # update ambient file and args before init
        self._keepamb = keepamb and ambcache
        if ambcache:
            ambfile = f"{scene.outdir}/{skyname}.amb"
        else:
            ambfile = None
        engine_args = scene.formatter.get_standard_args(rcopts, ambfile)
        super().__init__(scene, stype=skyname, fdres=fdres,
                         engine_args=engine_args,  **kwargs)
        self.engine.load_source(srcdef)
        self.sources = srcdef

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, srcdef):
        self._srcdef = srcdef
        srcs, acc = self.scene.formatter.extract_sources(srcdef, self.accuracy)
        self.accuracy = acc
        if len(srcs) > 0:
            srcs = np.stack(srcs)
            self._sources = SunSetterBase(self.scene, srcs,
                                          f"{self.stype}_sources")
        else:
            self._sources = None

    def sample(self, vecf, vecs):
        """call rendering engine to sample sky contribution"""
        return super().sample(vecf, vecs).ravel()

    def run_callback(self):
        super().run_callback()
        if self._sources is not None:
            svs = SunViewSampler(self.scene, self.sources, srcdef=self._srcdef,
                                 stype=f"{self.stype}_sources",
                                 checkviz=False, plotp=False)
            svs.run()
        if not self._keepamb:
            try:
                os.remove(f'{self.scene.outdir}/{self.stype}.amb')
            except (IOError, TypeError):
                pass
