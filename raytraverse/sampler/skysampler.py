# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np
import re
from clasp.script_tools import pipeline

from raytraverse import renderer, io
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
    accuracy: float, optional
        final sampling level will set # of samples by the number of
        samples with variance greater than 1/4 this number, which has
        units of radiance for bright sky conditions this should be set to
        a correspondingly higher value (default is 6.0).
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
        self.engine = renderer.Rtrace()
        engine_args = f"{rcopts} -oZ"
        # update ambient file and args before init
        self._keepamb = keepamb and ambcache
        self.skyname = skyname
        if ambcache:
            engine_args += f" -af {scene.outdir}/{skyname}.amb"
        super().__init__(scene, stype=skyname, fdres=fdres,
                         engine_args=engine_args,
                         srcdef=srcdef, **kwargs)
        self.engine.load_source(srcdef)
        self.sources = srcdef

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, srcdef):
        self._srcdef = srcdef
        srcs = []
        srctxt = pipeline([f"xform {srcdef}"])
        srclines = re.split(r"[\n\r]+", srctxt)
        for i, v in enumerate(srclines):
            if re.match(r"[\d\w]+\s+source\s+[\d\w]+", v):
                src = " ".join(srclines[i:]).split()
                srcd = np.array(src[6:10], dtype=float)
                if srcd[-1] < 3:
                    modsrc = " ".join(srclines[:i]).split()
                    modidx = next(j for j in reversed(range(len(modsrc)))
                                  if modsrc[j] == src[0])
                    modi = io.rgb2rad(np.array(modsrc[modidx+4:modidx+7],
                                               dtype=float))
                    srcs.append(np.concatenate((srcd, [modi])))
                    # 1/(np.square(0.2665 * np.pi / 180) * .5) = 92444
                    # the ratio of suns area to hemisphere
                    self.accuracy = self.accuracy*modi/92444
                    break
        if len(srcs) > 0:
            srcs = np.stack(srcs)
            self._sources = SunSetterBase(self.scene, srcs,
                                          f"{self.skyname}_sources")
        else:
            self._sources = None

    def sample(self, vecf):
        """call rendering engine to sample sky contribution

        Parameters
        ----------
        vecf: str
            path of file name with sample vectors
            shape (N, 6) vectors in binary float format

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        return super().sample(vecf).ravel()

    def run_callback(self):
        super().run_callback()
        if self._sources is not None:
            svs = SunViewSampler(self.scene, self.sources, srcdef=self._srcdef,
                                 stype=f"{self.skyname}_sources",
                                 checkviz=False, plotp=False)
            svs.run()
        if not self._keepamb:
            try:
                os.remove(f'{self.scene.outdir}/{self.skyname}.amb')
            except (IOError, TypeError):
                pass
