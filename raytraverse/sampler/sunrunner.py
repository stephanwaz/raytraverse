# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate
from raytraverse.sampler import SingleSunSampler, SunViewSampler, SunSampler

from memory_profiler import profile


class SunRunner(object):
    """factory class for SingleSunSamplers.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    """

    def __init__(self, scene, suns, **kwargs):
        self.scene = scene
        self.suns = suns
        self.viewsampler = SunViewSampler(scene, suns)
        self.sampleargs = dict(idres=10, fdres=12, maxrate=.01, minrate=.005)
        sunuv = translate.xyz2uv(self.suns.suns)
        scheme = np.load(f'{self.scene.outdir}/sky_scheme.npy').astype(int)
        side = int(np.sqrt(scheme[0, 4]))
        self.sunbin = translate.uv2bin(sunuv, side).astype(int)
        self.basesampler = SunSampler(self.scene, self.suns, idres=4, fdres=6,
                                      maxrate=1, minrate=.1)
        self.reflsampler = None
        self.sampleargs.update(**kwargs)

    def run(self, **skwargs):
        print("Sampling Sun Visibility")
        self.viewsampler.run(rcopts='-ab 0')
        # print("Sampling Ambient Direct Sun contribution")
        # self.basesampler.run(**skwargs)
        # for sidx, sb in enumerate(self.sunbin):
        #     print(f"Sampling Sun Reflections {sidx+1} of {self.sunbin.size}")
        #     self.reflsampler = SingleSunSampler(self.scene, self.suns, sidx, sb,
        #                                         **self.sampleargs)
        #     self.reflsampler.run(**skwargs)
