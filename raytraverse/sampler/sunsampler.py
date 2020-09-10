# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import sys

import numpy as np

from raytraverse import translate
from raytraverse.sampler import SingleSunSampler, SunViewSampler


class SunSampler(object):
    """factory class for SingleSunSamplers.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    """

    def __init__(self, scene, suns, plotp=False, **kwargs):
        self.scene = scene
        self.suns = suns
        self.plotp = plotp
        #: raytraverse.sampler.SunViewSampler
        self.viewsampler = SunViewSampler(scene, suns, plotp=False)
        #: dict: sampling arguments for SingleSunSampler
        self.sampleargs = dict(idres=4, fdres=10, speclevel=9, plotp=plotp)
        sunuv = translate.xyz2uv(self.suns.suns, flipu=False)
        #: np.array: sun bins for each sun position (used to match naming)
        self.sunbin = translate.uv2bin(sunuv, scene.skyres).astype(int)
        #: raytraverse.sampler.SingleSunSampler
        self.reflsampler = None
        self.sampleargs.update(**kwargs)

    def run(self, view=True, reflection=True, executable='rtrace',
            rcopts='-ab 6 -ad 3000 -as 1500 -st 0 -ss 16 -aa .1',
            **kwargs):
        if view and self.suns.suns.size > 0:
            print("Sampling Sun Visibility", file=sys.stderr)
            self.viewsampler.run()
        if reflection:
            for sidx, sb in enumerate(self.sunbin):
                print(f"Sampling Sun Reflections {sidx+1} of "
                      f"{self.sunbin.size}", file=sys.stderr)
                print(f'Sun Position: {self.suns.suns[sidx]}', file=sys.stderr)
                self.reflsampler = SingleSunSampler(self.scene, self.suns, sidx,
                                                    sb, **self.sampleargs)
                self.reflsampler.run()
