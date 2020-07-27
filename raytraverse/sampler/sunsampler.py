# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
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

    def __init__(self, scene, suns, **kwargs):
        self.scene = scene
        self.suns = suns
        #: raytraverse.sampler.SunViewSampler
        self.viewsampler = SunViewSampler(scene, suns)
        #: dict: sampling arguments for SingleSunSampler
        self.sampleargs = dict(idres=4, fdres=10, speclevel=9, accuracy=.01)
        sunuv = translate.xyz2uv(self.suns.suns)
        scheme = np.load(f'{self.scene.outdir}/sky_scheme.npy').astype(int)
        side = int(np.sqrt(scheme[0, 4]))
        #: np.array: sun bins for each sun position (used to match naming)
        self.sunbin = translate.uv2bin(sunuv, side).astype(int)
        #: raytraverse.sampler.SingleSunSampler
        self.reflsampler = None
        self.sampleargs.update(**kwargs)

    def run(self, view=True, reflection=True, rcopts='', **kwargs):
        if view:
            print("Sampling Sun Visibility")
            self.viewsampler.run(rcopts='-ab 0')
        if reflection:
            for sidx, sb in enumerate(self.sunbin):
                print(f"Sampling Sun Reflections {sidx+1} of "
                      f"{self.sunbin.size}")
                print(f'Sun Position: {self.suns.suns[sidx]}')
                self.reflsampler = SingleSunSampler(self.scene, self.suns, sidx,
                                                    sb, **self.sampleargs)
                self.reflsampler.run(rcopts=rcopts, **kwargs)
