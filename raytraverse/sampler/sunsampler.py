# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
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
        #: raytraverse.sampler.SunViewSampler
        self.viewsampler = SunViewSampler(scene, suns, plotp=False)
        #: dict: sampling arguments for SingleSunSampler
        self.sampleargs = dict(idres=4, fdres=10, speclevel=9, plotp=plotp)
        #: raytraverse.sampler.SingleSunSampler
        self.reflsampler = None
        self.sampleargs.update(**kwargs)

    def run(self, view=True, reflection=True):
        if view and self.suns.suns.size > 0:
            print("Sampling Sun Visibility", file=sys.stderr)
            self.viewsampler.run()
        if reflection:
            suncnt = self.suns.suns.shape[0]
            aa = translate.xyz2aa(self.suns.suns)
            for sidx in range(suncnt):
                print(f"Sampling Sun Reflections {sidx+1} of "
                      f"{suncnt}", file=sys.stderr)
                print(f'Sun Position: alt={aa[sidx, 0]:.01f},'
                      f' az={aa[sidx, 1]:.01f}', file=sys.stderr)
                self.reflsampler = SingleSunSampler(self.scene, self.suns, sidx,
                                                    **self.sampleargs)
                self.reflsampler.run()
