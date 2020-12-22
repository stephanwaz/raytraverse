# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

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

    def __init__(self, scene, suns, plotp=False, checkviz=True, **kwargs):
        scene.log(self, "Initializing")
        self.scene = scene
        self.suns = suns
        #: raytraverse.sampler.SunViewSampler
        self.viewsampler = SunViewSampler(scene, suns, checkviz=checkviz,
                                          plotp=False)
        #: dict: sampling arguments for SingleSunSampler
        self.sampleargs = dict(idres=4, fdres=10, speclevel=9, plotp=plotp)
        #: raytraverse.sampler.SingleSunSampler
        self.reflsampler = None
        self.sampleargs.update(**kwargs)

    def run(self, view=True, reflection=True, replace=True):
        if view and self.suns.suns.size > 0:
            self.viewsampler.run()
        if reflection:
            suncnt = self.suns.suns.shape[0]
            aa = translate.xyz2aa(self.suns.suns)
            for sidx in range(suncnt):
                self.scene.log(self, f"Sampling Sun Reflections {sidx+1} of "
                               f"{suncnt}")
                self.scene.log(self, f'Sun Position: alt={aa[sidx, 0]:.01f},'
                               f' az={aa[sidx, 1]:.01f}')
                vals = f'{self.scene.outdir}/sun_{sidx:04d}_vals.out'
                vecs = f'{self.scene.outdir}/sun_{sidx:04d}_vecs.out'
                if (not replace) and (os.path.isfile(vals)
                                      and os.path.isfile(vecs)):
                    self.scene.log(self, f"Output files for Sun Reflections "
                                         f"{sidx+1} exist, skipping "
                                         f"(use replace=True to overwrite)")
                else:
                    self.reflsampler = SingleSunSampler(self.scene, self.suns,
                                                        sidx, **self.sampleargs)
                    self.reflsampler.run()
