# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse.mapper import ViewMapper
from raytraverse.sampler import Sampler
from raytraverse.lightpoint import LightPointKD, SunViewPoint


class SunViewSampler(Sampler):
    """sample view rays to direct suns.

    here idres and fdres are sampled on a per sun basis for a view centered
    on each sun direction with a view angle of .533 degrees (hardcoded in
    sunmapper class).

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    loadsrc: bool
        include suns.rad in base scene initialization. if False,
        self.engine.load_source must be invoked before call.
    """
    #: deterministic sample draws
    ub = 1

    def __init__(self, scene, engine, sun, sunbin, **kwargs):
        super().__init__(scene, engine, stype=f"sunview_{sunbin:04d}", idres=4,
                         fdres=6, **kwargs)
        self.sunpos = np.asarray(sun).flatten()[0:3]
        # load new source
        srcdef = f'{scene.outdir}/tmp_srcdef_{sunbin}.rad'
        f = open(srcdef, 'w')
        f.write(scene.formatter.get_sundef(sun, (1, 1, 1)))
        f.close()
        self.engine.load_source(srcdef)
        os.remove(srcdef)
        self.vecs = None
        self.lum = []

    def _offset(self, shape, dim):
        """no jitter on sun view because of very fine resolution and potentially
        large number of samples bog down random number generator"""
        return 0.5/dim

    def run_callback(self, point, posidx, vm):
        """post sampling, write full resolution (including interpolated values)
         non zero rays to result file."""
        skd = LightPointKD(self.scene, self.vecs, self.lum, vm, point, posidx,
                           self.stype, calcomega=False, write=False)
        shp = self.weights.shape
        si = np.stack(np.unravel_index(np.arange(np.product(shp)), shp))
        uv = (si.T + .5)/shp[1]
        grid = vm.uv2xyz(uv)
        # print(grid.shape)
        i = skd.query_ray(grid)[0]
        lumg = skd.lum[i, 0]
        keep = lumg > 1e-8
        if np.sum(keep) > 0:
            lightpoint = SunViewPoint(self.scene, grid[keep],
                                      np.average(lumg[keep]), point, posidx,
                                      self.stype, shp[1])
        else:
            lightpoint = None
        return lightpoint

    def run(self, point, posidx, vm=None, plotp=False, **kwargs):
        vm = ViewMapper(self.sunpos, 0.533, "sunview")
        return super().run(point, posidx, vm, plotp, **kwargs)
