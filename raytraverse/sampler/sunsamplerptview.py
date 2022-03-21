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

from raytraverse.mapper import ViewMapper
from raytraverse.sampler.samplerpt import SamplerPt
from raytraverse.lightpoint import LightPointKD, SrcViewPoint


class SunSamplerPtView(SamplerPt):
    """sample view rays to a source.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    sun: np.array
        the direction to the source
    sunbin: int
        index for naming
    """
    #: deterministic sample draws
    ub = 1

    def __init__(self, scene, engine, sun, sunbin, **kwargs):
        super().__init__(scene, engine, stype=f"sunview_{sunbin:04d}", idres=16,
                         nlev=3, **kwargs)
        self.sunpos = np.asarray(sun).flatten()[0:3]
        # load new source
        f, srcdef = tempfile.mkstemp(dir=f"./{scene.outdir}/", prefix='tmp_src')
        # srcdef = f'{scene.outdir}/tmp_srcdef_{sunbin}.rad'
        f = open(srcdef, 'w')
        f.write(scene.formatter.get_sundef(sun, (1, 1, 1)))
        f.close()
        self.engine.load_source(srcdef)
        os.remove(srcdef)
        self.vecs = None
        self.lum = []

    def run(self, point, posidx, vm=None, plotp=False, log=None, **kwargs):
        if vm is None:
            vm = ViewMapper(self.sunpos, 0.533, "sunview", jitterrate=0)
        if hasattr(vm, "dxyz"):
            return super().run(point, posidx, vm, plotp=plotp, log=log,
                               **kwargs)
        svpts = []
        args = self.engine.args
        # temporarily override arguments
        self.engine.set_args(self.engine.directargs)
        for v in vm:
            svpts.append(super().run(point, posidx, v, plotp=plotp, log=log,
                                     **kwargs))
        self.engine.set_args(args)
        return svpts

    def _run_callback(self, point, posidx, vm, write=False, **kwargs):
        """post sampling, write full resolution (including interpolated values)
         non zero rays to result file."""
        keep = self.lum > 1e-7
        skd = LightPointKD(self.scene, self.vecs, self.lum, vm, point, posidx,
                           self.stype, calcomega=False, write=write)
        if np.sum(keep) == 0:
            lightpoint = None
        else:
            shp = self.weights.shape
            si = np.stack(np.unravel_index(np.arange(np.product(shp)), shp))
            uv = (si.T + .5)/shp[1]
            grid = vm.uv2xyz(uv)
            i = skd.query_ray(grid)[0]
            lumg = skd.lum[i, 0]
            keep = lumg > 1e-8
            lightpoint = SrcViewPoint(self.scene, grid[keep],
                                      np.average(lumg[keep]), point, posidx,
                                      self.stype, shp[1], vm.area)
        self.vecs = None
        self.lum = []
        return lightpoint

