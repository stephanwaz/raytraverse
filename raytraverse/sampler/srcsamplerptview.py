# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import io
from raytraverse.sampler.samplerpt import SamplerPt
from raytraverse.lightpoint import LightPointKD, SrcViewPoint


class SrcSamplerPtView(SamplerPt):
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

    def __init__(self, scene, engine, nlev=3, idres=16, **kwargs):
        super().__init__(scene, engine, idres=idres,
                         nlev=nlev, **kwargs)
        self.vecs = None
        self.lum = []

    def run(self, point, posidx, vm=None, plotp=False, log=None, **kwargs):
        if vm is None:
            return []
        svpts = []
        args = self.engine.args
        # temporarily override arguments
        self.engine.set_args(self.engine.directargs)
        for v in vm:
            svpts.append(super().run(point, posidx, v, plotp=plotp, log=log,
                                     **kwargs))
        self.engine.set_args(args, io.get_nproc())
        return svpts

    def _run_callback(self, point, posidx, vm, write=False, **kwargs):
        """post sampling, write full resolution (including interpolated values)
         non-zero rays to result file."""
        if np.sum(self.lum > 1e-7) == 0:
            lightpoint = None
        else:
            skd = LightPointKD(self.scene, self.vecs, self.lum, vm, point,
                               posidx, self.stype, calcomega=False, write=write)
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

