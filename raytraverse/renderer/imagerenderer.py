# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from raytraverse import io, translate
from raytraverse.mapper import ViewMapper
from raytraverse.renderer.renderer import Renderer


class ImageRenderer(Renderer):
    """interface to treat image data as the source for ray tracing results"""

    vm = ViewMapper(viewangle=180)
    name = "ImageRenderer"

    @classmethod
    def initialize(cls, args, scene, nproc=None, viewmapper=None, **kwargs):
        if not cls.initialized:
            if viewmapper is not None:
                cls.vm = viewmapper
            cls.scene = scene
            cls.scene = io.hdr2array(scene).T
            x = np.arange(cls.scene.shape[0]) + .5
            cls.instance = RegularGridInterpolator((x, x), cls.scene,
                                                   bounds_error=False)
        cls.initialized = True

    @classmethod
    def call(cls, rays, store=True, outf=None):
        pxy = cls.vm.ray2pixel(rays[:, 3:6], cls.scene.shape[0], integer=False)
        return cls.instance(pxy)

