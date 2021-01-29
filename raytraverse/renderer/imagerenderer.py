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


class ImageRenderer(object):
    """interface to treat image data as the source for ray tracing results"""

    def __init__(self):
        self.name = "ImageRenderer"
        self.initialized = False
        self.instance = None
        self.Engine = None
        self.scene = None
        self.header = ""
        self.arg_prefix = ''
        self.vm = ViewMapper(viewangle=180)

    def initialize(self, args, scene, viewmapper=None, method="linear",
                   **kwargs):
        if viewmapper is not None:
            self.vm = viewmapper
        self.scene = io.hdr2array(scene)
        res = self.scene.shape[0]
        of = 1/res
        x = np.linspace(-1+of, 1-of, res)
        fv = np.median(np.concatenate((self.scene[0], self.scene[-1],
                                       self.scene[:,0], self.scene[:,-1])))
        self.instance = RegularGridInterpolator((x, x),
                                                self.scene[:, -1::-1].T,
                                                bounds_error=False,
                                                method=method,
                                                fill_value=fv)
        self.initialized = True

    def call(self, rays, store=True, outf=None):
        pxy = self.vm.xyz2xy(rays[:, 3:6])
        return self.instance(pxy)

    @classmethod
    def reset(cls):
        pass

    @classmethod
    def reset_instance(cls):
        pass
