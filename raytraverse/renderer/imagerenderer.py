# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from raytraverse import io
from raytraverse.mapper import ViewMapper


class ImageRenderer:
    """interface to treat image data as the source for ray tracing results

    not implemented as a singleton, so multiple instances can exist in
    parallel.

    Parameters
    ----------
    scene: str
        path to hdr image file with projecting matching ViewMapper
    viewmapper: raytraverse.mapper.ViewMapper, optional
        if None, assumes 180 degree angular fisheye (vta)
    method: str, optional
        passed to scipy.interpolate.RegularGridInterpolator
    """

    def __init__(self, scene, viewmapper=None, method="linear", color=False):
        self.srcn = 1
        if viewmapper is None:
            self.vm = ViewMapper(viewangle=180)
        else:
            self.vm = viewmapper
        if color:
            self.features = 3
            self.scene = io.hdr2carray(scene)
        else:
            self.features = 1
            self.scene = io.hdr2array(scene)
        fimg = self.scene[..., -1::-1].T
        res = fimg.shape[0]
        of = .5/res
        self.args = f"interpolation: {method}"
        x = np.linspace(of, 1-of, res)
        fv = np.median(np.concatenate((fimg[0], fimg[-1], fimg[:, 0],
                                       fimg[:, -1]), 0), 0)
        self.instance = RegularGridInterpolator((x, x), fimg,
                                                bounds_error=False,
                                                method=method,
                                                fill_value=fv)

    def __call__(self, rays):
        """tranforms rays to 2-D image space before calling interpolator

        Parameters
        ----------
        rays: np.array

        Returns
        -------
        np.array

        """
        pxy = self.vm.xyz2vxy(rays[:, 3:6])
        return self.instance(pxy)

    def run(self, *args, **kwargs):
        """alias for call, for consistency with SamplerPt classes for nested
        dimensions of evaluation"""
        return self(args[0])
