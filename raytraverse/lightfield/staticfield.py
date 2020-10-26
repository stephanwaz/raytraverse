# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import numpy as np

from raytraverse.lightfield.lightfieldkd import LightFieldKD
from raytraverse.lightfield.sunviewfield import SunViewField
from raytraverse.scene import SunSetterBase


class StaticField(LightFieldKD):
    """container for accessing sampled data for a single sky condition that
    may or may not include direct sources
    """

    def __init__(self, scene, sources=None, rebuild=False, prefix='static',
                 rmraw=False, **kwargs):
        if sources is not None:
            self.suns = sources
        elif os.path.isfile(f"{scene.outdir}/{prefix}_sources.rad"):
            self.suns = SunSetterBase(scene, prefix=f"{prefix}_sources")
        else:
            fvrays = 0
            self.suns = None
            self.view = None
        if self.suns is not None:
            fvrays = self.suns.srct*scene.maxspec
            self.view = SunViewField(scene, self.suns, rebuild=rebuild,
                                     rmraw=rmraw, prefix=f"{prefix}_sources")
        super().__init__(scene, rebuild=rebuild, prefix=prefix, rmraw=rmraw,
                         fvrays=fvrays, **kwargs)

    def add_to_img(self, img, mask, pi, vecs, coefs=1, vm=None, interp=1,
                   **kwargs):
        if vm is None:
            vm = self.scene.view
        super().add_to_img(img, mask, pi, vecs, coefs=coefs, interp=interp,
                           **kwargs)
        if self.view is not None:
            for i in range(len(self.suns.suns)):
                sun = np.concatenate((self.suns.suns[i], [coefs, ]))
                self.view.add_to_img(img, (pi, i), sun, vm)

    def get_applied_rays(self, pi, dxyz, skyvec, sunvec=None):
        """the analog to add_to_img for metric calculations"""
        rays, omega, lum = super().get_applied_rays(pi, dxyz, skyvec)
        if self.view is not None:
            for i in range(len(self.suns.suns)):
                sun = np.concatenate((self.suns.suns[i], [skyvec, ]))
                svw = self.view.get_ray((pi, i), dxyz, sun)
                if svw is not None:
                    rays = np.vstack((rays, svw[0][None, :]))
                    lum = np.concatenate((lum, [svw[1]]))
                    omega = np.concatenate((omega, [svw[2]]))
        return rays, omega, lum
