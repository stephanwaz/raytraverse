# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np

from raytraverse import Integrator


class SunIntegrator(Integrator):
    """loads sky sampling data for processing

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    rebuild: bool, optional
        build kd-tree even if one exists
    """

    def __init__(self, scene, **kwargs):
        kwargs.update(prefix='sun')
        super(SunIntegrator, self).__init__(scene, **kwargs)

    def apply_coefs(self, pis, coefs=None):
        cnt = self.lum[pis[0]].shape[1]
        if coefs is None:
            lum = [np.sum(self.lum[pi], (1, 2)).reshape(1, -1) for pi in pis]
        elif coefs.shape[-1] == cnt:
            skyvecs = coefs.reshape(-1, cnt).T
            lum = [(self.lum[pi].reshape(-1, cnt)@skyvecs).T for pi in pis]
        else:
            coefs = coefs.reshape(-1, 2).T
            bins = coefs[0].astype(int)
            c = coefs[1]
            print(bins, c)
            lum = [(self.lum[pi].reshape(-1, cnt)[:, bins]*c).T for pi in pis]
            print(lum)
        return lum

