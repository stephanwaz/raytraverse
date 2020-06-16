# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np

from raytraverse import Integrator


class SkyIntegrator(Integrator):
    """loads sky sampling data for processing

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    rebuild: bool, optional
        build kd-tree even if one exists
    """

    def __init__(self, scene, **kwargs):
        kwargs.update(prefix='sky')
        super(SkyIntegrator, self).__init__(scene, **kwargs)
        print([l.shape for l in self.lum])

    def apply_coefs(self, pis, coefs=None):
        if coefs is None:
            lum = [np.sum(self.lum[pi], 1).reshape(1, -1) for pi in pis]
        else:
            cnt = self.lum[pis[0]].shape[1]
            skyvecs = coefs.reshape(-1, cnt).T
            lum = [(self.lum[pi].reshape(-1, cnt)@skyvecs).T for pi in pis]
            print([l.shape for l in lum])
        return lum

