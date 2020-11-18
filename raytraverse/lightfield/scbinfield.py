# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.lightfield.lightfieldkd import LightFieldKD
from raytraverse import translate

class SCBinField(LightFieldKD):
    """container for accessing sampled data where every ray has a value for
    each source
    """

    def __init__(self, scene, rebuild=False, prefix='sky', ground=True,
                 **kwargs):
        super().__init__(scene, rebuild=rebuild, prefix=prefix,
                         srcn=scene.skyres**2 + ground, **kwargs)

    def apply_coef(self, pi, coefs):
        coefs = np.asarray(coefs)
        if np.mod(coefs.size, self.srcn) == 0:
            c = coefs.reshape(-1, self.srcn)
        else:
            c = np.broadcast_to(coefs, (coefs.size, self.srcn))
        return np.einsum('ij,kj->ik', c, self.lum[pi])

    def direct_view(self, res=512, showsample=True, showweight=True,
                    dpts=None, items=None, srcidx=None):
        """create a summary image of lightfield for each vpt"""
        if srcidx is not None and type(srcidx) is not int:
            uv = translate.xyz2uv(np.asarray(srcidx)[None, :], flipu=False)
            uvi = np.linspace(-.016667, .016667, 3)
            uvs = np.stack(np.meshgrid(uvi, uvi)).reshape(2, 9).T + uv
            srcidx = np.unique(translate.uv2bin(uvs,
                                                self.scene.skyres).astype(int))
        super().direct_view(res=res, showsample=showsample,
                            showweight=showweight, dpts=dpts, items=items,
                            srcidx=srcidx)
