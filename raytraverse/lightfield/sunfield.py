# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from raytraverse.helpers import ArrayDict, sunfield_load_item
from raytraverse.lightfield.lightfield import LightField
from raytraverse.lightfield.sunviewfield import SunViewField


class SunField(LightField):
    """container for sun view data

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun vectors and SunMapper (passed to SunViewField)
    rebuild: bool, optional
        build kd-tree even if one exists
    """

    nullidx = np.array([-1,], dtype=int)

    def __init__(self, scene, suns, rebuild=False):
        #: np.array: sun positions
        self.suns = suns.suns
        #: raytraverse.lightfield.SunViewField
        self.view = SunViewField(scene, suns, rebuild=rebuild)
        self.draw_sun = self.view.draw_sun
        super().__init__(scene, rebuild=rebuild, prefix='sun')

    @property
    def vlo(self):
        """sun data indexed by (point, sun)

        key (i, j) val: direction vector (3,) luminance (srcn,), omega (1,)

        :type: raytraverse.helpers.ArrayDict
        """
        return self._vlo

    def items(self):
        return itertools.product(super().items(), range(self.suns.shape[0]))

    def _mk_tree(self):
        npts = np.product(self.scene.ptshape)
        vlamb = self._get_vl(npts, pref='ambient')
        d_kd = {(-1, -1): None}
        vlo = ArrayDict({(-1, -1): None})
        fu = []
        with ProcessPoolExecutor() as exc:
            for i in range(self.suns.shape[0]):
                vlsun = self._get_vl(npts, pref=f'_{i:04d}')
                for j in range(npts):
                    fu.append(exc.submit(sunfield_load_item, vlamb[j],
                                         vlsun[j], i, j, self.scene.maxspec))
        for future in as_completed(fu):
            idx, v, d = future.result()
            vlo[idx] = v
            d_kd[idx] = d
        return d_kd, vlo

    def measure(self, psi, vecs, coefs=1e8, interp=1):
        d, i = self.d_kd[psi].query(vecs, k=interp)
        c = np.asarray(coefs).reshape(-1)
        lum = np.einsum('i,j->ij', c, self.vlo[psi][:, 3])
        if interp > 1:
            wgts = np.broadcast_to(1/d, (lum.shape[0],) + d.shape)
            lum = np.average(lum[:, i], weights=wgts, axis=-1)
        else:
            lum = lum[:, i]
        return np.squeeze(lum)
