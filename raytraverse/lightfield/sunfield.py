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

from raytraverse import io
from raytraverse.helpers import ArrayDict, sunfield_load_item
from raytraverse.lightfield.lightfieldkd import LightFieldKD
from raytraverse.lightfield.sunviewfield import SunViewField


class SunField(LightFieldKD):
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

    def __init__(self, scene, suns, rebuild=False):
        #: raytraverse.sunsetter.SunSetter
        self.suns = suns
        #: raytraverse.lightfield.SunViewField
        self.view = SunViewField(scene, suns, rebuild=rebuild)
        super().__init__(scene, rebuild=rebuild, prefix='sun')

    @property
    def vlo(self):
        """sun data indexed by (point, sun)

        key (i, j) val: direction vector (3,) luminance (srcn,), omega (1,)

        :type: raytraverse.helpers.ArrayDict
        """
        return self._vlo

    def _mk_tree(self):
        npts = np.product(self.scene.ptshape)
        vlamb = self._get_vl(npts, pref='ambient')
        d_kd = {(-1, -1): None}
        vlo = ArrayDict({(-1, -1): None})
        fu = []
        with ProcessPoolExecutor() as exc:
            for i in range(self.suns.suns.shape[0]):
                vlsun = self._get_vl(npts, pref=f'_{i:04d}')
                for j in range(npts):
                    fu.append(exc.submit(sunfield_load_item, vlamb[j],
                                         vlsun[j], i, j, self.scene.maxspec))
        for future in as_completed(fu):
            idx, v, d = future.result()
            vlo[idx] = v
            d_kd[idx] = d
        return d_kd, vlo

    def items(self):
        return itertools.product(super().items(),
                                 range(self.suns.suns.shape[0]))

    def apply_coef(self, pi, coefs):
        c = np.asarray(coefs).reshape(-1, 1)
        return super().apply_coef(pi, c)

    def _dview(self, idx, pdirs, mask, res=800):
        img = np.zeros((res, res*self.scene.view.aspect))
        i, d = self.query_ray(idx, pdirs[mask])
        self.add_to_img(img, mask, idx, i, d)
        sun = np.concatenate((self.suns.suns[idx[1]], [1,]))
        self.view.add_to_img(img, idx, sun, self.scene.view)
        outf = f"{self.outfile(idx)}.hdr"
        io.array2hdr(img, outf)
        return outf

    def get_illum(self, vm, pis, vdirs, coefs, scale=179):
        illums = []
        sun = itertools.cycle(coefs)
        for pi in pis:
            s = next(sun)[-1]
            if s > 0:
                lm = self.apply_coef(pi, s)
                idx = self.query_ball(pi, vdirs)
                for j, i in enumerate(idx):
                    v = self.vlo[pi][i, 0:3]
                    o = self.vlo[pi][i, -1]
                    illums.append(np.einsum('j,ij,j,->', vm.ctheta(v, j),
                                            lm[:, i], o, scale))
            else:
                illums += [0]*len(vdirs)
        illum = np.array(illums).reshape((-1, coefs.shape[0], len(vdirs)))
        illum2 = self.view.get_illum(vm, pis, coefs, scale=179)
        return illum.swapaxes(1, 2) + illum2

