# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import itertools
import os

import numpy as np
from scipy.stats import norm

from raytraverse.helpers import ArrayDict, MemArrayDict
from raytraverse import translate
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

    def _mk_tree(self):
        npts = np.product(self.scene.ptshape)
        d_kds = {(-1, -1): None}
        vecs = ArrayDict({(-1, -1): None})
        omegas = ArrayDict({(-1, -1): None})
        lums = MemArrayDict({})
        for i in range(self.suns.suns.shape[0]):
            dfile = f'{self.scene.outdir}/{self.prefix}_{i:04d}_vals.out'
            if os.path.isfile(dfile):
                print(dfile)
                d_kd, vs, omega, lum = super()._mk_tree(pref=f'_{i:04d}',
                                                        ltype=list)
                for j in range(npts):
                    d_kds[(j, i)] = d_kd[j]
                    vecs[(j, i)] = vs[j]
                    omegas[(j, i)] = omega[j]
                    lums[(j, i)] = lum[j]
        return d_kds, vecs, omegas, lums

    def items(self):
        return itertools.product(super().items(),
                                 range(self.suns.suns.shape[0]))

    def add_to_img(self, img, mask, pi, i, d, coefs=1, vm=None, radius=3):
        if vm is None:
            vm = self.scene.view
        lum = self.apply_coef(pi, coefs)
        if len(i.shape) > 1:
            y = norm(scale=translate.theta2chord(radius*np.pi/180))
            w = np.broadcast_to(y.pdf(d), (lum.shape[0],) + d.shape)
            lum = np.average(lum[:, i], weights=w, axis=-1)
        else:
            lum = lum[:, i]
        img[mask] += np.squeeze(lum)
        sun = np.concatenate((self.suns.suns[pi[1]], [coefs, ]))
        self.view.add_to_img(img, pi, sun, vm)

    def get_illum(self, vm, pis, vdirs, coefs, scale=179):
        illums = []
        sun = itertools.cycle(coefs)
        for pi in pis:
            s = next(sun)[-1]
            if s > 0:
                lm = self.apply_coef(pi, s)
                idx = self.query_ball(pi, vdirs)
                for j, i in enumerate(idx):
                    v = self.vec[pi][i]
                    o = self.omega[pi][i]
                    illums.append(np.einsum('j,ij,j,->', vm.ctheta(v, j),
                                            lm[:, i], o, scale))
            else:
                illums += [0]*len(vdirs)
        illum = np.array(illums).reshape((-1, coefs.shape[0], len(vdirs)))
        illum2 = self.view.get_illum(vm, pis, coefs, scale=179)
        return illum.swapaxes(1, 2) + illum2

