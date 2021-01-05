# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from raytraverse import io
from raytraverse.lightfield.lightfieldkd import LightFieldKD
from memory_profiler import profile


class SunSkyField(LightFieldKD):
    """cross match vectors and samples between a sun and sky.

    Currently does not write to file like other LightFieldKD so needs to be
    build on the fly by the Intergrator, again to save disk space, but this
    should probably be an option.

    Parameters
    ----------
    skyfield: raytraverse.lightfield.SCBinField
    sunfield: raytraverse.lightfield.LightFieldKD
    """

    def __init__(self, skyfield, sunfield, save=False):
        self._sunparent = sunfield
        self._skyparent = skyfield
        self._items = skyfield.items()
        super().__init__(skyfield.scene, prefix=f'{skyfield.prefix}'
                                                f'_{sunfield.prefix}',
                         srcn=sunfield.srcn + skyfield.srcn)
        self._d_kd, self._vec, self._omega, self._lum = self._mk_tree()

    @property
    def skyparent(self):
        return self._skyparent

    @property
    def sunparent(self):
        return self._sunparent

    def raw_files(self):
        return []

    def apply_coef(self, pi, coefs):
        coefs = np.asarray(coefs)
        if coefs.size == 1:
            coefs = np.full(self.srcn, coefs)
            # scale to 5 times the sky irradiance (for direct view)
            coefs[0] *= 5/(np.square(0.2665 * np.pi / 180) * .5)
        if np.mod(coefs.size, self.srcn) == 0:
            c = coefs.reshape(-1, self.srcn)
        else:
            c = np.broadcast_to(coefs, (coefs.size, self.srcn))
        return np.einsum('ij,kj->ik', c, self.lum[pi])

    @property
    def scene(self):
        """scene information

        :getter: Returns this integrator's scene
        :setter: Set this integrator's scene
        :type: raytraverse.scene.Scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        """Set this field's scene and load samples"""
        self._scene = scene

    @staticmethod
    def cross_interpolate(sk_kd, sk_vec, sk_lum, su_kd, su_vec, su_lum, idx):
        lum_sk = LightFieldKD.interpolate_query(sk_kd, sk_lum, sk_vec, su_vec, k=1)
        lum_su = LightFieldKD.interpolate_query(su_kd, su_lum, su_vec, sk_vec, k=1)
        vecs = np.vstack((su_vec, sk_vec))
        lum_sk = np.vstack((lum_sk, sk_lum))
        lum_su = np.vstack((su_lum, lum_su))
        lums = np.hstack((lum_su, lum_sk))
        kd, omega = LightFieldKD.mk_vector_ball(vecs)
        return idx, vecs, lums, kd, omega

    def _mk_tree(self, pref='', ltype=list):
        d_kds = {}
        vecs = {}
        omegas = {}
        lums = {}
        self.scene.log(self, "Interpolating kd-trees")
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            futures = []
            idxs = list(self.items())
            chunks = list(range(0, len(idxs), 100)) + [None]
            for c0, c1 in zip(chunks[:-1], chunks[1:]):
                for idx in idxs[c0:c1]:
                    sk_vec = self._skyparent.vec[idx]
                    sk_lum = self._skyparent.lum[idx]
                    sk_kd = self._skyparent.d_kd[idx]
                    if (idx in self._sunparent.d_kd and
                        self._sunparent.d_kd[idx] is not None):
                        su_vec = self._sunparent.vec[idx]
                        su_lum = self._sunparent.lum[idx]
                        su_kd = self._sunparent.d_kd[idx]
                        futures.append(exc.submit(SunSkyField.cross_interpolate,
                                                  sk_kd, sk_vec, sk_lum, su_kd,
                                                  su_vec, su_lum, idx))
                    else:
                        vecs[idx] = sk_vec
                        lums[idx] = sk_lum
                        d_kds[idx] = self._skyparent.d_kd[idx]
                        omegas[idx] = self._skyparent.omega[idx]
                for fu in as_completed(futures):
                    result = fu.result()
                    idx = result[0]
                    vecs[idx], lums[idx], d_kds[idx], omegas[idx] = result[1:]
                self.scene.log(self, "Interpolated kd-trees for points "
                                     f"{c0}-{c1}")
        return d_kds, vecs, omegas, lums
