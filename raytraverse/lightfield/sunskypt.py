# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from raytraverse import io
from raytraverse.lightfield.lightfieldkd import LightFieldKD


class SunSkyPt(LightFieldKD):
    """cross match vectors and samples between sun and sky data at a point
    unlike other fields does not build across points, because with many suns
    memory requirements quickly get to high as the luminance data for each
    skyfield needs to be duplicated for interpolation.

    Currently does not write to file like other LightFieldKD so needs to be
    build on the fly by the Intergrator, again to save disk space, but this
    should probably be an option.

    Parameters
    ----------
    skyfield: raytraverse.lightfield.SCBinField
        scene class containing geometry, location and analysis plane
    sunfield: raytraverse.lightfield.SunField
        sun class containing sun vectors and SunMapper (passed to SunViewField)
    ptidx: int
        point idx to build
    """

    def __init__(self, skyfield, sunfield, ptidx, save=False):
        #: raytraverse.sunsetter.SunSetter
        self.suns = sunfield.suns
        #: raytraverse.lightfield.SunViewField
        self.view = sunfield.view
        self._ptidx = ptidx
        self._sunparent = sunfield
        self._skyparent = skyfield
        self._items = [i for i in range(self.suns.suns.shape[0])
                       if (ptidx, i) in sunfield.items()]
        super().__init__(skyfield.scene, prefix=f'sunsky_{ptidx:04d}',
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

    def items(self):
        return self._items

    def ptitems(self, i):
        return [j for j in self.items() if j[0] == i]

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

    def add_to_img(self, img, mask, pi, vecs, coefs=1, vm=None, interp=1,
                   **kwargs):
        if vm is None:
            vm = self.scene.view
        super().add_to_img(img, mask, pi, vecs, coefs=coefs, interp=interp,
                           **kwargs)
        coefs = np.asarray(coefs)
        if coefs.size == 1:
            # scale to 5 times the sky irradiance (for direct view)
            coefs = coefs * 5/(np.square(0.2665 * np.pi / 180) * .5)
        sun = np.concatenate((self.suns.suns[pi],
                              np.asarray(coefs).reshape(-1)[0:1]))
        self.view.add_to_img(img, (self._ptidx, pi), sun, vm)

    def get_applied_rays(self, pi, dxyz, skyvec, sunvec=None):
        """the analog to add_to_img for metric calculations"""
        rays, omega, lum = super().get_applied_rays(pi, dxyz, skyvec)
        svw = self.view.get_ray((self._ptidx, pi), dxyz, sunvec)
        if svw is not None:
            rays = np.vstack((rays, svw[0][None, :]))
            lum = np.concatenate((lum, [svw[1]]))
            omega = np.concatenate((omega, [svw[2]]))
        return rays, omega, lum

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

    def _get_vlo(self, sidx, **kwargs):
        """

        Parameters
        ----------
        sidx: int
            sun index
        kwargs:
            passed to interpolate_query

        Returns
        -------
        vecs: np.array
        lums: np.array
        kd: scipy.spatial.cKDTree
        omega: np.array

        """
        sk_key = self._ptidx
        sk_vec = self._skyparent.vec[sk_key]
        sk_lum = self._skyparent.lum[sk_key]
        su_key = (self._ptidx, sidx)
        if su_key in self._sunparent.items():
            su_vec = self._sunparent.vec[su_key]
            su_lum = self._sunparent.lum[su_key]
            lum_sk = self._skyparent.interpolate_query(sk_key, su_vec, **kwargs)
            lum_su = self._sunparent.interpolate_query(su_key, sk_vec, **kwargs)
            vecs = np.vstack((su_vec, sk_vec))
            lum_sk = np.vstack((lum_sk, sk_lum))
            lum_su = np.vstack((su_lum, lum_su))
            lums = np.hstack((lum_su, lum_sk))
        else:
            vecs = sk_vec
            lums = sk_lum
        kd, omega = LightFieldKD.mk_vector_ball(vecs)
        return vecs, lums, kd, omega

    def _mk_tree(self, pref='', ltype=list):
        d_kds = {}
        vecs = {}
        omegas = {}
        lums = {}
        self.scene.log(self, "Interpolating kd-trees")
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            futures = []
            for sidx in self.items():
                futures.append((sidx, exc.submit(self._get_vlo, sidx)))
            for fu in futures:
                idx = fu[0]
                vecs[idx], lums[idx], d_kds[idx], omegas[idx] = fu[1].result()
        return d_kds, vecs, omegas, lums
