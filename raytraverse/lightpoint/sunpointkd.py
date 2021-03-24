# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import pickle
import numpy as np
from scipy.spatial import cKDTree
from clasp.script_tools import try_mkdir
from raytraverse import translate
from raytraverse.lightpoint.lightpointkd import LightPointKD


class SunPointKD(LightPointKD):
    """removes stray rays from accidental direct sun hits during build

    Parameters
    ----------
    scene: raytraverse.scene.Scene
    vec: np.array, optional
    lum: np.array, optional
    sun: tuple np.array list, optional
    sunview: raytraverse.lightpoint.SunViewPoint

    """

    def __init__(self, scene, vec=None, lum=None, sun=(0, 0, 0), sunview=None,
                 filterview=True, **kwargs):
        self.sunpos = translate.norm1(np.asarray(sun).flatten()[0:3])
        self.sunview = sunview
        self._filterview = filterview
        super().__init__(scene, vec, lum, **kwargs)

    def load(self):
        f = open(self.file, 'rb')
        loads = pickle.load(f)
        self._d_kd, self._vec, self._omega, self._lum = loads[0:4]
        try:
            self.sunview = loads[4]
        except IndexError:
            pass
        f.close()

    @property
    def srcdir(self):
        return self.sunpos.reshape(1, 3)

    def dump(self):
        try_mkdir(f"{self.scene.outdir}/{self.src}")
        f = open(self.file, 'wb')
        pickle.dump((self._d_kd, self._vec, self._omega, self._lum,
                     self.sunview), f, protocol=4)
        f.close()

    def add_to_img(self, img, vecs, mask=None, skyvec=1, interp=False,
                   omega=False, vm=None, rnd=False):
        """add luminance contributions to image array (updates in place).
        adds sunview if it exists.

        Parameters
        ----------
        img: np.array
            2D image array to add to (either zeros or with other source)
        vecs: np.array
            vectors corresponding to img pixels shape (N, 3)
        mask: np.array, optional
            indices to img that correspond to vec (in case where whole image
            is not being updated, such as corners of fisheye)
        skyvec: int float np.array, optional
            source coefficients, shape is (1,) or (srcn,)
        interp: bool, optional
            for linear interpolation (falls back to nearest outside of
            convexhull
        omega: bool
            if true, add value of ray solid angle instead of luminance
        vm: raytraverse.mapper.ViewMapper, optional
        """
        skyvec = np.atleast_1d(skyvec)
        if vm is None:
            vm = self.vm
        super(SunPointKD, self).add_to_img(img, vecs, mask, skyvec, interp,
                                           omega, vm, rnd)
        if self.sunview is not None:
            self.sunview.add_to_img(img, vecs, mask, skyvec[-1], vm)

    def get_applied_rays(self, skyvec, vm=None):
        """return rays within view with skyvec applied. this is the
        analog to add_to_img for metric calculations. includes sunview ray if
        it exists.

        Parameters
        ----------
        skyvec: int float np.array, optional
            source coefficients, shape is (1,) or (srcn,)
        vm: raytraverse.mapper.ViewMapper, optional

        Returns
        -------
        rays: np.array
            shape (N, 3) rays falling within view
        omega: np.array
            shape (N,) associated solid angles
        lum: np.array
            shape (N,) associated luminances
        """
        skyvec = np.atleast_1d(skyvec)
        if vm is None:
            vm = self.vm
        rays, omega, lum = super(SunPointKD, self).get_applied_rays(skyvec, vm)
        print(lum.shape)
        if self.sunview is not None:
            vr, vo, vl = self.sunview.get_applied_rays(skyvec[-1], vm)
            rays = np.concatenate((rays, [vr]), 0)
            omega = np.concatenate((omega, [vo]), 0)
            lum = np.concatenate((lum, [vl]), 0)
        return rays, omega, lum

    def _build(self, vec, lum, srcn):
        """load samples and build data structure
        remove lucky hits of direct sun (since these are accounted for
        by the SunViewSampler)"""
        d_kd, vec, lum = super()._build(vec, lum, srcn)
        if self._filterview:
            broken_clock = d_kd.query_ball_point(self.sunpos, 0.0046513)
            if len(broken_clock) > 0:
                vec = np.delete(vec, broken_clock, 0)
                lum = np.delete(lum, broken_clock, 0)
                d_kd = cKDTree(vec)
        return d_kd, vec, lum

    def compress(self, src=None, write=False, dist=0.19603428, lweight=10,
                 **kwargs):
        return super().compress(src, write, dist, lweight, sun=self.sunpos,
                                sunview=self.sunview, filterview=False)
