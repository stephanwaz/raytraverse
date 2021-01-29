# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np

from raytraverse import translate, plot
from raytraverse.mapper import ViewMapper
from scipy.interpolate import LinearNDInterpolator


class SunViewPoint(object):
    """interface for sun view data"""

    @staticmethod
    def _smudge(cnt, omegap, omegasp):
        """hack to ensure equal energy and max luminance)"""
        ocnt = cnt - (omegap/omegasp)
        smdg = np.sum(ocnt[ocnt > 0])
        cnt[ocnt > 0] = omegap[ocnt > 0]/omegasp
        # average to redistribute
        redist = smdg/np.sum(ocnt < 0)
        # redistribute over pixels with "room" (this could still
        # overshoot if too many pixels are close to threshold, but
        # maybe mathematically impossible?
        cnt[ocnt < -redist] += smdg/np.sum(ocnt < -redist)

    def __init__(self, vecs, lum, res=64, blursun=1.0):
        self.raster = vecs
        self.lum = lum
        # (0.533 / 2 * np.pi / 180)**2
        self.omega = 2.163461454244656e-05*vecs.shape[0]/(res * res)
        self.vec = np.average(vecs, 0)

    def _to_pix(self, atv, vm, res):
        if atv > 90:
            ppix = vm.ivm.ray2pixel(self.raster, res)
            ppix[:, 0] += res
        else:
            ppix = vm.ray2pixel(self.raster, res)
        rec = np.core.records.fromarrays(ppix.T)
        px, i, cnt = np.unique(rec, return_index=True,
                               return_counts=True)
        cnt = cnt.astype(float)
        omegap = vm.pixel2omega(ppix[i] + .5, res)
        return px, omegap, cnt

    def add_to_img(self, img, vecs, mask=None, coefs=1, vm=None):
        if vm is None:
            vm = ViewMapper(self.vec, .533)
        res = img.shape[1]
        atv = vm.degrees(self.vec)[0]
        if atv <= vm.viewangle/2:
            px, omegap, cnt = self._to_pix(atv, vm, res)
            omegasp = self.omega / self.raster.shape[0]
            self._smudge(cnt, omegap, omegasp)
            # apply average luminanace over each pixel
            clum = coefs * self.lum * cnt * omegasp / omegap
            px = tuple(zip(*px))
            if omegasp > np.average(omegap):
                interp = LinearNDInterpolator(px, clum, fill_value=0)
                ppix = vm.ray2pixel(vecs, res)
                img[mask] += interp(ppix).reshape(img[mask].shape)
            else:
                img[px] += clum

    def get_ray(self, psi, vm, s):
        sun = np.asarray(s[0:3]).reshape(1, 3)
        if vm.in_view(sun, indices=False)[0] and psi in self.items():
            svlm = self.lum[psi]*s[3]
            svo = self.omega[psi]
            return s[0:3], svlm/self.blursun, svo*self.blursun
        else:
            return None



