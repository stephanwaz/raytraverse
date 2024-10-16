# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np

from raytools import io
from raytraverse.evaluate import retina
from raytraverse.mapper import ViewMapper
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, LineString


class SrcViewPoint(object):
    """interface for sun view data"""

    @staticmethod
    def offset(points, target):
        hull = ConvexHull(points)
        tr = np.sqrt(target/np.pi)
        while abs(hull.volume - target)/target > .02:
            r = np.sqrt(hull.volume/np.pi)
            offset = tr - r
            hp = hull.points[hull.vertices]
            cp = np.average(hp, 0)
            vp = hp - cp
            asr = np.argsort(np.arctan2(vp[:, 0], vp[:, 1]))
            shp = hp[asr]
            p = LineString(np.concatenate((shp, shp[0:2])))
            b = p.offset_curve(offset, join_style=1)
            hull = ConvexHull(np.array(b.xy).T)
        return hull.points[hull.vertices]

    def __init__(self, scene, vecs, lum, pt=(0, 0, 0), posidx=0, src='sunview',
                 res=64, srcomega=6.796702357283834e-05):
        vecs = vecs.reshape(-1, 3)
        #: raytraverse.scene.Scene
        self.scene = scene
        #: int: index for point
        self.posidx = posidx
        #: np.array: point location
        self.pt = np.asarray(pt).flatten()[0:3]
        #: str: source key
        self.src = src
        #: float: source radius
        self.radius = (srcomega / np.pi) ** .5
        self.vec = np.average(vecs, 0)
        #: np.array: individual vectors that hit the source (pixels)
        if res == 1:
            res = 64
            self.raster = self.vm.uv2xyz(self.vm.idx2uv(np.arange(res*res), (res, res)))
            self.omega = srcomega
        else:
            self.raster = vecs
            # 2*np.pi*(1 - np.cos(0.533*np.pi/360))
            self.omega = srcomega * vecs.shape[0] / (res * res)
        if np.asarray(lum).size > 1:
            self.lum = io.rgb2rad(lum)
            self._color_coef = lum/self.lum
        else:
            #: float: source luminance (average)
            self.lum = lum
            self._color_coef = None



    @property
    def vm(self):
        return ViewMapper(self.vec, self.radius * 360 / np.pi)

    def _to_pix(self, atv, vm, res):
        if atv > 90:
            ppix = vm.ivm.ray2pixel(self.raster, res)
            omegap = vm.ivm.pixel2omega(ppix + .5, res)
            ppix[:, 0] += res
        else:
            ppix = vm.ray2pixel(self.raster, res)
            omegap = vm.pixel2omega(ppix + .5, res)
        rec = np.core.records.fromarrays(ppix.T)
        px, i, cnt = np.unique(rec, return_index=True, return_counts=True)
        cnt = cnt.astype(float)
        omegap = omegap[i]
        px = px.tolist()
        return px, omegap, cnt

    def _smudge(self, px, cnt, omegap, omegasp):
        """hack to ensure equal energy and max luminance)"""
        ocnt = cnt - (omegap/omegasp)
        smdg = np.sum(ocnt[ocnt > 0])
        room = -np.sum(ocnt[ocnt < 0])
        # reduce oversamples
        cnt[ocnt > 0] = omegap[ocnt > 0]/omegasp
        # allocate to undersamples
        if room > smdg:
            cnt[ocnt < 0] = (np.sum(cnt[ocnt < 0]) + smdg)/np.sum(ocnt < 0)
            return None
        # construct hull for interpolation
        else:
            target = self.omega/np.average(omegap)
            target = np.square(np.sqrt(target/np.pi) + .5) * np.pi
            try:
                hullpoints = SrcViewPoint.offset(px, target)
            except (ValueError, QhullError):
                return None
            return hullpoints

    def add_to_img(self, img, vecs, mask=None, coefs=1, vm=None):
        if vm is None:
            vm = ViewMapper(self.vec, .533)
        res = img.shape[-1]
        atv = vm.degrees(self.vec)[0]
        if atv <= vm.viewangle * vm.aspect / 2:
            px, omegap, cnt = self._to_pix(atv, vm, res)
            omegasp = self.omega / self.raster.shape[0]
            clum = coefs * self.lum*cnt*omegasp/omegap
            i2 = np.zeros(img.shape[-2:])
            hullpoints = self._smudge(px, cnt, omegap, omegasp)
            if hullpoints is not None:
                ppix = vm.ray2pixel(vecs, res, integer=False)
                pomega = vm.pixel2omega(ppix, res)
                # interpolate within original bounds
                interp = LinearNDInterpolator(px, clum, fill_value=0)
                luma = interp(ppix)
                # interpolate within expanded bounds to find perimeter
                clum = np.concatenate((clum, np.full(len(hullpoints),
                                                     coefs * self.lum)))
                xy = np.concatenate((px, hullpoints))
                interp = LinearNDInterpolator(xy, clum, fill_value=0)
                lumb = interp(ppix)
                # isolate perimeter
                corona = np.logical_and(lumb > 0, luma == 0)
                if np.sum(corona) > 0:
                    target = self.omega*self.lum*coefs
                    current = np.sum(pomega*luma)
                    # distribute gap in energy to corona pixels
                    luma[corona] = (target - current)/np.sum(pomega[corona])
                i2.flat[mask] = luma[mask]
            else:
                px = tuple(zip(*px))
                i2[px] += clum
            if vm.viewangle >= 10:
                r = res/vm.viewangle*.0625
                i2 = gaussian_filter(i2, r, truncate=8)
            try:
                ccf = self._color_coef
            except AttributeError:
                ccf = None
            if len(img.shape) > 2 and ccf is not None:
                img += np.maximum(i2 * ccf[:, None, None] - img, 0)
            else:
                img += np.maximum(i2-img, 0)

    @staticmethod
    def _hpsf(x, fwhm=0.183333):
        hm = np.square(fwhm/2)
        return hm/(np.square(x) + hm)

    @staticmethod
    def _inv_hpsf(y, fwhm=0.183333):
        hm = np.square(fwhm/2)
        return np.sqrt(hm/y - hm)

    def _blur_sun(self, lmax, lmin=279.33, fwhm=0.183333):
        r0 = np.sqrt(self.omega/np.pi) * 180/np.pi
        lmax = max(min(lmax, lmin/self._hpsf(1)), lmin)
        r = r0 + self._inv_hpsf(lmin/lmax, fwhm)
        return np.square(r/r0)

    def evaluate(self, sunval, vm=None, blursun=False):
        tol = np.sqrt(self.omega/np.pi)
        if (vm is None or vm.aspect == 2 or
                vm.in_view(self.vec, indices=False, tol=tol)[0]):
            svlm = self.lum * sunval
            svo = self.omega
            if blursun:
                ogas = np.atleast_1d(retina.blur_sun(self.omega, svlm))
                svlm = svlm / ogas
                svo = svo * ogas[0]
            svlm = np.atleast_1d(np.squeeze(svlm))
            return self.vec.reshape(-1, 3), np.array([svo]), svlm
        else:
            return self.vec.reshape(-1, 3), np.zeros(1), np.zeros(1)

    def direct_view(self, res=80):
        vm = ViewMapper(self.vec, self.vm.viewangle*1.2)
        vecs = vm.pixelrays(res).reshape(-1, 3)
        img = np.zeros((res, res))
        mask = vm.in_view(vecs)
        self.add_to_img(img, vecs, mask, vm=vm)
        outf = f"{self.scene.outdir}_{self.src}_{self.posidx:06d}.hdr"
        vstr = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'
                ' -vp {4} {5} {6}'.format(vm.viewangle, *vm.dxyz, *self.pt))
        io.array2hdr(img, outf, [vstr])
        return outf
