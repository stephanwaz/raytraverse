# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np

from raytraverse import io
from raytraverse.mapper import ViewMapper
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull
from scipy.ndimage.filters import gaussian_filter
from shapely.geometry import Polygon


class SrcViewPoint(object):
    """interface for sun view data"""

    @staticmethod
    def offset(points, target):
        hull = ConvexHull(points)
        tr = np.sqrt(target/np.pi)
        while abs(hull.volume - target)/target > .02:
            r = np.sqrt(hull.volume/np.pi)
            offset = tr - r
            p = Polygon(hull.points[hull.vertices])
            b = p.boundary.parallel_offset(offset, join_style=2)
            hull = ConvexHull(np.array(b.xy).T)
        return hull.points[hull.vertices]

    def __init__(self, scene, vecs, lum, pt=(0, 0, 0), posidx=0, src='sunview',
                 res=64, blursun=1.0, srcomega=6.796702357283834e-05):
        #: raytraverse.scene.Scene
        self.scene = scene
        #: int: index for point
        self.posidx = posidx
        #: np.array: point location
        self.pt = np.asarray(pt).flatten()[0:3]
        #: str: source key
        self.src = src
        #: np.array: individual vectors that hit the source (pixels)
        self.raster = vecs
        #: float: source luminance (average)
        self.lum = lum
        #: float: source radius
        self.radius = (srcomega/np.pi)**.5
        # 2*np.pi*(1 - np.cos(0.533*np.pi/360))
        self.omega = srcomega*vecs.shape[0]/(res * res)
        self.vec = np.average(vecs, 0)
        self.blursun = blursun

    @property
    def vm(self):
        return ViewMapper(self.vec, .533)

    def _to_pix(self, atv, vm, res):
        if atv > 90:
            ppix = vm.ivm.ray2pixel(self.raster, res)
            ppix[:, 0] += res
        else:
            ppix = vm.ray2pixel(self.raster, res)
        rec = np.core.records.fromarrays(ppix.T)
        px, i, cnt = np.unique(rec, return_index=True, return_counts=True)
        cnt = cnt.astype(float)
        omegap = vm.pixel2omega(ppix[i] + .5, res)
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
            hullpoints = SrcViewPoint.offset(px, target)
            return hullpoints

    def add_to_img(self, img, vecs, mask=None, coefs=1, vm=None):
        if vm is None:
            vm = ViewMapper(self.vec, .533)
        res = img.shape[1]
        atv = vm.degrees(self.vec)[0]
        if atv <= vm.viewangle/2:
            px, omegap, cnt = self._to_pix(atv, vm, res)
            omegasp = self.omega / self.raster.shape[0]
            hullpoints = self._smudge(px, cnt, omegap, omegasp)
            clum = coefs * self.lum*cnt*omegasp/omegap
            i2 = np.zeros(img.shape)
            if hullpoints is not None:
                ppix = vm.ray2pixel(vecs, res, integer=False)
                pomega = vm.pixel2omega(ppix, res)

                # interpolate within original bounds
                interp = LinearNDInterpolator(px, clum, fill_value=0)
                luma = interp(ppix).reshape(img[mask].shape)
                # interpolate within expanded bounds to find perimeter
                clum = np.concatenate((clum, np.full(len(hullpoints),
                                                     coefs * self.lum)))
                xy = np.concatenate((px, hullpoints))
                interp = LinearNDInterpolator(xy, clum, fill_value=0)
                lumb = interp(ppix).reshape(img[mask].shape)
                # isolate perimeter
                corona = np.logical_and(lumb > 0, luma == 0)

                target = self.omega*self.lum*coefs
                current = np.sum(pomega*luma)
                gap = (target - current)/np.sum(pomega[corona])
                luma[corona] = gap * coefs * self.lum
                i2[mask] = luma
            else:
                px = tuple(zip(*px))
                i2[px] += clum
            if vm.viewangle >= 10:
                r = res/vm.viewangle*.0625
                i2 = gaussian_filter(i2, r, truncate=8)
            img += i2

    def evaluate(self, sunval, vm=None):
        if vm is None or vm.in_view(self.vec, indices=False)[0]:
            svlm = self.lum * sunval/self.blursun
            svo = self.omega * self.blursun
            return self.vec, svo, svlm
        else:
            return self.vec, 0, 0

    def direct_view(self, res=80):
        vm = ViewMapper(self.vec, .666)
        vecs = vm.pixelrays(res)
        img = np.zeros((res, res))
        mask = vm.in_view(vecs)
        self.add_to_img(img, vecs[mask], mask, vm=vm)
        outf = f"{self.scene.outdir}_{self.src}_{self.posidx:06d}.hdr"
        vstr = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'
                ' -vp {4} {5} {6}'.format(vm.viewangle, *vm.dxyz, *self.pt))
        io.array2hdr(img, outf, [vstr])
        return outf
