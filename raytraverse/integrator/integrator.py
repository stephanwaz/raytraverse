# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np
from raytraverse import translate, io, skycalc
from raytraverse.mapper import ViewMapper


class Integrator(object):
    """loads scene and sampling data for processing

    Parameters
    ----------
    skyfield: raytraverse.lightfield.SrcBinField
        class containing sky data
    sunfield: raytraverse.lightfield.SunField
        class containing sun data
    """

    def __init__(self, skyfield, sunfield=None, stol=10.0):
        #: raytraverse.scene.Scene
        self.scene = skyfield.scene
        try:
            suns = sunfield.suns
        except AttributeError:
            suns = None
        #: raytraverse.sunsetter.SunSetter
        self.suns = suns
        self.stol = stol
        #: raytraverse.lightfield.SunField
        self.sunfield = sunfield
        #: raytraverse.lightfield.SrcBinField
        self.skyfield = skyfield
        self.dayhours = self.scene.skydata[:, 0] > 0

    def get_sky_mtx(self):
        sxyz = translate.aa2xyz(self.scene.skydata[self.dayhours, 0:2])
        if self.suns is not None:
            hassun, si = self.suns.proxy_src(sxyz, tol=self.stol)
            nosun = np.arange(hassun.size)[np.logical_not(hassun)]
            oor = len(si)
            hassun = np.where(hassun, si, oor)
        else:
            nosun = np.arange(sxyz.shape[0])
            hassun = np.full(sxyz.shape[0], sxyz.shape[0])
        sunuv = translate.xyz2uv(sxyz[nosun], flipu=False)
        sunbins = translate.uv2bin(sunuv, self.scene.skyres)
        dirdif = self.scene.skydata[self.dayhours, 2:]
        smtx, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, self.scene.skyres)
        # ratio between actual solar disc and patch
        omegar = np.square(0.2665 * np.pi * self.scene.skyres / 180) * .5
        plum = sun[nosun, -1] * omegar
        smtx[nosun, sunbins] += plum
        return smtx, grnd, sun, hassun

    def hdr(self, pts, vdir, smtx, suns, hassun,
            vname='view', viewangle=180.0, res=400, interp=1):
        """

        Parameters
        ----------
        pts: np.array
            points
        vdir: (float, float, float)
            view direction for images
        smtx: np.array
            sky matrix
        suns: np.array
            sun values
        hassun: np.array
            boolean array if a high res sun exists
        vname: str
            view name for output file
        viewangle: float, optional
            degree opening of view cone
        res: int, optional
            image resolution
        interp: int, optional
            number of nearest points to interpolate between. 1 will resemble
            voronoi patches

        Returns
        -------

        """
        perrs, pis = self.scene.pt_kd.query(pts)
        vm = ViewMapper(viewangle=viewangle, dxyz=vdir, name=vname)
        pdirs = vm.pixelrays(res)
        mask = vm.in_view(pdirs)
        for pi in pis:
            si = self.skyfield.query_ray(pi, pdirs[mask], interp=interp)
            for sj, skyv in enumerate(smtx):
                sun = suns[sj]
                outf = f"{self.scene.outdir}_{vm.name}_{pi:04d}_{sj:04d}.hdr"
                img = np.zeros((res*vm.aspect, res))
                self.skyfield.add_to_img(img, mask, pi, *si, coefs=skyv)
                if (self.suns and hassun[sj] < self.suns.suns.shape[0]
                        and sun[-1] > 0):
                    psi = (pi, hassun[sj])
                    j, e = self.sunfield.query_ray(psi, pdirs[mask],
                                                   interp=interp)
                    self.sunfield.add_to_img(img, mask, psi, j, e, sun[-1])
                    self.sunfield.view.add_to_img(img, pi, sun, vm)
                io.array2hdr(img, outf)

    def illum(self, pts, vdirs, smtx, suns, hassun):
        """calculate illuminance for given sensor locations and skyvecs

        Parameters
        ----------
        pts: np.array
            points
        vdirs:
            view directions for illuminance
        smtx: np.array
            sky matrix
        suns: np.array
            sun values
        hassun: np.array
            boolean array if a high res sun exists

        Returns
        -------
        illum: np.array
            illuminance at each point/direction and sky weighting
        """
        perrs, pis = self.scene.pt_kd.query(pts)
        vm = ViewMapper(viewangle=180, dxyz=vdirs)
        illum = []
        for pi in pis:
            si = self.skyfield.query_ball(pi, vdirs)
            skylums = self.skyfield.get_illum(vm, pi, si, smtx)
            # Daylight factor
            # df = self.skyfield.get_illum(vm, pi, si, 1, 1/np.pi)
            for sj, sklm in enumerate(skylums):
                if (self.suns and hassun[sj] < self.suns.suns.shape[0] and
                        suns[sj, -1] > 0):
                    psi = (pi, hassun[sj])
                    si = self.sunfield.query_ball(psi, vdirs)
                    sulm = self.sunfield.get_illum(vm, psi, si, suns[sj, -1])
                    illum.append(sulm + sklm)
                else:
                    illum.append(sklm)
        return np.array(illum)

