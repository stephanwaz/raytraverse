# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi
import clasp.script_tools as cst
from raytraverse import translate, io, optic, skycalc
from raytraverse.lightfield import SunField, SrcBinField
from raytraverse.mapper import ViewMapper


class Integrator(object):
    """loads scene and sampling data for processing

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun vectors
    """

    def __init__(self, scene, suns, stol=10.0):
        #: raytraverse.scene.Scene
        self.scene = scene
        #: raytraverse.sunsetter.SunSetter
        self.suns = suns
        self.stol = stol
        self.sunfield = SunField(scene, suns)
        self.skyfield = SrcBinField(scene, prefix='sky')
        self.dayhours = self.scene.skydata[:, 0] > 0

    def get_sky_mtx(self):
        sxyz = translate.aa2xyz(self.scene.skydata[self.dayhours, 0:2])
        hassun, si = self.suns.proxy_src(sxyz, tol=self.stol)
        nosun = np.arange(hassun.size)[np.logical_not(hassun)]
        sunuv = translate.xyz2uv(sxyz[nosun], flipu=False)
        sunbins = translate.uv2bin(sunuv, self.scene.skyres)
        dirdif = self.scene.skydata[self.dayhours, 2:]
        smtx, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, self.scene.skyres)
        # ratio between actual solar disc and patch
        omegar = np.square(0.2665 * np.pi * self.scene.skyres / 180) * .5
        plum = sun[nosun, -1] * omegar
        smtx[nosun, sunbins] += plum
        oor = len(si)
        return smtx, grnd, sun, np.where(hassun, si, oor)

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
        vm = ViewMapper(viewangle=viewangle, dxyz=vdir)
        pdirs, mask = vm.pixelrays(res)
        img = np.zeros((res * vm.aspect, res))
        for pi in pis:
            skylum = self.skyfield.measure(pi, pdirs[mask], smtx, interp=interp)
            for sj, skyv in enumerate(skylum):
                outf = f"{self.scene.outdir}_{vname}_{pi:04d}_{sj:04d}.hdr"
                img[mask] = skyv
                if hassun[sj] < self.suns.suns.shape[0] and suns[sj, -1] > 0:
                    psi = (pi, hassun[sj])
                    img[mask] += self.sunfield.measure(psi, pdirs[mask],
                                                       suns[sj, -1],
                                                       interp=interp)
                    spix, svals = self.sunfield.draw_sun(psi, suns[sj], vm, res)
                    if spix is not None:
                        print(outf)
                        img[spix[:, 0], spix[:, 1]] += svals
                io.array2hdr(img, outf)

    def illum(self, pts, vdir, smtx, suns, hassun):
        """calculate illuminance for given sensor locations and skyvecs

        Parameters
        ----------
        pts: np.array
            points
        vdir: (float, float, float)
            view direction for illuminance
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
        vm = ViewMapper(viewangle=180, dxyz=vdir)
        for pi in pis:
            for vd in vdir:
                l, v, o = self.skyfield.gather(pi, vd, smtx)
                print(np.sum(vm.ctheta(v)*l*o, 1)*179)
            # for sj, skyv in enumerate(skylum):
            #     outf = f"{self.scene.outdir}_{vname}_{pi:04d}_{sj:04d}.hdr"
            #     img[mask] = skyv
            #     if hassun[sj] < self.suns.suns.shape[0] and suns[sj, -1] > 0:
            #         psi = (pi, hassun[sj])
            #         img[mask] += self.sunfield.measure(psi, pdirs[mask],
            #                                            suns[sj, -1],
            #                                            interp=interp)
            #         spix, svals = self.sunfield.draw_sun(psi, suns[sj], vm, res)
            #         if spix is not None:
            #             print(outf)
            #             img[spix[:, 0], spix[:, 1]] += svals
            #     io.array2hdr(img, outf)

