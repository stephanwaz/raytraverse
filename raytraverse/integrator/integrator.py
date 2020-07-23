# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import itertools

import numpy as np
from raytraverse import translate, io, skycalc, helpers
from raytraverse.mapper import ViewMapper


class Integrator(object):
    """loads scene and sampling data for processing

    Parameters
    ----------
    skyfield: raytraverse.lightfield.SCBinField
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
        #: raytraverse.lightfield.SCBinField
        self.skyfield = skyfield
        self.dayhours = self.scene.skydata[:, 0] > 0

    def get_sky_mtx(self):
        sxyz = translate.aa2xyz(self.scene.skydata[self.dayhours, 0:2])
        if self.suns is not None:
            hassun, si = self.suns.proxy_src(sxyz, tol=self.stol)
            nosun = np.arange(hassun.size)[np.logical_not(hassun)]
        else:
            nosun = np.arange(sxyz.shape[0])
            hassun = 0
            si = np.zeros(sxyz.shape[0])
        sunuv = translate.xyz2uv(sxyz[nosun], flipu=False)
        sunbins = translate.uv2bin(sunuv, self.scene.skyres)
        dirdif = self.scene.skydata[self.dayhours, 2:]
        smtx, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, self.scene.skyres)
        # ratio between actual solar disc and patch
        omegar = np.square(0.2665 * np.pi * self.scene.skyres / 180) * .5
        plum = sun[nosun, -1] * omegar
        smtx[nosun, sunbins] += plum
        sun[:, -1] *= hassun
        return smtx, grnd, sun, si

    def hdr(self, pts, vdir, smtx, suns=None, suni=None,
            vname='view', viewangle=180.0, res=400, interp=1):
        """

        Parameters
        ----------
        pts: np.array
            points
        vdir: tuple
            view direction for images
        smtx: np.array
            sky matrix
        suns: np.array, optional
            sun values
        suni: np.array, optional
            sun indices
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
        hdr = helpers.header(self.scene)
        vstring = 'VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'.format(viewangle,
                                                                      *vdir)
        pdirs = vm.pixelrays(res)
        mask = vm.in_view(pdirs)
        for pi in pis:
            vstr = vstring + ' -vp {} {} {}'.format(*self.scene.idx2pt([pi])[0])
            si = self.skyfield.query_ray(pi, pdirs[mask], interp=interp)
            for sj, skyv in enumerate(smtx):
                outf = f"{self.scene.outdir}_{vm.name}_{pi:04d}_{sj:04d}.hdr"
                img = np.zeros((res*vm.aspect, res))
                self.skyfield.add_to_img(img, mask, pi, *si, coefs=skyv)
                if self.suns and suns[sj][-1] > 0:
                    sun = suns[sj]
                    psi = (pi, suni[sj])
                    j, e = self.sunfield.query_ray(psi, pdirs[mask],
                                                   interp=interp)
                    self.sunfield.add_to_img(img, mask, psi, j, e, sun[-1], vm)
                io.array2hdr(img, outf, hdr + [vstr])

    def illum(self, pts, vdirs, smtx, suns, si):
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
        si: np.array
            sin indices

        Returns
        -------
        illum: np.array
            illuminance at each point/direction and sky weighting
        """
        perrs, pis = self.scene.pt_kd.query(pts)
        vm = ViewMapper(viewangle=180, dxyz=vdirs)
        skylums = self.skyfield.get_illum(vm, pis, vdirs, smtx)
        psi = list(itertools.product(pis, si))
        sunlums = self.sunfield.get_illum(vm, psi, vdirs, suns)
        return sunlums + skylums

