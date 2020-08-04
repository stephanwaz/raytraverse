# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import itertools
import os
from datetime import datetime, timezone
import subprocess

import numpy as np
import raytraverse
from raytraverse import translate, io, skycalc
from raytraverse.mapper import ViewMapper
from raytraverse.scene import SkyInfo


class Integrator(object):
    """class to generate outputsfrom skyfield sunfield and sky conditions

    This class provides an interface to:

        1. generate sky data using the perez
        2. combine lightfield data for sky and sun (handling sparsely populated
           sun data)
        3. apply sky data to the lightfield queries
        4. output luminance maps and photometric quantities and visual
           comfort metrics

    Parameters
    ----------
    skyfield: raytraverse.lightfield.SCBinField
        class containing sky data
    sunfield: raytraverse.lightfield.SunField
        class containing sun data (should share its scene parameter with the
        given skyfield
    wea: str, np.array, optional
        path to epw, wea, or .npy file or np.array, if loc not set attempts to
        extract location data (if needed). The Integrator does not need to be
        initialized with weather data but for convinience can be. If skydata
        is not set then the optional parameter of get_sky_matrix() is required
        or an Exception will be raised.
    loc: (float, float, int), optional
        location data given as lat, lon, mer with + west of prime meridian
        overrides location data in wea (but not in sunfield)
    skyro: float, optional
        angle in degrees counter-clockwise to rotate sky
        (to correct model north, equivalent to clockwise rotation of scene)
        does not override rotation in SunField)
    """

    def __init__(self, skyfield, sunfield=None, wea=None, loc=None, skyro=0.0,
                 **kwargs):
        #: raytraverse.scene.Scene
        self.scene = skyfield.scene
        try:
            #: raytraverse.sunsetter.SunSetter
            self.suns = sunfield.suns
        except AttributeError:
            self.suns = None
        try:
            #: raytraverse.scene.SkyInfo
            self.sky = self.suns.sky
        except AttributeError:
            if loc is None and wea is not None:
                loc = skycalc.get_loc_epw(wea)
            self.sky = SkyInfo(loc, skyro=skyro)
        #: raytraverse.lightfield.SunField
        self.sunfield = sunfield
        #: raytraverse.lightfield.SCBinField
        self.skyfield = skyfield
        if wea is not None:
            self.skydata = wea
        else:
            try:
                self.skydata = f'{self.scene.outdir}/skydat.txt'
            except OSError:
                self._skydata = None

    @property
    def skydata(self):
        """sky data formatted as dx, dy, dz, dirnorm, diffhoriz

        :getter: Returns this scene's skydata
        :setter: Sets this scene's skydata from file path or
        :type: np.array
        """
        return self._skydata

    @skydata.setter
    def skydata(self, wea):
        self._skydata = self.format_skydata(wea)
        np.savetxt(f'{self.scene.outdir}/skydat.txt', self._skydata)

    def format_skydata(self, dat):
        """process dat argument as skydata

        Parameters
        ----------
        dat: str, np.array
            This method takes either a file path or np.array. File path can
            point to a wea, epw, or .npy file. Loaded array must be one of the
            following:
                - 4 col: alt, az, dir, diff
                - 5 col: dx, dy, dz, dir, diff
                - 5 col: m, d, h, dir, diff'

        Returns
        -------
        np.array
            dx, dy, dz, dir, diff
        """
        loc = self.sky.loc
        if hasattr(dat, 'shape'):
            skydat = dat
        else:
            try:
                skydat = np.loadtxt(dat)
            except ValueError:
                if self.sky.loc is None:
                    loc = skycalc.get_loc_epw(dat)
                skydat = skycalc.read_epw(dat)
        if skydat.shape[1] == 4:
            xyz = translate.aa2xyz(skydat[:, 0:2])
            skydat = np.hstack((xyz, skydat[:, 2:]))
        elif skydat.shape[1] == 5:
            if np.max(skydat[:, 2]) > 2:
                if self.sky.loc is None:
                    raise ValueError("cannot parse wea data without a Location")
                times = skycalc.row_2_datetime64(skydat[:, 0:3])
                xyz = skycalc.sunpos_xyz(times, *loc, ro=self.sky.skyro)
                skydat = np.hstack((xyz, skydat[:, 3:]))
        else:
            raise ValueError('input data should be one of the following:'
                             '\n4 col: alt, az, dir, diff'
                             '\n5 col: dx, dy, dz, dir, diff'
                             '\n5 col: m, d, h, dir, diff')
        return skydat

    def get_sky_mtx(self, skydata=None):
        """generate sky, grnd and sun coefficients from sky data using perez

        Parameters
        ----------
        skydata: str, np.array. optional
            if None, uses object skydata (will raise error if unassigned)
            see format_skydata() for argument specifics.

        Returns
        -------
        smtx: np.array
            shape (len(skydata), skyres**2) coefficients for each sky patch
            each row is a timestep, timesteps where a sun exists exclude the
            sun coefficient, otherwise the patch enclosing the sun position
            contains the energy of the sun
        grnd: np.array
            shape (len(skydata),) coefficients for ground at each timestep
        sun: np.array
            shape (len(skydata), 4) sun position and coefficients for sun at
            each timestep assuming the true solid angle of the sun
            (value is zeroed out if no sun sample exists
        si: np.array
             shape (len(skydata),) index array for self.suns.suns pointing to
             correct proxy source
        daysteps: np.array
            shape (len(skydata),) boolean array masking timesteps when sun is
            below horizon
        """
        if skydata is None:
            skydata = self.skydata
        else:
            skydata = self.format_skydata(skydata)
        daysteps = skydata[:, 2] > 0
        sxyz = skydata[daysteps, 0:3]
        if self.suns is not None:
            hassun, si = self.suns.proxy_src(sxyz, tol=self.scene.skyres)
            nosun = np.arange(hassun.size)[np.logical_not(hassun)]
        else:
            nosun = np.arange(sxyz.shape[0])
            hassun = 0
            si = np.zeros(sxyz.shape[0])
        sunuv = translate.xyz2uv(sxyz, flipu=False)
        sunbins = translate.uv2bin(sunuv, self.scene.skyres)
        dirdif = skydata[daysteps, 2:]
        smtx, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, self.scene.skyres)
        # ratio between actual solar disc and patch
        omegar = np.square(0.2665 * np.pi * self.scene.skyres / 180) * .5
        plum = sun[:, -1] * omegar
        smtx[nosun, sunbins] += plum
        sun[:, -1] *= hassun
        return smtx, grnd, sun, si, daysteps

    def header(self):
        """generate image header string"""
        octe = f"{self.scene.outdir}/scene.oct"
        hdr = subprocess.run(f'getinfo {octe}'.split(), capture_output=True,
                             text=True)
        hdr = [i.strip() for i in hdr.stdout.split('\n')]
        hdr = [i for i in hdr if i[0:5] == 'oconv']
        tf = "%Y:%m:%d %H:%M:%S"
        hdr.append("CAPDATE= " + datetime.now().strftime(tf))
        hdr.append("GMT= " + datetime.now(timezone.utc).strftime(tf))
        radversion = subprocess.run('rpict -version'.split(),
                                    capture_output=True,
                                    text=True)
        hdr.append(f"SOFTWARE= {radversion.stdout}")
        lastmod = os.path.getmtime(os.path.dirname(raytraverse.__file__))
        tf = "%a %b %d %H:%M:%S %Z %Y"
        lm = datetime.fromtimestamp(lastmod, timezone.utc).strftime(tf)
        hdr.append(
            f"SOFTWARE= RAYTRAVERSE {raytraverse.__version__} lastmod {lm}")
        try:
            hdr.append("LOCATION= lat: {} lon: {} tz: {}".format(*self.sky.loc))
        except TypeError:
            pass
        return hdr

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
                io.array2hdr(img, outf, self.header() + [vstr])

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
