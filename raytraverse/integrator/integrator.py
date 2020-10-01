# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import subprocess

import numpy as np
import raytraverse
from raytraverse import translate, io, skycalc, metric
from raytraverse.mapper import ViewMapper
from raytraverse.scene import SkyInfo
from raytraverse.crenderer import cRtrace

class Integrator(object):
    """class to generate outputs from skyfield sunfield and sky conditions

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
                 ground_fac=0.15, **kwargs):
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
                try:
                    loc = skycalc.get_loc_epw(wea)
                except ValueError:
                    pass
            self.sky = SkyInfo(loc, skyro=skyro)
        #: raytraverse.lightfield.SunField
        self.sunfield = sunfield
        #: raytraverse.lightfield.SCBinField
        self.skyfield = skyfield
        self.ground_fac = ground_fac
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

    def get_sky_mtx(self, skydata=None, ground_fac=None):
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
            shape (len(skydata), 5) sun position (index 0,1,2) and coefficients
            for sun at each timestep assuming the true solid angle of the sun
            (index 3) and the weighted value for the sky patch (index 4).
        si: np.array
             shape (len(skydata), 2) index for self.suns.suns pointing to
             correct proxy source (col 0) and sunbin for patch mapping (col 1)
        daysteps: np.array
            shape (len(skydata),) boolean array masking timesteps when sun is
            below horizon
        serrs: np.array
            error (angle in degrees) between sun and proxy source
        skydata: no.array
            sun position and dirnorm diffhoriz
        """
        if skydata is None:
            skydata = self.skydata
        else:
            skydata = self.format_skydata(skydata)
        daysteps = skydata[:, 2] > 0
        sxyz = skydata[daysteps, 0:3]
        if self.suns is not None:
            si, serrs = self.suns.proxy_src(sxyz, tol=self.suns.sunres)
            # nosun = np.arange(hassun.size)[np.logical_not(hassun)]
        else:
            # nosun = np.arange(sxyz.shape[0])
            # hassun = 0
            si = np.zeros(sxyz.shape[0], dtype=int)
            serrs = si
        sunuv = translate.xyz2uv(sxyz, flipu=False)
        sunbins = translate.uv2bin(sunuv, self.scene.skyres)
        dirdif = skydata[daysteps, 3:]
        if ground_fac is None:
            ground_fac = self.ground_fac
        smtx, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, self.scene.skyres,
                                          ground_fac=ground_fac)
        # ratio between actual solar disc and patch
        omegar = np.square(0.2665 * np.pi * self.scene.skyres / 180) * .5
        plum = sun[:, -1] * omegar
        sun = np.hstack((sun, plum[:, None]))
        si = np.stack((si, sunbins)).T
        return smtx, grnd, sun, si, daysteps, serrs, skydata

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
        hdr.append(f"SOFTWARE= {cRtrace.version}")
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

    def _hdr(self, res, vm, keymap, pi, suni, sun, pdirs, si, vstr, outf,
             interp, mask, skyv):
        img = np.zeros((res*vm.aspect, res))
        if keymap[pi, suni[0]] and sun[-1] > 0:
            sun = sun[0:4]
            psi = (pi, suni[0])
            j, e = self.sunfield.query_ray(psi, pdirs[mask],
                                           interp=interp)
            self.sunfield.add_to_img(img, mask, psi, j, e, sun[-1], vm)
        else:
            skyv[suni[1]] += sun[4]
        self.skyfield.add_to_img(img, mask, pi, *si, coefs=skyv)
        io.array2hdr(img, outf, self.header() + [vstr])
        return outf

    def hdr(self, pts, smtx, grnd=None, suns=None, suni=None,
            daysteps=None, sunerrs=None, skydata=None, vname='view',
            viewangle=180.0, res=400, interp=1):
        """

        Parameters
        ----------
        pts: np.array
            points and view dirs shape (N, 6)
        smtx: np.array
            shape (len(skydata), skyres**2) coefficients for each sky patch
            each row is a timestep, timesteps where a sun exists exclude the
            sun coefficient, otherwise the patch enclosing the sun position
            contains the energy of the sun
        grnd: np.array
            shape (len(skydata),) coefficients for ground at each timestep
        suns: np.array
            shape (len(skydata), 5) sun position (index 0,1,2) and coefficients
            for sun at each timestep assuming the true solid angle of the sun
            (index 3) and the weighted value for the sky patch (index 4).
        suni: np.array
             shape (len(skydata), 2) index for self.suns.suns pointing to
             correct proxy source (col 0) and sunbin for patch mapping (col 1)
        daysteps: np.array
            shape (len(skydata),) boolean array masking timesteps when sun is
            below horizon
        sunerrs: np.array
            error (angle in degrees) between sun and proxy source
        skydata: no.array
            sun position and dirnorm diffhoriz
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
        npts = self.scene.area.npts
        perrs, pis = self.scene.area.pt_kd.query(pts[:, 0:3])
        if self.suns:
            keymap = self.sunfield.keymap()
        else:
            keymap = np.full((npts, 1), False)
        if grnd is not None:
            smtx = np.hstack((smtx, grnd[:, None]))
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            fu = []
            for pj, (pi, vdir) in enumerate(zip(pis, pts[:, 3:6])):
                vm = ViewMapper(viewangle=viewangle, dxyz=vdir, name=vname)
                vp = self.scene.area.idx2pt([pi])[0]
                vstr = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'
                        ' -vp {4} {5} {6}'.format(viewangle, *vdir, *vp))
                pdirs = vm.pixelrays(res)
                mask = vm.in_view(pdirs)
                si = self.skyfield.query_ray(pi, pdirs[mask], interp=interp)
                for sj, skyv in enumerate(smtx):
                    outf = (f"{self.scene.outdir}_{vname}_{pj:04d}_{sj:04d}"
                            ".hdr")
                    fu.append(exc.submit(self._hdr, res, vm, keymap, pi,
                                         suni[sj], suns[sj], pdirs, si, vstr,
                                         outf, interp, mask, skyv))
            for future in as_completed(fu):
                print(future.result(), file=sys.stderr)

    def metric(self, pts, smtx, grnd=None, suns=None, suni=None,
               daysteps=None, sunerrs=None, skydata=None,
               metricfuncs=(metric.illum,), **kwargs):
        """calculate luminance based metrics for given sensors and skyvecs

        Parameters
        ----------
        pts: np.array
            points and view dirs shape (N, 6)
        smtx: np.array
            shape (len(skydata), skyres^2) coefficients for each sky patch
            each row is a timestep, timesteps where a sun exists exclude the
            sun coefficient, otherwise the patch enclosing the sun position
            contains the energy of the sun
        grnd: np.array
            shape (len(skydata),) coefficients for ground at each timestep
        suns: np.array
            shape (len(skydata), 5) sun position (index 0,1,2) and coefficients
            for sun at each timestep assuming the true solid angle of the sun
            (index 3) and the weighted value for the sky patch (index 4).
        suni: np.array
             shape (len(skydata), 2) index for self.suns.suns pointing to
             correct proxy source (col 0) and sunbin for patch mapping (col 1)
        daysteps: np.array
            shape (len(skydata),) boolean array masking timesteps when sun is
            below horizon
        sunerrs: np.array
            error (angle in degrees) between sun and proxy source
        skydata: no.array
            sun position and dirnorm diffhoriz
        metricfuncs: tuple
            list of metric functions to apply, with the call signature:
            f(view, rays, omegas, lum, kwargs)

        Returns
        -------
        metrics: np.array
            metrics at each point/direction and sky weighting axes ordered as:
            (view, points, skys, metrics) an additional column on the final axis
            is 0 when only the skyfield is used, 1 when sun reflections exists
            and 2 when a view to the sun also exists.
        colhdr: list
            column headers
        """
        npts = self.scene.area.npts
        perrs, pis = self.scene.area.pt_kd.query(pts[:, 0:3])
        mthdr = [f.__name__ for f in metricfuncs]
        errhdr = ["sun-value", "pt-err", "sun-err"]
        poshdr = ["x", "y", "z", "dx", "dy", "dz"]
        sunhdr = ["sun-x", "sun-y", "sun-z", "dir-norm", "diff-horiz"]
        colhdr = poshdr + mthdr + errhdr + sunhdr
        if self.suns:
            keymap = self.sunfield.keymap()
        else:
            keymap = np.full((npts, 1), False)
        pmetrics = []
        if grnd is not None:
            smtx = np.hstack((smtx, grnd[:, None]))
        for pi, pt, vdir, perr in zip(pis, pts[:, 0:3], pts[:, 3:6], perrs):
            # get sky vectors
            skyidx = self.skyfield.query_ball(pi, [vdir])
            smetrics = []
            vm = ViewMapper(vdir, viewangle=180)
            for skyv, s, (si, sb), serr, skydat in zip(smtx, suns, suni,
                                                       sunerrs, skydata):
                psi = (pi, si)
                if keymap[pi, si] and s[-2] > 0:
                    skyvec = skyv
                    sunidx = self.sunfield.query_ball(psi, [vdir])
                    sunlm = np.squeeze(self.sunfield.apply_coef(psi, s[-2]))
                else:
                    skyvec = np.copy(skyv)
                    skyvec[sb] += s[-1]
                    sunidx = False
                    sunlm = False
                skylm = np.squeeze(self.skyfield.apply_coef(pi, skyvec))
                skyrays = self.skyfield.vec[pi][skyidx[0]]
                osky = np.squeeze(self.skyfield.omega[pi][skyidx[0]])
                lmsky = skylm[skyidx[0]]
                sunvalue = 0
                mstack = []
                skyweight = 1
                if sunidx:
                    sunvalue += 1
                    sunrays = self.sunfield.vec[psi][sunidx[0]]
                    osun = np.squeeze(self.sunfield.omega[psi][sunidx[0]])
                    lmsun = sunlm[sunidx[0]]
                    print(sunrays.shape, skyrays.shape, osun.shape, osky.shape)
                    # TODO cross match samples
                    # currently implements a nearest neighbor and then
                    # averages the two results (using the skyweight term,
                    # this is subject to some error a better approach is
                    # to query the ray for neighbors with a radius derived
                    # from omega and average in the case of no result
                    # (when omega_other is larger) use the nearest neighbor.
                    # This will likely need to be implemented in c++ to be
                    # practical, and alternative is to interpolate samples
                    # like hdr image generation and then naively add.
                    sunsky, err = self.sunfield.query_ray(psi, skyrays)
                    skysun, err = self.skyfield.query_ray(pi, sunrays)
                    lmsun += skylm[skysun]
                    lmsky += sunlm[sunsky]
                    skyweight = .5
                    fsun = [f(vm, sunrays, osun, lmsun, **kwargs) * skyweight
                            for f in metricfuncs]
                    mstack.append(fsun)
                fsky = [f(vm, skyrays, osky, lmsky, **kwargs) * skyweight
                        for f in metricfuncs]
                mstack.append(fsky)
                try:
                    fsv = self.sunfield.view.metric(psi, vm, s, metricfuncs,
                                                    **kwargs)
                except (AttributeError, ValueError):
                    pass
                else:
                    sunvalue += 1
                    mstack.append(fsv)
                # colhdr = poshdr + mthdr + errhdr + sunhdr
                mtcs = np.sum(mstack, 0)
                errs = [sunvalue, perr, serr]
                poss = [*pt, *vdir]
                smetrics.append(np.concatenate((poss, mtcs, errs, skydat)))
            pmetrics.append(smetrics)
        # loop returns shape: (points, skys, metrics)
        # transpose reorganizes as: (view, points, skys, metrics)
        return np.array(pmetrics), colhdr
