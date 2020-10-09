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
from raytraverse import translate, io, skycalc, metricfuncs
from raytraverse.mapper import ViewMapper
from raytraverse.scene import SkyInfo
from raytraverse.crenderer import cRtrace
from raytraverse.lightfield import SunSkyPt


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

    def _get_metric_info(self, sunskyfield, skyv, sun, suni, pi, vm):
        if suni[0] in sunskyfield.items() and sun[-2] > 0:
            lf = sunskyfield
            li = suni[0]
            skyvec = np.concatenate((sun[-2:-1], skyv))
            sunview = lf.view.get_ray((pi, li), vm, sun[0:4])
            sunvalue = 1 + (sunview is not None)
        else:
            lf = self.skyfield
            li = pi
            skyvec = np.copy(skyv)
            skyvec[suni[1]] += sun[-1]
            sunview = None
            sunvalue = 0
        return lf, li, skyvec, sunview, sunvalue

    def hdr(self, sunskyfield, skyv, sun, suni, pi, vm, pdirs, mask, vstr,
            outf, interp=1):
        """interpolate and write hdr image for a single skyv/sun/pt-index
        combination

        Parameters
        ----------
        sunskyfield: raytraverse.lightfield.sunskypt
        skyv: np.array
            sky patch coefficients, size=self.skyfield.srcn
            (last value is ground)
        sun: np.array
            sun value (dx, dy, dz, true radiance, patch radiance)
        suni: tuple
            (sun index, sun bin) index is position in self.suns.suns, bin is
            corresponding patch in skyfield
        pi: int
            point index
        vm: raytraverse.mapper.viewmapper
            should have a view angle of 180 degrees, the analyis direction
        pdirs: np.array
            pixel ray directions, shape (res, res, 3)
        mask: tuple
            pair of integer np.array representing pixel coordinates of images
            to calculate
        vstr: str
            view string for radiance compatible header
        outf: str
            destination file path
        interp: int
            number of rays to search for in query, interpolation always happens
            between 3 points, but in order to find a valid mesh triangle more
            than 3 points is typically necessary. 16 seems to be a safe number
            set to 1 (default) to turn off interpolation and use nearest ray
            this will result in voronoi patches in the final image.

        Returns
        -------
        outf: str
            saved output file
        returntype: str
            'hdr' indicating format of result (useful when called
            as parallel process to seperate from 'metric' or other outputs)

        """
        img = np.zeros(pdirs.shape[:-1])
        lf, li, skyvec, _, _ = self._get_metric_info(sunskyfield, skyv, sun,
                                                     suni, pi, vm)
        lf.add_to_img(img, mask, li, pdirs[mask], coefs=skyvec, vm=vm,
                      interp=interp)
        io.array2hdr(img, outf, self.header() + [vstr])
        return outf, 'hdr'

    def metric(self, sunskyfield, skyv, sun, suni, pi, vm, info,
               mfuncs=(metricfuncs.illum,), **kwargs):
        """calculate metrics for a single skyv/sun/pt-index combination

        Parameters
        ----------
        sunskyfield: raytraverse.lightfield.sunskypt
        skyv: np.array
            sky patch coefficients, size=self.skyfield.srcn
            (last value is ground)
        sun: np.array
            sun value (dx, dy, dz, true radiance, patch radiance)
        suni: tuple
            (sun index, sun bin) index is position in self.suns.suns, bin is
            corresponding patch in skyfield
        pi: int
            point index
        vm: raytraverse.mapper.viewmapper
            should have a view angle of 180 degrees, the analyis direction
        mfuncs: tuple
            tuple of callables with the signature:
            ``f(vm, rays, omega, lum, **kwargs)``
            return a float
        kwargs:
            passed to metricfuncs

        Returns
        -------
        data: np.array
            results for skyv and pi, shape (len(info[0]) + len(metricfuncs) +
            1 + len(info[1])
        returntype: str
            'metric' indicating format of result (useful when called
            as parallel process to seperate from 'hdr' or other outputs)
        """
        lf, li, skyvec, svw, svl = self._get_metric_info(sunskyfield, skyv, sun,
                                                         suni, pi, vm)
        idx = lf.query_ball(li, vm.dxyz)[0]
        omega = np.squeeze(lf.omega[li][idx])
        rays = lf.vec[pi][idx]
        lum = np.squeeze(lf.apply_coef(li, skyvec))[idx]
        if svw is not None:
            rays = np.vstack((rays, svw[0][None, :]))
            lum = np.concatenate((lum, [svw[1]]))
            omega = np.concatenate((omega, [svw[2]]))
        fmetric = [f(vm, rays, omega, lum, **kwargs)
                   for f in mfuncs]
        data = np.concatenate((info[0], fmetric, [svl, ], info[1]))
        return data, 'metric'

    def integrate(self, pts, smtx, grnd=None, suns=None, suni=None,
                  daysteps=None, sunerrs=None, skydata=None, dohdr=True,
                  dometric=True, vname='view', viewangle=180.0, res=400,
                  interp=1, mfuncs=(metricfuncs.illum,),
                  metric_return_opts=None, **kwargs):
        """

        Parameters
        ----------
        pts
        smtx
        grnd
        suns
        suni
        daysteps
        sunerrs
        skydata
        dohdr
        dometric
        vname
        viewangle
        res
        interp
        mfuncs
        metric_return_opts: dict
            boolean dictionary of columns to print with metric output. Default:
            {"idx": False, "sensor": False, False, "sky": False}
        kwargs

        Returns
        -------

        """

        metricreturn = {"idx": False, "sensor": False,
                        "err": False, "sky": False}
        if metric_return_opts is not None:
            metricreturn.update(metric_return_opts)
        metricreturn["metric"] = True
        perrs, pis = self.scene.area.pt_kd.query(pts[:, 0:3])
        sort = np.argsort(pis)
        s_pis = pis[sort]
        s_perrs = perrs[sort]
        s_pts = pts[sort]
        if grnd is not None:
            smtx = np.hstack((smtx, grnd[:, None]))
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            fu = []
            last_pi = None
            sunsky = None
            for pj, (pi, pt, vdir, perr) in enumerate(zip(s_pis, s_pts[:, 0:3],
                                                          s_pts[:, 3:6],
                                                          s_perrs)):
                if pi != last_pi:
                    sunsky = SunSkyPt(self.skyfield, self.sunfield, pi)
                vm = ViewMapper(viewangle=viewangle, dxyz=vdir, name=vname)
                vmm = ViewMapper(viewangle=180, dxyz=vdir)
                vp = self.scene.area.idx2pt([pi])[0]
                vstr = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'
                        ' -vp {4} {5} {6}'.format(viewangle, *vdir, *vp))
                pdirs = vm.pixelrays(res)
                mask = vm.in_view(pdirs)
                for sj in range(len(smtx)):
                    if dohdr:
                        outf = (f"{self.scene.outdir}_{vname}_{pj:04d}_{sj:04d}"
                                ".hdr")
                        fu.append(exc.submit(self.hdr, sunsky, smtx[sj],
                                             suns[sj], suni[sj], pi, vm, pdirs,
                                             mask, vstr, outf, interp))
                    if dometric:
                        info = [[], []]
                        infos0 = dict(idx=[pi, sj], sensor=[*pt, *vdir])
                        infos1 = dict(err=[perr, sunerrs[sj]],
                                      sky=[*skydata[sj]])
                        for k, v in infos0.items():
                            if metricreturn[k]:
                                info[0] += v
                        for k, v in infos1.items():
                            if metricreturn[k]:
                                info[1] += v
                        fu.append(exc.submit(self.metric, sunsky, smtx[sj],
                                             suns[sj], suni[sj], pi, vmm, info,
                                             mfuncs, **kwargs))
            outmetrics = []
            for future in fu:
                out, kind = future.result()
                if kind == 'hdr':
                    print(out, file=sys.stderr)
                elif kind == 'metric':
                    outmetrics.append(out)
        if dometric:
            headers = dict(
                idx=["pt-idx", "sky-idx"],
                sensor=["x", "y", "z", "dx", "dy", "dz"],
                metric=[f.__name__ for f in mfuncs] + ["sun-value"],
                err=["pt-err", "sun-err"],
                sky=["sun-x", "sun-y", "sun-z", "dir-norm", "diff-horiz"])
            colhdr = []
            for k, v in headers.items():
                if metricreturn[k]:
                    colhdr += v
            d = np.array(outmetrics)
            cols = d.shape[1]
            unsort = np.argsort(sort)
            d = d.reshape(-1, len(smtx), cols)[unsort].reshape(-1, cols)
            return colhdr, d
        return None, []
