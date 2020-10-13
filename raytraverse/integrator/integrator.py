# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import subprocess

import numpy as np
import raytraverse
from raytraverse import translate, io, skycalc
from raytraverse.mapper import ViewMapper
from raytraverse.crenderer import cRtrace
from raytraverse.lightfield import SunSkyPt
from raytraverse.integrator.metricset import MetricSet


class Integrator(object):
    """class to generate outputs from lightfield(s) and sky conditions

    This class provides an interface to:

        1. generate sky data using the perez
        2. combine lightfield data
        3. apply sky data to the lightfield queries
        4. output luminance maps as hdr images, photometric quantities, and
           visual comfort metrics

    prior to calling integrate, if updating wea, loc or skyro, use the "sky"
    parameter setter. if only the wea is changing, the skydata setter can be
    called directly.

    Parameters
    ----------
    lightfield: raytraverse.lightfield.LightFieldKD
        class containing sample data
    wea: str, np.array, optional
        path to epw, wea, or .npy file or np.array, if loc not set attempts to
        extract location data (if needed). The Integrator does not need to be
        initialized with weather data but for convinience can be. However,
        self.skydata must be initialized (directly or through self.sky) before
        calling integrate.
    loc: (float, float, int), optional
        location data given as lat, lon, mer with + west of prime meridian
        overrides location data in wea (but not in sunfield)
    skyro: float, optional
        angle in degrees counter-clockwise to rotate sky
        (to correct model north, equivalent to clockwise rotation of scene)
        does not override rotation in SunField)
    ground_fac: float, optional
        ground reflectance
    """

    def __init__(self, lightfield, wea=None, loc=None, skyro=0.0,
                 ground_fac=0.15):
        #: raytraverse.lightfield.LightFieldKD
        self.skyfield = lightfield
        #: raytraverse.scene.Scene
        self.scene = lightfield.scene
        self.ground_fac = ground_fac
        # in case a child has already set
        if not hasattr(self, "suns"):
            try:
                self.suns = lightfield.suns
            except AttributeError:
                self.suns = None
        if loc is None and wea is not None:
            try:
                loc = skycalc.get_loc_epw(wea)
            except ValueError:
                pass
        self.sky = (wea, loc, skyro)

    @property
    def sky(self):
        """location and sky rotation information

        :getter: Returns (loc, skyro)
        :setter: Sets (loc, skyro) and updates skydata"""
        return self._loc, self._skyro

    @sky.setter
    def sky(self, skyinfo):
        """sky setter

        Parameters
        ----------
        skyinfo: a tuple of (wea, loc, and skyro):
            wea: str, np.array
                This method takes either a file path or np.array. File path can
                point to a wea, epw, or .npy file. Loaded array must be one of
                the following:
                - 4 col: alt, az, dir, diff
                - 5 col: dx, dy, dz, dir, diff
                - 5 col: m, d, h, dir, diff'
            loc: tuple
                lot, lon, mer (in degrees, west is positive)
            skyro: float
                sky rotation (in degrees, ccw)
        """
        wea, self._loc, self._skyro = skyinfo
        if wea is not None:
            self.skydata = wea
        else:
            self._skydata = None
            self._sunproxy = None
            self._serr = None

    @property
    def skydata(self):
        """a tuple of sky data required to integrate

        skydata contains:

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
        daysteps: np.array
            shape (len(skydata),) boolean array masking timesteps when sun is
            below horizon
        skydata: no.array
            sun position and dirnorm diffhoriz
        """
        return self._skydata

    def _format_skydata(self, dat):
        """process dat argument as skydata

        see sky.setter for details on argument

        Returns
        -------
        np.array
            dx, dy, dz, dir, diff
        """
        loc = self._loc
        if hasattr(dat, 'shape'):
            skydat = dat
        else:
            try:
                skydat = np.loadtxt(dat)
            except ValueError:
                if loc is None:
                    loc = skycalc.get_loc_epw(dat)
                skydat = skycalc.read_epw(dat)
        if skydat.shape[1] == 4:
            xyz = translate.aa2xyz(skydat[:, 0:2])
            skydat = np.hstack((xyz, skydat[:, 2:]))
        elif skydat.shape[1] == 5:
            if np.max(skydat[:, 2]) > 2:
                if loc is None:
                    raise ValueError("cannot parse wea data without a Location")
                times = skycalc.row_2_datetime64(skydat[:, 0:3])
                xyz = skycalc.sunpos_xyz(times, *loc, ro=self._skyro)
                skydat = np.hstack((xyz, skydat[:, 3:]))
        else:
            raise ValueError('input data should be one of the following:'
                             '\n4 col: alt, az, dir, diff'
                             '\n5 col: dx, dy, dz, dir, diff'
                             '\n5 col: m, d, h, dir, diff')
        return skydat

    @skydata.setter
    def skydata(self, wea):
        """calculate sky matrix data and store in self._skydata"""
        skydata = self._format_skydata(wea)
        daysteps = skydata[:, 2] > 0
        sxyz = skydata[daysteps, 0:3]
        dirdif = skydata[daysteps, 3:]
        smtx, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, self.scene.skyres,
                                          ground_fac=self.ground_fac)
        # ratio between actual solar disc and patch
        omegar = np.square(0.2665 * np.pi * self.scene.skyres / 180) * .5
        plum = sun[:, -1] * omegar
        sun = np.hstack((sun, plum[:, None]))
        smtx = np.hstack((smtx, grnd[:, None]))
        self._skydata = (smtx, sun, daysteps, skydata)
        self.sunproxy = sxyz

    @property
    def sunproxy(self):
        return self._sunproxy

    @sunproxy.setter
    def sunproxy(self, sxyz):
        sunuv = translate.xyz2uv(sxyz, flipu=False)
        sunbins = translate.uv2bin(sunuv, self.scene.skyres)
        try:
            si, serrs = self.suns.proxy_src(sxyz, tol=self.suns.sunres)
        except AttributeError:
            si = [0] * len(sunbins)
            serrs = sunbins
        self._serr = serrs
        self._sunproxy = list(zip(sunbins, si))

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
            hdr.append("LOCATION= lat: {} lon: {} tz: {}".format(*self._loc))
        except TypeError:
            pass
        return hdr

    def _prep_data(self, sunskyfield, skyv, sun, sunb, pi):
        """prepare arguments for hdr/metric computation"""
        skyvec = np.copy(skyv)
        skyvec[sunb[0]] += sun[-1]
        idxtype = type(next(iter(sunskyfield.items())))
        # sky dependent lightfied (eg. SunField)
        if idxtype == tuple:
            pi = (pi, sunb[1])
        # sun only lightfield (eg. SunField)
        if self.skyfield.srcn == 1:
            skyvec = sun[3]
        return self.skyfield, pi, skyvec

    def hdr(self, sunskyfield, skyv, sun, sunb, pi, vm, pdirs, mask, vstr,
            outf, interp=1):
        """interpolate and write hdr image for a single skyv/sun/pt-index
        combination

        Parameters
        ----------
        skyv: np.array
            sky patch coefficients, size=self.skyfield.srcn
            (last value is ground)
        sun: np.array
            sun value (dx, dy, dz, true radiance, patch radiance)
        sunb: (sun index, sun bin) index is position in self.suns.suns, bin is
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
        sunskyfield: raytraverse.lightfield.sunskypt, optional
            used by child classes
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
        lf, li, skyvec = self._prep_data(sunskyfield, skyv, sun, sunb, pi)
        try:
            lf.add_to_img(img, mask, li, pdirs[mask], coefs=skyvec, vm=vm,
                          interp=interp)
        except KeyError:
            outf = f"skipped (no entry in LightField): {outf}"
        else:
            io.array2hdr(img, outf, self.header() + [vstr])
        return outf, 'hdr'

    def metric(self, sunskyfield, skyv, sun, sunb, pi, vm, metricset, info,
               **kwargs):
        """calculate metrics for a single skyv/sun/pt-index combination

        Parameters
        ----------
        skyv: np.array
            sky patch coefficients, size=self.skyfield.srcn
            (last value is ground)
        sun: np.array
            sun value (dx, dy, dz, true radiance, patch radiance)
        sunb: (sun index, sun bin) index is position in self.suns.suns, bin is
            corresponding patch in skyfield
        pi: int
            point index
        vm: raytraverse.mapper.ViewMapper
            analysis point
        metricset: str, list
            string or list of strings naming metrics.
            see raytraverse.integrator.MetricSet.metricdict for valid names
        info: list
            constant column values to include in row output
        sunskyfield: raytraverse.lightfield.sunskypt, optional
            used by child classes
        kwargs:
            passed to metricset

        Returns
        -------
        data: np.array
            results for skyv and pi, shape (len(info[0]) + len(metricfuncs) +
            1 + len(info[1])
        returntype: str
            'metric' indicating format of result (useful when called
            as parallel process to seperate from 'hdr' or other outputs)
        """
        lf, li, skyvec = self._prep_data(sunskyfield, skyv, sun, sunb, pi)
        try:
            rays, omega, lum = lf.get_applied_rays(li, vm, skyvec, sunvec=sun)
        except KeyError:
            print("skipped (no entry in LightField), returning zero line"
                  f": {info[0] + info[1]}", file=sys.stderr)
            if metricset is None or len(metricset) == 0:
                nmets = len(MetricSet.allmetrics)
            else:
                nmets = len(metricset)
            data = np.zeros(len(info[0]) + len(info[1]) + nmets)
        else:
            fmetric = MetricSet(vm, rays, omega, lum, metricset, **kwargs)()
            data = np.concatenate((info[0], fmetric, info[1]))
        return data, 'metric'

    @staticmethod
    def _metric_info(metricreturn, pi, sj, sensor, perr, sky, serr):
        info = [[], []]
        infos0 = dict(idx=[pi, sj], sensor=sensor)
        infos1 = dict(err=[perr, serr],
                      sky=[*sky])
        for k, v in infos0.items():
            if metricreturn[k]:
                info[0] += v
        for k, v in infos1.items():
            if metricreturn[k]:
                info[1] += v
        return info

    def _loop_smtx_hdr(self, exc, sunsky, pi, pj, vdir, res=400,
                       viewangle=180.0, vname='view', interp=1):
        smtx, suns, daysteps, skydata = self.skydata
        vm = ViewMapper(viewangle=viewangle, dxyz=vdir, name=vname)
        vp = self.scene.area.idx2pt([pi])[0]
        vstr = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'
                ' -vp {4} {5} {6}'.format(viewangle, *vdir, *vp))
        pdirs = vm.pixelrays(res)
        mask = vm.in_view(pdirs)
        fu = []
        for sj in range(len(smtx)):
            outf = (f"{self.scene.outdir}_{vname}_{pj:04d}_{sj:04d}"
                    ".hdr")
            fu.append(exc.submit(self.hdr, sunsky, smtx[sj],
                                 suns[sj], self.sunproxy[sj], pi, vm, pdirs,
                                 mask, vstr, outf, interp))
        return fu

    def _loop_smtx_met(self, exc, sunsky, pi, vdir, pt, perr, metricreturn,
                       metricset, **kwargs):
        smtx, suns, daysteps, skydata = self.skydata
        vm = ViewMapper(viewangle=180, dxyz=vdir)
        fu = []
        for sj in range(len(smtx)):
            info = self._metric_info(metricreturn, pi, sj, [*pt, *vdir],
                                     perr, skydata[sj], self._serr[sj])
            fu.append(exc.submit(self.metric, sunsky, smtx[sj], suns[sj],
                                 self.sunproxy[sj], pi, vm, metricset, info,
                                 **kwargs))
        return fu

    def pt_field(self, pi):
        return self.skyfield

    def integrate(self, pts, dohdr=True, dometric=True, vname='view',
                  viewangle=180.0, res=400, interp=1, metricset="illum",
                  metric_return_opts=None, **kwargs):
        """iterate through points and sky vectors to efficiently compute
        both hdr output and metrics, sharing intermediate calculations where
        possible.

        Parameters
        ----------
        pts: np.array
            shape (N, 6) points (with directions) to compute
        dohdr: bool, optional
            if True, output hdr images
        dometric: bool, optional
            if True, output metric data
        vname: str, optional
            view name for hdr output
        viewangle: float, optional
            view angle (in degrees) for hdr output (always an angular fisheye)
        res: int, optional
            pixel resolution of output hdr
        interp: int, optional
            if greater than one the bandwidth to search for nearby rays. from
            this set, a triangle including the closest point is formed for a
            barycentric interpolation.
        metricset: str, list, optional
            string or list of strings naming metrics.
            see raytraverse.integrator.MetricSet.allmetrics for valid choices
        metric_return_opts: dict, optional
            boolean dictionary of columns to print with metric output. Default:
            {"idx": False, "sensor": False, False, "sky": False}
        kwargs:
            additional parameters for integrator.MetricSet

        Returns
        -------
        header: list
            if dometric, a list of column headers, else an empty list
        data: np.array
            if dometric, an array of output data, else None

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
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            fu = []
            last_pi = None
            sunsky = None
            loopdat = zip(s_pis, s_pts[:, 0:3], s_pts[:, 3:6], s_perrs)
            for pj, (pi, pt, vdir, perr) in enumerate(loopdat):
                if pi != last_pi:
                    sunsky = self.pt_field(pi)
                fu = []
                if dohdr:
                    fu += self._loop_smtx_hdr(exc, sunsky, pi, pj, vdir, res,
                                              viewangle, vname, interp)
                if dometric:
                    fu += self._loop_smtx_met(exc, sunsky, pi, vdir, pt, perr,
                                              metricreturn, metricset, **kwargs)
            outmetrics = []
            for future in fu:
                out, kind = future.result()
                if kind == 'hdr':
                    print(out, file=sys.stderr)
                elif kind == 'metric':
                    outmetrics.append(out)
        if dometric:
            if len(metricset) == 0:
                metricset = MetricSet.allmetrics
            headers = dict(
                idx=["pt-idx", "sky-idx"],
                sensor=["x", "y", "z", "dx", "dy", "dz"],
                metric=(" ".join(metricset)).split(),
                err=["pt-err", "sun-err"],
                sky=["sun-x", "sun-y", "sun-z", "dir-norm", "diff-horiz"])
            colhdr = []
            for k, v in headers.items():
                if metricreturn[k]:
                    colhdr += v
            d = np.array(outmetrics)
            cols = d.shape[1]
            unsort = np.argsort(sort)
            d = d.reshape((len(pts), -1, cols))[unsort].reshape(-1, cols)
            return colhdr, d
        return None, []
