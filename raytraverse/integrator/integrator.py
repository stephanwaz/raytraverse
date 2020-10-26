# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from raytraverse import translate, skycalc
from raytraverse.mapper import ViewMapper
from raytraverse.integrator.baseintegrator import BaseIntegrator


class Integrator(BaseIntegrator):
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

    rowheaders0 = dict(idx=["pt-idx", "sky-idx"],
                       sensor=["x", "y", "z", "dx", "dy", "dz"])
    rowheaders1 = dict(err=["pt-err", "sun-err"],
                       sky=["sun-x", "sun-y", "sun-z", "dir-norm", "diff-horiz"])

    def __init__(self, lightfield, wea=None, loc=None, skyro=0.0,
                 ground_fac=0.15):
        super().__init__(lightfield)
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
        skydat = np.atleast_2d(skydat)
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
        hdr = super().header()
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

    def _all_hdr(self, exc, pi, pj, vdir, res=400, viewangle=180.0,
                 vname='view', interp=1, altlf=None):
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
            lf, li, skyvec = self._prep_data(altlf, smtx[sj], suns[sj],
                                             self.sunproxy[sj], pi)
            fu.append(exc.submit(self.hdr, li, vm, pdirs, mask, vstr, outf,
                                 interp=interp, altlf=lf, coefs=skyvec))
        return fu

    def _all_metric(self, exc, pi, vdir, pt, perr, metricset, altlf, **kwargs):
        smtx, suns, daysteps, skydata = self.skydata
        vm = ViewMapper(viewangle=180, dxyz=vdir)
        fu = []
        for sj in range(len(smtx)):
            lf, li, skyvec = self._prep_data(altlf, smtx[sj], suns[sj],
                                             self.sunproxy[sj], pi)

            info = self._metric_info([pi, sj], [*pt, *vdir],
                                     [perr, self._serr[sj]], skydata[sj])
            fu.append(exc.submit(self.metric, li, vm, metricset, info, altlf=lf,
                                 coefs=skyvec, sunvec=suns[sj], **kwargs))
        return fu
