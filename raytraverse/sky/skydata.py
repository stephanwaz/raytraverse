# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from raytraverse import translate
from raytraverse.sky import skycalc
from raytraverse.formatter import RadianceFormatter as RadFmt


class SkyData(object):
    """class to generate sky conditions

    This class provides an interface to generate sky data using the perez sky
    model

    Parameters
    ----------
    wea: str np.array
        path to epw, wea, .npy file or np.array, or .npz file, if loc not set
        attempts to extract location data (if needed).
    loc: tuple, optional
        location data given as lat, lon, mer with + west of prime meridian
        overrides location data in wea (but not in sunfield)
    skyro: float, optional
        angle in degrees counter-clockwise to rotate sky
        (to correct model north, equivalent to clockwise rotation of scene)
    ground_fac: float, optional
        ground reflectance
    intersky: bool, optional
        include interreflection between ground and sky (mimics perezlum.cal,
        not present in gendaymtx)
    skyres: float, optional
        approximate square patch size in degrees
    minalt: float, optional
        minimum solar altitude for daylight masking
    mindiff: float, optional
        minumum diffuse horizontal irradiance for daylight masking
    """

    def __init__(self, wea, loc=None, skyro=0.0, ground_fac=0.2, intersky=True,
                 skyres=12.0, minalt=2.0, mindiff=5.0, mindir=0.0):
        self.skyres = skyres
        self.intersky = intersky
        if wea is None:
            ground_fac = 1
            minalt = 0
            mindiff = 0
            mindir = 0
            skyro = 0
            loc = None
        self.ground_fac = ground_fac
        if loc is None and wea is not None:
            try:
                loc = skycalc.get_loc_epw(wea)
            except ValueError:
                pass
        #: location and sky rotation information
        self._loc = loc
        self._minalt = minalt
        self._mindiff = mindiff
        self._mindir = mindir
        self._skyro = skyro
        self._sunproxy = None
        self.skydata = wea
        self.mask = None

    @property
    def skyres(self):
        return self._skyres

    @skyres.setter
    def skyres(self, s):
        self._skyres = int(np.floor(90/s)*2)

    @property
    def skyro(self):
        """sky rotation (in degrees, ccw)"""
        return self._skyro

    @property
    def loc(self):
        """lot, lon, mer (in degrees, west is positive)"""
        return self._loc

    @property
    def skydata(self):
        """sun position and dirnorm diffhoriz"""
        return self._skydata

    @skydata.setter
    def skydata(self, wea):
        """calculate sky matrix data and store in self._skydata

        Parameters
        ----------
        wea: str, np.array
            This method takes either a file path or np.array. File path can
            point to a wea, epw, tsv file, or npz. tsv array must be one of
            the following:
            - 4 col: alt, az, dir, diff
            - 5 col: dx, dy, dz, dir, diff
            - 5 col: m, d, h, dir, diff
            npz file will rewrite skyres
        """
        npzdata = self._load(wea)
        if npzdata:
            skydata, smtx, sun, daymask = npzdata
        elif wea is not None:
            skydata, md, td = self.format_skydata(wea)
            minz = np.sin(self._minalt * np.pi / 180)
            daymask = np.logical_and(skydata[:, 2] > minz,
                                     skydata[:, 4] > self._mindiff)
            daymask = np.logical_and(daymask, skydata[:, 3] > self._mindir)
            sxyz = skydata[daymask, 0:3]
            dirdif = skydata[daymask, 3:]
            if md is not None:
                md = md[daymask]
            if hasattr(td, "__len__"):
                td = td[daymask]
            smtx, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, self.skyres, md=md,
                                              ground_fac=self.ground_fac, td=td,
                                              intersky=self.intersky)
            # ratio between actual solar disc and patch
            omegar = np.square(0.2665 * np.pi * self.skyres / 180) * .5
            plum = sun[:, -1] * omegar
            sun = np.hstack((sun, plum[:, None]))
            smtx = np.hstack((smtx, grnd[:, None]))
        else:
            smtx = np.ones((1, self.skyres**2+1))
            sun = np.array([[0, 0, 1, 0, 0]])
            daymask = np.array([True])
            skydata = np.array([[0, 0, 1, 0, 1]])
        self._smtx = smtx
        self._sun = sun
        self._daymask = daymask
        self._skydata = skydata
        self.sunproxy = skydata[daymask, 0:3]
        self._daysteps = smtx.shape[0]

    def write(self, name="skydata", scene=None, compressed=True):
        file = f"{name}.npz"
        if scene is not None:
            file = f"{scene.outdir}/{file}"
        kws = dict(skydata=self._skydata, smtx=self._smtx, sun=self._sun,
                   daymask=self._daymask, loc=self._loc)
        if compressed:
            np.savez_compressed(file, **kws)
        else:
            np.savez(file, **kws)

    def _load(self, file):
        try:
            result = np.load(file)
        except (ValueError, TypeError, FileNotFoundError):
            return False
        else:
            skydata = result['skydata']
            smtx = result['smtx']
            sun = result['sun']
            daymask = result['daymask']
            try:
                loc = result['loc']
            except ValueError:
                self._loc = None
            else:
                self._loc = (loc[0], loc[1], int(loc[2]))
            self._skyres = int(np.sqrt(smtx.shape[1] - 1))
            return skydata, smtx, sun, daymask

    def format_skydata(self, dat):
        """process dat argument as skydata

        see sky.setter for details on argument

        Returns
        -------
        np.array
            dx, dy, dz, dir, diff
        """
        loc = self._loc
        md = None
        td = 10.9735311509
        if hasattr(dat, 'shape'):
            skydat = dat
        else:
            try:
                skydat = np.loadtxt(dat)
            except ValueError:
                if loc is None:
                    loc = skycalc.get_loc_epw(dat)
                skydat = skycalc.read_epw(dat)
            try:
                td = skycalc.read_epw_full(dat, ["t_dewpoint", ]).ravel()
            except (TypeError, IndexError, ValueError):
                pass
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
                md = skydat[:, 0:2].astype(int)
                skydat = np.hstack((xyz, skydat[:, 3:]))
        else:
            raise ValueError('input data should be one of the following:'
                             '\n4 col: alt, az, dir, diff'
                             '\n5 col: dx, dy, dz, dir, diff'
                             '\n5 col: m, d, h, dir, diff')
        return skydat, md, td

    @property
    def daysteps(self):
        return self._daysteps

    @property
    def daymask(self):
        """shape (len(skydata),) boolean array masking timesteps when sun is
        below horizon"""
        return self._daymask

    @property
    def mask(self):
        """an additional mask for smtx data"""
        return self._mask

    @property
    def fullmask(self):
        return self._fullmask

    @property
    def maskindices(self):
        return self._maskindices

    @mask.setter
    def mask(self, m):
        """if m has length = daysteps, sets mask directly as bool array. if
        m has length = skydata, sets mask from daysteps of m as bool array.
        otherwise, assumes m is an index array (indexed by row of skydata) and
        sets all indices of m withing daysteps to True. reset with m=None
        """
        if m is None:
            m = np.full(np.sum(self.daymask), True)
        if len(m) == self.daysteps:
            self._mask = np.asarray(m, bool)
        elif len(m) == len(self.skydata):
            self._mask = np.asarray(m, bool)[self.daymask]
        else:
            self._mask = np.full(self.daysteps, False)
            self._mask[self.masked_idx(m)] = True
        self._fullmask = np.copy(self.daymask)
        self._fullmask[self.daymask] = self._mask
        self._maskindices = np.arange(len(self._fullmask))[self._fullmask]

    @property
    def smtx(self):
        """shape (np.sum(daymask), skyres**2 + 1) coefficients for each sky
        patch each row is a timestep, coefficients exclude sun"""
        return self._smtx[self.mask]

    @property
    def sun(self):
        """shape (np.sum(daymask), 5) sun position (index 0,1,2) and
        coefficients for sun at each timestep assuming the true solid angle of
        the sun (index 3) and the weighted value for the sky patch (index 4)."""
        return self._sun[self.mask]

    @property
    def sunproxy(self):
        """corresponding sky bin for each sun position in daymask"""
        return self._sunproxy[self.mask]

    @sunproxy.setter
    def sunproxy(self, sxyz):
        self._sunproxy = translate.xyz2skybin(sxyz, self.skyres)

    def smtx_patch_sun(self, includesky=True):
        """generate smtx with solar energy applied to proxy patch
        for directly applying to skysampler data (without direct sun components)
        can also be used in a partial mode (with sun view / without sun
        reflection.)"""
        if includesky:
            wsun = np.copy(self.smtx)
        else:
            wsun = np.zeros_like(self.smtx)
        r = wsun.shape[0]
        wsun[range(r), self.sunproxy] += self.sun[:, 4]
        return wsun

    def header(self):
        """generate image header string"""
        try:
            hdr = "LOCATION= lat: {} lon: {} tz: {} ro: {}".format(*self.loc,
                                                                   self.skyro)
        except TypeError:
            hdr = "LOCATION= None"
        return hdr

    def fill_data(self, x, fill_value=0.0):
        """
        Parameters
        ----------
        x: np.array
            first axis size = len(self.daymask[self.mask])
        fill_value: Union[int, float], optional
            value in padded array

        Returns
        -------
        np.array
            data in x padded with fill value to original shape of skydata
        """
        mask = np.copy(self.daymask)
        mask[self.daymask] = self.mask
        px = np.full((self.skydata.shape[0], *x.shape[1:]), fill_value,
                     dtype=x.dtype)
        px[mask] = x
        return px

    def masked_idx(self, i):
        j = np.searchsorted(self._maskindices, i)
        return j[self._maskindices[j] == i]

    def sky_description(self, i, prefix="skydata", grid=False, sun=True,
                        ground=True):
        """generate radiance scene files to directly render sky data at index i

        Parameters
        ----------
        i: int
            index of sky vector to generate (indexed from skydata, not daymask)
        prefix: str, optional
            name/path for output files
        grid: bool, optional
            render sky patches with grid lines
        sun: bool, optional
            include sun source in rad file

        Returns
        -------
        str
            basename of 3 files written: prefix_i (.rad, .cal, and .dat)
            .cal and .dat must be located in RAYPATH (which can include .)
            or else edit the .rad file to explicitly point to their locations.
            note that if grid is True, the sky will not be accurate, so only
            use this for illustrative purposes.

        Raises
        ------
        IndexError
            if i is not in masked indices
        """
        mi = self.masked_idx(i).item()
        outf = f"{prefix}_{i:04d}"
        f = open(f"{outf}.rad", 'w')
        if grid:
            fun = "grid"
        else:
            fun = "noop"
        f.write(f"void brightdata skyfunc 4 {fun} {outf}.dat {outf}.cal bin 0 "
                "0\nskyfunc glow skyglow 0 0 4 1 1 1 0\n"
                "skyglow source sky 0 0 4 0 0 1 180\n")
        if ground:
            f.write("skyglow source ground 0 0 4 0 0 -1 180\n")
        if sun:
            c = (self.sun[mi, 3], self.sun[mi, 3], self.sun[mi, 3])
            f.write(RadFmt.get_sundef(self.sun[mi, 0:3], c))
        f.close()
        f = open(f"{outf}.cal", 'w')
        f.write(f"side:{self.skyres};\n{translate.scbinscal}")
        f.close()
        data = self.smtx[mi]
        nrbins = data.size
        header = "1\n0 {} {}\n".format(nrbins - 1, nrbins)
        np.savetxt(f"{outf}.dat", data, delimiter="\n", header=header,
                   comments="")
        return outf
