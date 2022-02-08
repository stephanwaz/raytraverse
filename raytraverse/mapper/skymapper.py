# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from matplotlib.path import Path

from raytraverse import translate
from raytraverse.sky import skycalc
from raytraverse.mapper.mapper import Mapper
from raytraverse.mapper.angularmixin import AngularMixin


class SkyMapper(AngularMixin, Mapper):
    """translate between world direction vectors and normalized UV space for a
    given view angle. pixel projection yields equiangular projection

    Parameters
    ----------
    loc: any, optional
        can be a number of formats:

            1. either a numeric iterable of length 3 (lat, lon, mer)
               where lat is +west and mer is tz*15 (matching gendaylit).
            2. an array (or tsv file loadable with np.loadtxt) of shape
               (N,3), (N,4), or (N,5):

                    a. 2 elements: alt, azm (angles in degrees)
                    b. 3 elements: dx,dy,dz of sun positions
                    c. 4 elements: alt, azm, dirnorm, diffhoriz (angles in degrees)
                    d. 5 elements: dx, dy, dz, dirnorm, diffhoriz.

            3. path to an epw or wea formatted file
            4. None (default) all possible sun positions are considered
               self.in_solarbounds always returns True

        in the case of a geo location, sun positions are considered valid when
        in the solar transit for that location. for candidate options, sun
        positions are drawn from this set (with one randomly chosen from all
        candidates within bin.
    skyro: float, optional
        counterclockwise sky-rotation in degrees (equivalent to clockwise
        project north rotation)
    sunres: float, optional
        initial sampling resolution for suns
    name: str, optional

    """

    _flipu = False
    _xsign = 1

    def __init__(self, loc=None, skyro=0.0, sunres=20.0, name='sky',
                 jitterrate=0.5):
        self._viewangle = 180.0
        self._chordfactor = 1.0
        self._ivm = None
        self.skyro = skyro
        self.sunres = sunres

        super().__init__(name=name, aspect=1, jitterrate=jitterrate)
        self.loc = loc

    @property
    def skyro(self):
        return self._skyro

    @skyro.setter
    def skyro(self, ro):
        self._skyro = ro
        self._skyromtx = translate.rmtx_elem(ro)

    @property
    def sunres(self):
        return self._sunres

    @sunres.setter
    def sunres(self, s):
        self._sunres = int(np.floor(180/s))

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, location):
        if location is None:
            self._candidates = None
            self._loc = None
            self.in_solarbounds_uv = self._test_uv_none
            self.solar_grid_uv = self._solar_grid_uv
        elif type(location) == str:
            self._load_sun_data(location)
            self.in_solarbounds_uv = self._test_uv_candidates
            self.solar_grid_uv = self._solar_grid_uv_candidates
        elif (np.asarray(location).size > 3 or
              len(np.asarray(location).shape) == 2):
            self._norm_input_data(np.asarray(location))
            self.in_solarbounds_uv = self._test_uv_candidates
            self.solar_grid_uv = self._solar_grid_uv_candidates
        else:
            self._load_sun_border(location)
            self.in_solarbounds_uv = self._test_uv_boundary
            self.solar_grid_uv = self._solar_grid_uv

    @property
    def solarbounds(self):
        return self._solarbounds

    @property
    def candidates(self):
        return self._candidates

    def in_solarbounds(self, xyz, level=0, include='center'):
        """for checking if src direction is in solar transit

        Parameters
        ----------
        xyz: np.array
            source directions
        level: int
            for determining patch size, 2**level resolution from sunres
        include: {'center', 'all', 'any'}, optional
            boundary test condition. 'center' tests uv only, 'all' requires
            for corners of box centered at uv to be in, 'any' requires atleast
            one corner. 'any' is the least restrictive and 'all' is the most,
            but with increasing levels 'any' will exclude more positions while
            'all' will exclude less (both approaching 'center' as level -> N)

        Returns
        -------
        result: np.array
            Truth of ray.src within solar transit
        """
        uv = self.xyz2uv(xyz)
        return self.in_solarbounds_uv(uv, level=level, include=include)

    def shape(self, level=0):
        s = self.sunres * 2**level
        return s, s

    def solar_grid(self, jitter=True, level=0, masked=True):
        """generate a grid of solar positions

        Parameters
        ----------
        jitter: bool, optional
            if None, use the instance default, if True jitters point samples
            within stratified grid
        level: int, optional
            sets the resolution of the grid as a power of 2 from sunress
        masked: bool, optional
            apply in_solarbounds to suns before returning

        Returns
        -------
        np.array
            shape (N, 3)
        """
        return self.uv2xyz(self.solar_grid_uv(jitter, level, masked))

    def _solar_grid_uv(self, jitter=True, level=0, masked=True):
        """add a grid of UV coordinates

        Parameters
        ----------
        jitter: bool, optional
            if None, use the instance default, if True jitters point samples
            within stratified grid
        level: int, optional
            sets the resolution of the grid as a power of 2 from ptres
        masked: bool, optional
            apply in_solarbounds before returning

        Returns
        -------
        np.array
            shape (N, 2)
        """
        shape = self.shape(level)
        idx = np.arange(np.product(shape))
        uv = self.idx2uv(idx, shape, jitter)
        if masked:
            return uv[self.in_solarbounds_uv(uv)]
        else:
            return uv

    def _solar_grid_uv_candidates(self, jitter=True, level=0, masked=True):
        """add a grid of UV coordinates

        Parameters
        ----------
        jitter: bool, optional
            jitters weighting of condidate selection (not centered)
        level: int, optional
            sets the resolution of the grid as a power of 2 from ptres
        masked: bool, optional
            apply in_solarbounds before returning

        Returns
        -------
        np.array
            shape (N, 2)
        """
        uvsize, _ = self.shape(level)
        cbins = translate.uv2bin(self._candidates, uvsize)
        sbins = np.arange(uvsize**2)
        suv = self._solar_grid_uv(jitter=False, level=level, masked=False)
        if masked:
            mask = self.in_solarbounds_uv(suv, level=level)
            sbins = sbins[mask]
            suv = suv[mask]
        sunsuv = []
        badsun = np.array([1.5, .5])
        for b, uv in zip(sbins, suv):
            # choose sun position, priortizing values closer to center of bin
            # or closest to random point with jitter=True
            candidates = self._candidates[cbins == b]
            if candidates.shape[0] == 1:
                a = candidates[0]
            elif candidates.size > 1:
                binc = self.uv2xyz(uv)
                bp = np.linalg.norm(self.uv2xyz(candidates) - binc,
                                    axis=1)
                if jitter:
                    a = np.random.default_rng().choice(candidates, axis=0,
                                                       p=bp/np.sum(bp))
                else:
                    a = candidates[np.argmin(bp)]
            else:
                a = badsun
            sunsuv.append(a)
        return np.array(sunsuv)

    def _load_sun_border(self, loc):
        self._candidates = None
        self._loc = loc
        jun = np.arange('2020-06-21', '2020-06-22', 5,
                        dtype='datetime64[m]')
        dec = np.arange('2020-12-21', '2020-12-22', 5,
                        dtype='datetime64[m]')
        jxyz = skycalc.sunpos_xyz(jun, *loc)
        dxyz = skycalc.sunpos_xyz(dec, *loc)
        # to close paths well outside UV box
        extrema = np.array([(1000, 0, 0), (-1000, 0, 0)])
        jxyz = (self._skyromtx@jxyz.T).T
        dxyz = (self._skyromtx@dxyz.T).T
        extrema = (self._skyromtx@extrema.T).T
        juv = self.xyz2uv(jxyz[jxyz[:, 2] > 0])
        duv = self.xyz2uv(dxyz[dxyz[:, 2] > 0])[::-1]
        # handle (ant)arctic circles
        if duv.size == 0:
            dxyz = (self._skyromtx@np.array([(0, -1000, 0)]).T).T
            duv = self.xyz2uv(dxyz)
        if juv.size == 0:
            jxyz = (self._skyromtx@np.array([(0, 1000, 0)]).T).T
            juv = self.xyz2uv(jxyz)
        euv = self.xyz2uv(extrema)
        bounds = np.vstack((euv[0:1], juv, euv[1:], duv, euv[0:1]))
        bpath = Path(bounds, closed=True)
        self._solarbounds = bpath

    def _load_sun_data(self, wea):
        try:
            dat = np.atleast_2d(np.loadtxt(wea))
            self._norm_input_data(dat)
        except ValueError:
            loc = skycalc.get_loc_epw(wea)
            wdat = skycalc.read_epw(wea)
            times = skycalc.row_2_datetime64(wdat[:, 0:3])[wdat[:, 3] > 0]
            cxyz = skycalc.sunpos_xyz(times, *loc)
            self._loc = loc
            self._candidates = self.xyz2uv((self._skyromtx@cxyz.T).T)
            self._solarbounds = None

    def _norm_input_data(self, dat):
        if dat.shape[1] == 2:
            cxyz = translate.aa2xyz(dat[dat[:, 0] > 0])
        elif dat.shape[1] == 3:
            cxyz = translate.norm(dat[dat[:, 2] > 0])
        elif dat.shape[1] == 4:
            cxyz = translate.aa2xyz(dat[np.logical_and(dat[:, 0] > 0,
                                                       dat[:, 2] > 0), 0:2])
        elif dat.shape[1] == 5:
            cxyz = translate.norm(dat[np.logical_and(dat[:, 2] > 0,
                                                     dat[:, 3] > 0), 0:3])
        else:
            raise ValueError("Data has wrong shape, must be (N, {2,3,4,5})")
        self._loc = None
        self._candidates = self.xyz2uv((self._skyromtx@cxyz.T).T)
        self._solarbounds = None

    def _test_uv_candidates(self, uv, level=0, **kwargs):
        uvsize = self.sunres * 2**level
        cbins = np.unique(translate.uv2bin(self._candidates, uvsize))
        uvbins = translate.uv2bin(uv, uvsize)
        return np.isin(uvbins, cbins)

    def _test_uv_boundary(self, uv, level=0, include='center'):
        """ for checking if src direction is in solar transit when
        self._solarbounds is set

        Parameters
        ----------
        uv: np.array
            source directions
        level: int
            for determining patch size, 2**level resolution from sunres
        include: {'center', 'all', 'any'}, optional
            boundary test condition. 'center' tests uv only, 'all' requires
            for corners of box centered at uv to be in, 'any' requires atleast
            one corner. 'any' is the least restrictive and 'all' is the most,
            but with increasing levels 'any' will exclude more positions while
            'all' will exclude less (both approaching 'center' as level -> N)

        Returns
        -------
        result: np.array
            Truth of ray.src within solar transit
        """
        uvs = uv.reshape(-1, 2)
        if include == 'center':
            result = self._solarbounds.contains_points(uvs)
        else:
            offsets = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])
            offsets *= 1 / (self.sunres * 2**(level + 1))
            uvs = uv[:, None, :] + offsets[None, ...]
            result = self._solarbounds.contains_points(uvs.reshape(-1, 2))
            # include should be any or all -> np.all, np.any
            result = getattr(np, include)(result.reshape(-1, 4), 1)
        return result

    @staticmethod
    def _test_uv_none(uv, **kwargs):
        """default no uv solar position filtering (see loc.setter)"""
        return np.full(uv.shape[0], True)
