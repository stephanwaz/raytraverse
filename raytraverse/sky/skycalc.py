# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for loading sky data and computing sun position"""
import os
import datetime
import re

import numpy as np
from scipy.interpolate import interp1d
from skyfield.api import Topos, utc, Loader

from raytraverse import translate

load = Loader(os.path.dirname(translate.__file__))
planets = load('de421.bsp')
sun = planets['sun']
earth = planets['earth']


def read_epw(epw):
    """read daylight sky data from epw or wea file

    Returns
    -------
    out: np.array
        (month, day, hour, dirnorn, difhoriz)
    """
    f = open(epw, 'r')
    lines = f.readlines()
    f.close()
    hours = [re.split(r'[ \t,]+', i) for i in lines if re.match(r"\d.*", i)]
    data = []
    for h in hours:
        if len(h) > 23:
            dp = [h[1], h[2], h[3], h[14], h[15]]
            hoff = .5
        else:
            dp = [h[0], h[1], h[2], h[3], h[4]]
            hoff = 0
        data.append([int(i.strip()) for i in dp[0:2]] + [float(dp[2]) - hoff] +
                    [float(i.strip()) for i in dp[3:]])
    return np.array(data)


col_headers = {'year': 0, 'month': 1, 'day': 2, 'hour': 3, 'minute': 4,
               'note': 5, 't_drybulb': 6, 't_dewpoint': 7, 'rh': 8, 'asp': 9,
               'ext_hor_rad': 10, 'ext_dir_norm_rad': 11,
               'hor_infr_rad_int': 12, 'global_hor_rad': 13,
               'dir_norm_rad': 14, 'dif_hor_rad': 15, 'global_hor_illum': 16,
               'dir_norm_illum': 17, 'diff_hor_illum': 18, 'zenith_lum': 19,
               'wind_dir': 20, 'wind_spd': 21, 'sky_cover': 22,
               'opaque_sky_cover': 23, 'visibility': 24, 'ceil_height': 25,
               'weather_obs': 26, 'weather_codes': 27, 'precip_water': 28,
               'aerosol_optical_depth': 29, 'snow_depth': 30,
               'days_last_snow': 31, 'albedo': 32, 'liquid_precip_depth': 33,
               'liquid_precip_quant': 34}


col_indices = dict(zip(col_headers.values(), col_headers.keys()))

def read_epw_full(epw, columns=None):
    """

    Parameters
    ----------
    epw
    columns: list, optional
        integer indices or keys of columns to return

    Returns
    -------
    requested columns from epw as np.array shape (8760, N)
    """
    f = open(epw, 'r')
    lines = f.readlines()
    f.close()
    hours = [re.split(r'[ \t,]+', i) for i in lines if re.match(r"\d.*", i)]
    data = np.array(hours).T
    data[5] = '0'
    data = data.astype(float)
    # correct hour offset as instantaneous
    data[3] -= 0.5
    if columns is not None:
        c = []
        for i in columns:
            if i in col_headers:
                c.append(col_headers[i])
            elif i in col_indices:
                c.append(i)
            else:
                raise ValueError(f"Column {i} is not a valid key: {col_indices}")
        columns = c
    return data[columns].T


def get_loc_epw(epw, name=False):
    """get location from epw or wea header"""
    try:
        f = open(epw)
        hdr = f.readlines()[0:8]
        f.close()
        if len(hdr[0].split(",")) > 5:
            hdr = hdr[0].split(",")
            lat = hdr[-4]
            lon = hdr[-3]
            tz = float(hdr[-2])
            lon = str(-float(lon))
            tz = str(int(-tz*15))
            loc = hdr[1]
            elev = hdr[-1].strip()
        else:
            lat = [i.split()[-1] for i in hdr if re.match(r"latitude.*", i)][0]
            lon = [i.split()[-1] for i in hdr if re.match(r"longitude.*", i)][0]
            tz = [i.split()[-1] for i in hdr if re.match(r"time_zone.*", i)][0]
            loc = [i.split()[-1] for i in hdr if re.match(r"place.*", i)][0]
            try:
                elev = [i.split()[-1] for i in hdr if
                        re.match(r"site_elevation.*", i)][0]
            except IndexError:
                elev = '0.0'
    except Exception as e:
        raise ValueError("bad epw header", e)
    if name:
        return float(lat), float(lon), int(tz), loc, elev
    else:
        return float(lat), float(lon), int(tz)


def sunpos_utc(timesteps, lat, lon, builtin=True):
    """Calculate sun position with local time

    Calculate sun position (altitude, azimuth) for a particular location
    (longitude, latitude) for a specific date and time (time is in UTC)

    Parameters
    ----------
    timesteps : np.array(datetime.datetime)
    lon : float
        longitude in decimals. West is +ve
    lat : float
        latitude in decimals. North is +ve
    builtin: bool
        use skyfield builtin timescale

    Returns
    -------
    (skyfield.units.Angle, skyfield.units.Angle)
    altitude and azimuth in degrees
    """
    dt = np.apply_along_axis(lambda x: x[0].replace(tzinfo=utc), 1,
                             timesteps.reshape(-1, 1))
    # use radiance +west longitude coordinates
    loc = earth + Topos(lat, -lon)
    # faster but requires updating periodically
    ts = load.timescale(builtin=builtin)
    astro = loc.at(ts.utc(dt)).observe(sun)
    app = astro.apparent()
    return app.altaz('standard')[0:2]


def row_2_datetime64(ts, year=2020):
    ts = np.asarray(ts).astype(float)
    if len(ts.shape) == 1:
        if len(ts) < 4:
            hm = np.modf(ts[2])
            ts = np.concatenate((ts[0:2], [hm[1], hm[0]*60]))
        st = ['{}-{:02.00f}-{:02.00f}T{:02.00f}:{:02.00f}'.format(year, *ts), ]
    else:
        if ts.shape[1] < 4:
            hm = np.modf(ts[:, 2:3])
            ts = np.hstack((ts[:, 0:2], hm[1], hm[0]*60))
        st = ['{}-{:02.00f}-{:02.00f}T{:02.00f}:{:02.00f}'.format(year, *t)
              for t in ts]
    return np.array(st).astype('datetime64[m]')


def datetime64_2_datetime(timesteps, mer=0.):
    """convert datetime representation and offset for timezone

    Parameters
    ----------
    timesteps: np.array(np.datetime64)
    mer: float
        Meridian of the time zone. West is +ve

    Returns
    -------
    np.array(datetime.datetime)

    """
    tz = mer/15.
    dt = (timesteps + np.timedelta64(int(tz*60), 'm')).astype(datetime.datetime)
    return dt


def sunpos_degrees(timesteps, lat, lon, mer, builtin=True, ro=0.0):
    """Calculate sun position with local time

    Calculate sun position (altitude, azimuth) for a particular location
    (longitude, latitude) for a specific date and time (time is in local time)

    Parameters
    ----------
    timesteps : np.array(np.datetime64)
    lon : float
        longitude in decimals. West is +ve
    lat : float
        latitude in decimals. North is +ve
    mer: float
        Meridian of the time zone. West is +ve
    builtin: bool, optional
        use skyfield builtin timescale
    ro: float, optional
        ccw rotation (project to true north) in degrees

    Returns
    -------
    np.array([float, float])
        Sun position as (altitude, azimuth) in degrees
    """
    dt = datetime64_2_datetime(timesteps, mer=mer)
    alt, az = sunpos_utc(dt, lat, lon, builtin)
    # south is az=0
    return np.column_stack([alt.degrees, az.degrees - 180 - ro])


def sunpos_radians(timesteps, lat, lon, mer, builtin=True, ro=0.0):
    """Calculate sun position with local time

    Calculate sun position (altitude, azimuth) for a particular location
    (longitude, latitude) for a specific date and time (time is in local time)

    Parameters
    ----------
    timesteps : np.array(np.datetime64)
    lon : float
        longitude in decimals. West is +ve
    lat : float
        latitude in decimals. North is +ve
    mer: float
        Meridian of the time zone. West is +ve
    builtin: bool
        use skyfield builtin timescale
    ro: float, optional
        ccw rotation (project to true north) in radians

    Returns
    -------
    np.array([float, float])
        Sun position as (altitude, azimuth) in radians
    """
    dt = datetime64_2_datetime(timesteps, mer=mer)
    alt, az = sunpos_utc(dt, lat, lon, builtin)
    # south is az=0
    return np.column_stack([alt.radians, az.radians - np.pi - ro])


def sunpos_xyz(timesteps, lat, lon, mer, builtin=True, ro=0.0):
    """Calculate sun position with local time

    Calculate sun position (altitude, azimuth) for a particular location
    (longitude, latitude) for a specific date and time (time is in local time)

    Parameters
    ----------
    timesteps : np.array(np.datetime64)
    lon : float
        longitude in decimals. West is +ve
    lat : float
        latitude in decimals. North is +ve
    mer: float
        Meridian of the time zone. West is +ve
    builtin: bool
        use skyfield builtin timescale
    ro: float, optional
        ccw rotation (project to true north) in degrees

    Returns
    -------
    np.array
        Sun position as (x, y, z)
    """
    dt = datetime64_2_datetime(timesteps, mer=mer)
    alt, az = sunpos_utc(dt, lat, lon, builtin)
    az = az.radians - ro*np.pi/180
    # translate to spherical rhs before calling translate.tp2xyz
    thetaphi = np.column_stack([np.pi/2 - alt.radians, np.pi/2 - az])
    return translate.tp2xyz(thetaphi)


def generate_wea(ts, wea, interp='linear'):
    skydat = read_epw(wea)
    wtimes = row_2_datetime64(skydat[:, 0:3]).astype(int)
    qtimes = row_2_datetime64(ts).astype(int)
    fdir = interp1d(wtimes, skydat[:, 3], kind=interp)
    fdif = interp1d(wtimes, skydat[:, 4], kind=interp)
    idir = fdir(qtimes)[:, None]
    idif = fdif(qtimes)[:, None]
    return np.hstack((ts, idir, idif))


# Below is an implementation of perez all weather sky model:

# Perez, R., R. Seals, and J. Michalsky. “All-Weather Model for Sky
# Luminance Distribution—Preliminary Configuration and Validation.”
# Solar Energy 50, no. 3 (March 1, 1993): 235–45.
# https://doi.org/10.1016/0038-092X(93)90017-I.

# Code adapted to and tested against the gendaylit and genskyvec programs
# of Radiance:

# The Radiance Software License, Version 1.0
#
# Copyright (c) 1990 - 2018 The Regents of the University of California,
# through Lawrence Berkeley National Laboratory.   All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#         notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#
# 3. The end-user documentation included with the redistribution,
#           if any, must include the following acknowledgment:
#             "This product includes Radiance software
#                 (http://radsite.lbl.gov/)
#                 developed by the Lawrence Berkeley National Laboratory
#               (http://www.lbl.gov/)."
#       Alternately, this acknowledgment may appear in the software itself,
#       if and wherever such third-party acknowledgments normally appear.
#
# 4. The names "Radiance," "Lawrence Berkeley National Laboratory"
#       and "The Regents of the University of California" must
#       not be used to endorse or promote products derived from this
#       software without prior written permission. For written
#       permission, please contact radiance@radsite.lbl.gov.
#
# 5. Products derived from this software may not be called "Radiance",
#       nor may "Radiance" appear in their name, without prior written
#       permission of Lawrence Berkeley National Laboratory.
#
# THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED.   IN NO EVENT SHALL Lawrence Berkeley National Laboratory OR
# ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ====================================================================
#
# This software consists of voluntary contributions made by many
# individuals on behalf of Lawrence Berkeley National Laboratory.   For more
# information on Lawrence Berkeley National Laboratory, please see
# <http://www.lbl.gov/>.

# gendaylit source copyright:

# Copyright (c) 1994,2006 *Fraunhofer Institut for Solar Energy Systems
# Heidenhofstr. 2, D-79110 Freiburg, Germany
# *Agence de l'Environnement et de la Maitrise de l'Energie
# Centre de Valbonne, 500 route des Lucioles, 06565 Sophia
# Antipolis Cedex, France
# BOUYGUES
# 1 Avenue Eugene Freyssinet, Saint-Quentin-Yvelines, France
# print colored output if activated in command line (-C).
# Based on model from A. Diakite, TU-Berlin. Implemented by J. Wienold,
# August 26 2018
# */

perez_constants = {
    'cats': np.array([1, 1.065, 1.23, 1.50, 1.95, 2.80, 4.50, 6.20, 12.01]),
    'cp': np.array([1.3525, -0.2576, -0.2690, -1.4366, -0.7670,
                    0.0007, 1.2734, -0.1233, 2.8000, 0.6004, 1.2375, 1.000,
                    1.8734, 0.6297, 0.9738, 0.2809, 0.0356, -0.1246, -0.5718,
                    0.9938, -1.2219, -0.7730, 1.4148, 1.1016, -0.2054, 0.0367,
                    -3.9128, 0.9156, 6.9750, 0.1774, 6.4477, -0.1239, -1.5798,
                    -0.5081, -1.7812, 0.1080, 0.2624, 0.0672, -0.2190, -0.4285,
                    -1.1000, -0.2515, 0.8952, 0.0156, 0.2782, -0.1812, -4.5000,
                    1.1766, 24.7219, -13.0812, -37.7000, 34.8438, -5.0000,
                    1.5218, 3.9229, -2.6204, -0.0156, 0.1597, 0.4199, -0.5562,
                    -0.5484, -0.6654, -0.2672, 0.7117, 0.7234, -0.6219, -5.6812,
                    2.6297, 33.3389, -18.3000, -62.2500, 52.0781, -3.5000,
                    0.0016, 1.1477, 0.1062, 0.4659, -0.3296, -0.0876, -0.0329,
                    -0.6000, -0.3566, -2.5000, 2.3250, 0.2937, 0.0496, -5.6812,
                    1.8415, 21.000, -4.7656, -21.5906, 7.2492, -3.5000, -0.1554,
                    1.4062, 0.3988, 0.0032, 0.0766, -0.0656, -0.1294, -1.0156,
                    -0.3670, 1.0078, 1.4051, 0.2875, -0.5328, -3.8500, 3.3750,
                    14.0000, -0.9999, -7.1406, 7.5469, -3.4000, -0.1078, -1.075,
                    1.5702, -0.0672, 0.4016, 0.3017, -0.4844, -1.00, 0.0211,
                    0.5025, -0.5119, -0.3, 0.1922, 0.7023, -1.6317, 19.0, -5.0,
                    1.2438, -1.9094, -4.0000, 0.0250, 0.3844, 0.2656, 1.0468,
                    -0.3788, -2.4517, 1.4656, -1.0500, 0.0289, 0.4260, 0.3590,
                    -0.325, 0.1156, 0.7781, 0.0025, 31.0625, -14.5, -46.1148,
                    55.375, -7.2312, 0.405, 13.35,  0.6234, 1.5, -0.6426,
                    1.8564, 0.5636]).reshape((8, 5, 4)),
    'theta': np.array([84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
                       84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
                       84, 84, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
                       72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
                       72, 72, 72, 72, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                       60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                       48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
                       48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 36, 36, 36, 36,
                       36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
                       24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 12, 12,
                       12, 12, 12, 12, 0])*np.pi/180,
    'phi': np.array([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156,
                     168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300,
                     312, 324, 336, 348, 0, 12, 24, 36, 48, 60, 72, 84, 96, 108,
                     120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252,
                     264, 276, 288, 300, 312, 324, 336, 348, 0, 15, 30, 45, 60,
                     75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240,
                     255, 270, 285, 300, 315, 330, 345, 0, 15, 30, 45, 60, 75,
                     90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255,
                     270, 285, 300, 315, 330, 345, 0, 20, 40, 60, 80, 100, 120,
                     140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 0,
                     30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 0, 60,
                     120, 180, 240, 300, 0])*np.pi/180,
    'nfc': np.array([[2.766521, 0.547665, -0.369832, 0.009237, 0.059229],
                     [3.5556, -2.7152, -1.3081, 1.0660, 0.60227]]),
    'mdays': np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    }


def coeff_lum_perez(sunz, epsilon, delta, catn):
    """matches coeff_lum_perez in gendaylit.c"""
    mide = np.logical_and(epsilon > 1.065, epsilon < 2.8)
    delta[mide] = np.maximum(delta[mide], 0.2)
    x = perez_constants['cp'][catn]
    abcde = (x[..., 0] + x[..., 1]*sunz[:, None] + delta[:, None] *
             (x[..., 2] + x[..., 3]*sunz[:, None]))
    lowe = catn == 0
    abcde[lowe, 2] = np.exp(np.power(delta[lowe] * (x[lowe, 2, 0] +
                            x[lowe, 2, 1] * sunz[lowe]),
                            x[lowe, 2, 2])) - x[lowe, 2, 3]
    abcde[lowe, 3] = (-np.exp(delta[lowe] * (x[lowe, 3, 0] +
                                             x[lowe, 3, 1] * sunz[lowe])) +
                      x[lowe, 3, 2] + delta[lowe] * x[lowe, 3, 3])
    return abcde


def perez_apply_coef(coefs, cgamma, dz):
    c = coefs[:, None, :]
    z = np.maximum(dz, 0.01)
    gamma = np.arccos(cgamma)
    lum = ((1 + c[..., 0]*np.exp(c[..., 1]/z[None, :])) *
           (1 + c[..., 2]*np.exp(c[..., 3]*gamma) +
            c[..., 4]*np.square(cgamma)))
    return lum


def perez_lum_raw(tp, dz, sunz, coefs):
    """matches calc_rel_lum_perez in gendaylit.c"""
    cg = (np.cos(sunz[:, None])*np.cos(tp[:, 0]) +
          np.sin(sunz[:, None])*np.sin(tp[:, 0]) *
          np.cos(tp[:, 1]))
    return perez_apply_coef(coefs, cg, dz)


def perez_lum(xyz, coefs):
    """matches perezlum.cal"""
    sxyz = coefs[:, 7:]
    cgamma = np.sum(xyz * sxyz[:, None, :], -1)
    rawlum = perez_apply_coef(coefs[:, 2:7], cgamma, xyz[:, 2])
    c = coefs[:, None, 0:2]
    swght = np.power(xyz[:, 2] + 1.01, 10)[None, :]
    gwght = np.power(xyz[:, 2] + 1.01, -10)[None, :]
    return (swght * c[..., 0] * rawlum + gwght * c[..., 1]) / (swght + gwght)


def scale_efficacy(dirdif, sunz, csunz, skybright, catn, td=10.9735311509):
    abcdf = np.array([[97.24, 107.22, 104.97, 102.39, 100.71, 106.42, 141.88,
                       152.23, 0],
                      [-0.46, 1.15, 2.96, 5.59, 5.94, 3.83, 1.90, 0.35, 0],
                      [12.0, 0.59, -5.53, -13.95, -22.75, -36.15, -53.24,
                       -45.27, 0],
                      [-8.91, -3.95, -8.77, -13.90, -23.74, -28.83, -14.03,
                       -7.98, 0]]).T
    abcdd = np.array([[57.20, 98.99, 109.83, 110.34, 106.36, 107.19, 105.75,
                       101.18, 0],
                      [-4.55, -3.46, -4.90, -5.84, -3.97, -1.25, 0.77, 1.58, 0],
                      [-2.98, -1.21, -1.71, -1.99, -1.75, -1.51, -1.26, -1.10,
                       0],
                      [117.12, 12.38, -8.81, -4.56, -6.16, -26.73, -34.44,
                       -8.29, 0]]).T
    precwater = np.broadcast_to(np.exp(0.07*td - .075), dirdif.shape[0])
    effimultd = np.stack((np.ones(dirdif.shape[0]), precwater,
                          np.exp(5.73*sunz - 5), skybright)).T
    effimultf = np.stack((np.ones(dirdif.shape[0]), precwater,
                          csunz, np.log(skybright))).T
    ef = np.sum(abcdf[catn]*effimultf, 1)
    # this can go negative due to extrapolation, check if this matches
    # gendaylit code (which does set these conditions to zero).
    ed = np.maximum(np.sum(abcdd[catn]*effimultd, 1), 0)
    directi = dirdif[:, 0]*ed
    diffusei = dirdif[:, 1]*ef
    return directi, diffusei


def perez(sxyz, dirdif, md=None, ground_fac=0.2, td=10.9735311509):
    """compute perez coefficients

    Notes
    -----
    to match the results of gendaylit, for a given sun angle without associated
    date, the assumed eccentricity is 1.035020


    Parameters
    ----------
    sxyz: np.array
        (N, 3) dx, dy, dz sun position
    dirdif: np.array
        (N, 2) direct normal, diffuse horizontal W/m^2
    md: np.array, optional
        (N, 2) month day of sky calcs (for more precise eccentricity calc)
    ground_fac: float
        scaling factor (reflectance) for ground brightness
    td: np.array, float
        (N,) dew point temperature in C

    Returns
    -------
    perez: np.array
        (N, 10) diffuse normalization, ground brightness, perez coefs, x, y, z
    """
    n = sxyz.shape[0]
    # match constants from gendaylit.c
    sole = 1367
    dn = 0
    if md is not None:
        dn = perez_constants['mdays'][md[:, 0] - 1] + md[:, 1]
    da = 2*np.pi*(dn - 1)/365
    eccentricity = (1.00011 + 0.034221*np.cos(da) + 0.00128*np.sin(da) +
                    0.000719*np.cos(2*da) + 0.000077*np.sin(2*da))
    alt = np.arcsin(sxyz[:, 2])
    sunz = np.pi/2 - alt
    csunz = np.cos(sunz)
    sunz3 = 1.041*np.power(sunz, 3)
    airmass = 1/(csunz + 0.15*np.exp(-np.log(93.885 - sunz*180/np.pi)*1.253))
    skybright = dirdif[:, 1]*airmass/(sole * eccentricity)
    skyclear = ((dirdif[:, 1] + dirdif[:, 0])/dirdif[:, 1] + sunz3)/(1 + sunz3)
    catn = np.minimum(np.searchsorted(perez_constants['cats'], skyclear,
                                      side='right') - 1, 7)
    directi, diffusei = scale_efficacy(dirdif, sunz, csunz,
                                       skybright, catn, td)
    cperez = coeff_lum_perez(sunz, skyclear, skybright, catn)
    tp = np.stack((perez_constants['theta'], perez_constants['phi'])).T
    dz = np.cos(tp[:, 0])
    normvals = perez_lum_raw(tp, dz, sunz, cperez)
    normc = np.sum(normvals * dz, 1)*2*np.pi/145
    diffnorm = diffusei/normc/179
    # half_sun_angle = 0.2665
    solarrad = directi/(2*np.pi*(1 - np.cos(0.2665*np.pi/180)))/179
    zenithbr = perez_lum_raw(np.array([0, 0])[None, :], np.array([1]),
                             sunz, cperez).flatten()
    zenithbr *= diffnorm
    inter = (skyclear <= 6)
    normsc = perez_constants['nfc'][inter.astype(int)]
    x = (alt - np.pi/4) / (np.pi/4)
    p = np.arange(5)[None, :]
    f2 = np.where(inter, (2.739 + .9891*np.sin(.3119 + 2.6*alt)) *
                  np.exp(-(np.pi/2 - alt)*(.4441 + 1.48*alt)),
                  0.274*(0.91 + 10*np.exp(-3*(np.pi/2 - alt)) +
                  0.45*np.square(sxyz[:, 2])))
    normfactor = np.sum(np.power(x[:, None], p) * normsc, 1)/f2/np.pi
    normfactor[skyclear == 1] = 0.777778
    groundbr = zenithbr*normfactor
    groundbr[skyclear > 1] += (6.8e-5/np.pi*solarrad*sxyz[:, 2])[skyclear > 1]
    groundbr *= ground_fac
    coefs = np.hstack((diffnorm[:, None], groundbr[:, None], cperez, sxyz))
    return coefs, solarrad


def sky_mtx(sxyz, dirdif, side, jn=4, ground_fac=0.2):
    """generate sky, ground and sun values from sun position and sky values

    Parameters
    ----------
    sxyz: np.array
        sun directions (N, 3)
    dirdif: np.array
        direct normal and diffuse horizontal radiation (W/m^2) (N, 2)
    side: int
        sky subdivision
    jn: int
        sky patch subdivision n = jn^2
    ground_fac: float
        scaling factor (reflecctance) for ground brightness

    Returns
    -------
    skymtx: np.array
        (N, side*side)
    grndval: np.array
        (N,)
    sunval: np.array
        (N, 4) - sun direction and radiance
    """
    coefs, solarrad = perez(sxyz, dirdif, ground_fac=ground_fac)
    uv = translate.bin2uv(np.arange(side*side), side, offset=0.0)
    jitter = translate.bin2uv(np.arange(jn*jn), jn, offset=0.0) + .5/jn
    uvj = uv[:, None, :] + jitter/side
    xyz = translate.uv2xyz(uvj.reshape(-1, 2), xsign=1).reshape(-1, 3)
    lum = perez_lum(xyz, coefs).reshape(coefs.shape[0], -1, jn*jn)
    lum = np.average(lum, -1)
    return lum, coefs[:, 1], np.hstack((sxyz, solarrad[:, None]))
