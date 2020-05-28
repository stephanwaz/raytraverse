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
from skyfield.api import load_file, Topos, load, utc

from raytraverse import translate

try:
    planets = load_file(os.path.dirname(__file__) + '/de421.bsp')
except OSError as ex:
    planets = load('de421.bsp')
sun, earth = planets['sun'], planets['earth']



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
            try:
                dp = [h[0], h[1], h[2], h[3], h[4]]
            except IndexError:
                dp = [h[0], h[1], h[2], h[3], h[4]]
            hoff = 0
        data.append([int(i.strip()) for i in dp[0:2]] +
                    [float(dp[2]) - hoff] +
                    [float(i.strip()) for i in dp[3:]])
    return np.array(data)


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
            lat = [i.split()[-1] for i in hdr if re.match("latitude.*", i)][0]
            lon = [i.split()[-1] for i in hdr if re.match("longitude.*", i)][0]
            tz = [i.split()[-1] for i in hdr if re.match("time_zone.*", i)][0]
            loc = [i.split()[-1] for i in hdr if re.match("place.*", i)][0]
            try:
                elev = \
                [i.split()[-1] for i in hdr if re.match("site_elevation.*", i)][
                    0]
            except Exception:
                elev = '0.0'
    except Exception as ex:
        raise ValueError("bad epw header", ex)
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
        st = ['{}-{:02.00f}-{:02.00f}T{:02.00f}:{:02.00f}'.format(year, *ts),]
    else:
        if ts.shape[1] < 4:
            hm = np.modf(ts[:, 2:3])
            ts = np.hstack((ts[:,0:2], hm[1], hm[0]*60))
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
    tz = mer / 15.
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
    np.array([float, float, float])
        Sun position as (x, y, z)
    """
    dt = datetime64_2_datetime(timesteps, mer=mer)
    alt, az = sunpos_utc(dt, lat, lon, builtin)
    az = az.radians - ro * np.pi/180
    # translate to spherical rhs before calling translate.tp2xyz
    thetaphi = np.column_stack([np.pi/2 - alt.radians, np.pi/2 - az])
    return translate.tp2xyz(thetaphi)
