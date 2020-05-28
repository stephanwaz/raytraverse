# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for translating between coordinate spaces and resolutions"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter, uniform_filter


def norm(v):
    """normalize 2D array of vectors along last dimension"""
    return v/np.sqrt(np.sum(np.square(v), -1)).reshape(-1, 1)


def tpnorm(thetaphi):
    """normalize angular vector to 0-pi, 0-2pi"""
    thetaphi[:,0] = np.mod(thetaphi[:,0] + np.pi, np.pi)
    thetaphi[:, 1] = np.mod(thetaphi[:, 1] + 2*np.pi, 2*np.pi)
    return thetaphi


def uv2xy(uv):
    """translate from 2 x unit square (0,2),(0,1) to disk (x,y)
    http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric
    -map.html.
    """
    np.seterr(all="ignore")
    xy = np.empty((uv.shape[0], 2), np.float)
    u = uv[:, 0]
    v = uv[:, 1]
    n = np.where(u > 1.0, -1, 1)
    u = np.where(u > 1.0, u - 1.0, u)
    a = 2. * u - 1
    b = 2. * v - 1
    cond = a*a > b*b
    r = np.where(cond, a, np.where(b == 0, 0, b))
    phi = np.where(cond, b/(2*a), np.where(b == 0, 0, 1 - a/(2*b)))*np.pi/2
    xy[:, 0] = n*np.cos(phi)*r
    xy[:, 1] = np.sin(phi)*r
    return xy


def uv2xyz(uv, axes=(0, 1, 2), rhs=True):
    """translate from 2 x unit square (0,2),(0,1) to unit sphere (x,y,z)
    http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric
    -map.html.
    """
    np.seterr(all="ignore")
    rhs = (-1, 1)[rhs]
    xyz = np.empty((uv.shape[0], 3), np.float)
    u = uv[:, 0]
    v = uv[:, 1]
    n = np.where(u > 1.0, -1, 1) * rhs
    u = np.where(u > 1.0, u - 1.0, u)
    a = 2. * u - 1
    b = 2. * v - 1
    cond = a*a > b*b
    r = np.where(cond, a, np.where(b == 0, 0, b))
    phi = np.where(cond, b/(2*a), np.where(b == 0, 0, 1 - a/(2*b)))*np.pi/2
    sphterm = r*np.sqrt(2 - r*r)
    xyz[:, axes[0]] = n*np.cos(phi)*sphterm
    xyz[:, axes[1]] = np.sin(phi)*sphterm
    xyz[:, axes[2]] = n*(1 - r*r)
    return xyz


def xyz2uv(xyz, normalize=False, axes=(0, 1, 2), rhs=True):
    """translate from vector x,y,z (normalized) to u,v (0,2),(0,1)
    Shirley, Peter, and Kenneth Chiu. A Low Distortion Map Between Disk and
    Square. Journal of Graphics Tools, vol. 2, no. 3, Jan. 1997, pp. 45-52.
    Taylor and Francis+NEJM, doi:10.1080/10867651.1997.10487479.
    """
    if normalize:
        xyz = norm(xyz)
    rhs = (-1, 1)[rhs]
    uv = np.empty((xyz.shape[0], 2), np.float)
    n = np.where(xyz[:, axes[2]] < 0, -1, 1)
    r2 = 1 - n*xyz[:, axes[2]]
    x = xyz[:, axes[0]] / np.sqrt(2 - r2)
    y = xyz[:, axes[1]] / np.sqrt(2 - r2)
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y, x)
    pi4 = np.pi/4
    phi = phi + np.where(phi < -pi4, 2*np.pi, 0)
    a = np.where(phi < pi4, (r, phi*r/pi4),
                 np.where(phi < 3*pi4, (-(phi - np.pi/2)*r/pi4, r),
                          np.where(phi < 5*pi4, (-r, -(phi - np.pi)*r/pi4),
                                   ((phi - 3*np.pi/2)*r/pi4, -r)))).T
    uv[:, 0] = (a[:, 0]*n + np.where(n*rhs < 0, 3, 1)) / 2.
    uv[:, 1] = (a[:, 1] + 1) / 2.
    return uv


def tp2xyz(thetaphi):
    """calculate x,y,z vector from theta (0-pi) and phi (0-2pi) RHS Z-up"""
    theta = thetaphi[:, 0]
    phi = thetaphi[:, 1]
    sint = np.sin(theta)
    xyz = np.array([sint*np.cos(phi), sint*np.sin(phi),
                    np.cos(theta)]).T
    return norm(xyz)


def xyz2tp(xyz):
    """calculate theta (0-pi), phi from x,y,z RHS Z-up"""
    theta = np.arccos(xyz[:, 2])
    phi = np.where(np.isclose(theta, 0.0, atol=1e-10), np.pi,
                   np.where(np.isclose(theta, np.pi, atol=1e-10),
                            np.pi, np.arctan2(xyz[:, 1], xyz[:, 0])))
    return tpnorm(np.column_stack([theta, phi]))


def tp2uv(thetaphi):
    """calculate UV from theta (0-pi), phi"""
    return xyz2uv(tp2xyz(thetaphi))


def uv2tp(uv):
    """calculate theta (0-pi), phi from UV"""
    return xyz2tp(uv2xyz(uv))


def array2uv(ar, uaxis, vaxis):
    shape = (ar.shape[uaxis], ar.shape[vaxis])
    aspect = shape[0]/shape[1]
    inc = 1/shape[1]
    u = np.arange(inc/2, aspect, inc)
    v = np.arange(inc/2, 1, inc)
    return np.stack(np.meshgrid(u, v)).T.reshape(-1, 2)


def bin2uv(b, shape):
    inc = 1/shape[1]
    uv = np.vstack(np.unravel_index(b, shape)).T * inc + inc/2
    return uv


def ij2uv(ij, shape):
    inc = 1/shape[1]
    uv = ij * inc + inc/2
    return uv


def uv2bin(uv, shape):
    ij = uv2ij(uv, shape)
    return np.ravel_multi_index((ij[:, 0], ij[:, 1],), shape)


def uv2ij(uv, shape):
    aspect = shape[0]/shape[1]
    row = np.mod(np.floor(shape[0]*uv[:, 0]/aspect), shape[0]).astype(int)
    col = np.mod(np.floor(shape[1]*uv[:, 1]), shape[1]).astype(int)
    return np.vstack((row, col)).T


def resample(samps, ts, gauss=True, radius=None):
    rs = np.array(ts)/np.array(samps.shape)
    if np.prod(rs) > 1:
        for i in range(len(rs)):
            samps = np.repeat(samps, rs[i], i)
        if gauss:
            if radius is None:
                radius = tuple(rs - 1)
            samps = gaussian_filter(samps, radius)
    elif np.prod(rs) < 1:
        rs = (1/rs).astype(int)
        og = (-rs/2).astype(int)
        samps = uniform_filter(samps, rs, origin=og)
        for i, j in enumerate(rs):
            samps = np.take(samps, np.arange(0, samps.shape[i], j), i)
    return samps
