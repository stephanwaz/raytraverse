# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for computing optical values"""
import numpy as np


def rgbe2lum(rgbe):
    """
    convert from Radiance hdr rgbe 4-byte data format to floating point
    luminance.

    Parameters
    ----------
    rgbe: np.array
        r,g,b,e unsigned integers according to:
        http://radsite.lbl.gov/radiance/refer/filefmts.pdf

    Returns
    -------
    lum: luminance in cd/m^2
    """
    v = np.power(2., rgbe[:, 3] - 128).reshape(-1, 1) / 256
    lum = np.where(rgbe[:, 0:3] == 0, 0, (rgbe[:, 0:3] + 0.5) * v)
    # luminance = 179 * (0.265*R + 0.670*G + 0.065*B)
    return np.einsum('ij,j', lum, [47.435, 119.93, 11.635])


def rgb2rad(rgb):
    return np.einsum('ij,j', rgb, [0.265, 0.670, 0.065])


def rgb2lum(rgb):
    return np.einsum('ij,j', rgb, [47.435, 119.93, 11.635])


def calc_illum(sensor, rays, omegas, lum, normalize_omega=True):
    """calculate illuminance from a collection of rays

    Parameters
    ----------
    sensor: array like
        x,y,z,dx,dy,dz for sensor location
    rays: np.array (N, 3)
        direction vectors corresponding to omegas/luminances
    omegas: np.array (N,)
        unnormalized solid angles
    lum: np.array (d, N)
        luminances of each ray
    normalize_omega: bool, optional
            normalize estimated solid angles assuming a hemisphere is returned
            if Sampler used a subset of spherical sampling this should be set
            to False
    Returns
    -------

    """
    ctheta = np.dot(sensor, rays)
    if normalize_omega:
        tom = np.sum(omegas)
        nc = 2*np.pi/tom
        if np.abs(np.pi*2 - tom) > np.pi*.1:
            print("Warning, solid angle estimate off by > 5%! "
                  f"{tom} =/= {np.pi*2}")
    else:
        nc = 1
    return np.sum(lum * omegas * ctheta, axis=-1) * nc
