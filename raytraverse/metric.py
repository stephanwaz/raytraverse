# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""standardized metric functions"""
import numpy as np
from raytraverse import translate


def illum(vm, vec, omega, lum, scale=179, **kwargs):
    return np.einsum('i,i,i->', vm.ctheta(vec), lum, omega) * scale


def avglum(vm, vec, omega, lum, scale=179, area=None, **kwargs):
    if area is None:
        area = np.sum(omega)
    return np.einsum('i,i->', lum, omega) * scale / area


def sqlum(vm, vec, omega, lum, scale=179, area=None, **kwargs):
    if area is None:
        area = np.sum(omega)
    alum2 = avglum(vm, vec, omega, lum, scale=scale, area=area, **kwargs)**2
    a2lum = np.einsum('i,i,i->', lum, lum, omega) * scale**2 / area
    return a2lum / alum2


def to_plane(n, vec):
    nv = n.reshape(1, 3)
    proj = vec - np.tensordot(nv, vec, (-1, -1)).T * nv
    return translate.norm(proj)


def angle_vv(a, b):
    return np.arccos(np.tensordot(a, b, (-1, -1)))


def get_pidx_guth(sigma, tau):
    return np.exp((35.2 - 0.31889 * tau - 1.22 * np.exp(-2 * tau / 9)) /
                  1000 * sigma + (21 + 0.26667 * tau - 0.002963 * tau * tau) /
                  100000 * sigma * sigma)


def get_pidx_iwata(phi, theta):
    d = 1/np.tan(phi)
    s = np.tan(theta)/np.tan(phi)
    r = np.sqrt((1 + np.square(s))/np.square(d))
    return 1 + np.where(r > 0.6, np.minimum(3, r) * 1.2, r * 0.8)


def get_pidx_kim(sigma, tau):
    tau3 = np.power(tau, 3)
    tau2 = np.power(tau, 2)
    return np.exp((sigma - (-0.000009*tau3 + 0.0014*tau2 + 0.0866*tau + 21.633))
                  / (-0.000009*tau3 + 0.0013*tau2 + 0.0853*tau + 8.772))


def get_pos_idx(vm, vec, guth=True):
    vec = translate.norm(vec)
    up = vm.view2world(np.array((0, 1, 0)))
    #: sigma: angle between source and view direction
    sigma = vm.degrees(vec)
    #: tau: angle between vertical and source projected to view plane
    tau = angle_vv(up, to_plane(vm.dxyz, vec)) * 180 / np.pi
    if guth:
        hv = np.cross(vm.dxyz, up)
        vv = np.cross(vm.dxyz, hv)
        #: phi: vertical angle
        phi = angle_vv(vv, vec) - np.pi/2.0
        #: theta: horizontal angle
        theta = np.pi/2.0 - angle_vv(hv, vec)
        posidx = np.where(phi < 0, get_pidx_iwata(phi, theta),
                          get_pidx_guth(sigma, tau))
        posidx = np.minimum(16, posidx)
    else: # KIM model
        # from src/Radiance/util/evalglare.c
        posidx = get_pidx_kim(sigma, tau)
    return posidx


