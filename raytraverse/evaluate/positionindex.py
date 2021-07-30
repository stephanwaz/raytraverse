# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np
from raytraverse import translate


class PositionIndex(object):
    """calculate position index according to guth/iwata or kim

    Parameters
    ----------
    guth: bool
        if True, use Guth for the upper field of view and iwata for the lower
        if False, use Kim
    """

    def __init__(self, guth=True):
        self.guth = guth

    def positions(self, vm, vec):
        """calculate position indices for a set of vectors

        Parameters
        ----------
        vm: raytraverse.mapper.ViewMapper
            the view/analysis point, should have 180 degree field of view
        vec: np.array
            shape (N,3) the view vectors to calculate

        Returns
        -------
        posidx: np.arrray
            shape (N,) the position indices

        """
        vec = translate.norm(vec)
        up = vm.view2world(np.array((0, 1, 0)))
        #: sigma: angle between source and view direction
        sigma = vm.degrees(vec)
        return self._positions(vm.dxyz, vec, sigma, up)

    def _positions(self, viewvec, vec, sigma, up):
        #: tau: angle between vertical and source projected to view plane
        tau = self._angle_vv(up, self._to_plane(viewvec, vec))*180/np.pi
        if self.guth:
            hv = np.cross(viewvec, up)
            vv = np.cross(viewvec, hv)
            #: phi: vertical angle
            phi = self._angle_vv(vv, vec) - np.pi/2.0
            # iwata (2010) below horizon
            tau[:, phi < 0] = 90.0
            posidx = self._get_pidx_guth(sigma, tau)
            posidx = np.minimum(16, posidx)
        else:  # KIM model
            # from src/Radiance/util/evalglare.c
            posidx = self._get_pidx_kim(sigma, tau)
        return posidx.ravel()

    def positions_vec(self, viewvec, srcvec, up=(0, 0, 1)):
        sigma = np.arccos(np.einsum("ki,ji->kj", np.atleast_2d(viewvec),
                                    np.atleast_2d(srcvec))) * 180/np.pi
        return self._positions(viewvec, srcvec, sigma, up)

    @staticmethod
    def _to_plane(n, vec):
        nv = n.reshape(-1, 3)
        proj = vec[None] - np.tensordot(nv, vec, (-1, -1))[..., None]*nv[:, None, :]
        proj = proj/np.linalg.norm(proj, axis=-1)[..., None]
        return proj

    @staticmethod
    def _angle_vv(a, b):
        return np.arccos(np.tensordot(a, b, (-1, -1)))

    @staticmethod
    def _get_pidx_guth(sigma, tau):
        return np.exp((35.2 - 0.31889*tau - 1.22*np.exp(-2*tau/9)) /
                      1000*sigma + (21 + 0.26667*tau - 0.002963*tau*tau) /
                      100000*sigma*sigma)

    @staticmethod
    def _get_pidx_kim(sigma, tau):
        tau3 = np.power(tau, 3)
        tau2 = np.power(tau, 2)
        return np.exp(
            (sigma - (-0.000009*tau3 + 0.0014*tau2 + 0.0866*tau + 21.633))
            / (-0.000009*tau3 + 0.0013*tau2 + 0.0853*tau + 8.772))
