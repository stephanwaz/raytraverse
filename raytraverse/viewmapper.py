# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate


class ViewMapper(object):
    """translate between world and normalized UV space based on direction
    and view angle

    Parameters
    ----------
    dxyz: (float, float, float), optional
        central view direction
    viewangle: int, optional
        if < 180, the horizontal and vertical view angle, if greater, view
        becomes 360,180
    """

    def __init__(self, dxyz=(0.0, 1.0, 0.0), viewangle=360):
        self._viewangle = viewangle
        # float: aspect ratio width/height
        self.aspect = 1
        self.dxyz = dxyz

    @property
    def dxyz(self):
        """(float, float, float) central view direction"""
        return self._dxyz

    @property
    def ymtx(self):
        """yaw rotation matrix (to standard z-direction y-up)"""
        return self._ymtx

    @property
    def pmtx(self):
        """pitch rotation matrix (to standard z-direction y-up)"""
        return self._pmtx

    @property
    def bbox(self):
        """np.array of shape (2,2): bounding box of view"""
        return self._bbox

    @property
    def sf(self):
        """bbox scale factor"""
        return self._sf

    @dxyz.setter
    def dxyz(self, xyz):
        """set view parameters"""
        self._dxyz = translate.norm(xyz)
        self._ymtx, self._pmtx = zip(*[translate.rmtx_yp(x) for x in self.dxyz])
        if self._viewangle > 180:
            self._sf = np.array((1, 1))
            self._bbox = np.stack(((0, 0), (2, 1)))
            self.aspect = 2
        else:
            self._sf = np.array((self._viewangle/180, self._viewangle/180))
            self._bbox = np.stack((.5 - self._sf/2, .5 + self._sf/2))

    def view2world(self, xyz, i=0):
        return (self.ymtx[i].T@(self.pmtx[i].T@xyz.T)).T

    def world2view(self, xyz, i=0):
        return (self.pmtx[i]@(self.ymtx[i]@xyz.T)).T

    def xyz2uv(self, xyz, i=0):
        # rotate from world to view space
        rxyz = self.world2view(xyz, i)
        # translate to uv
        uv = translate.xyz2uv(rxyz)
        # scale to view uv space
        uv = (uv - self.bbox[None, 0])/self.sf[None, :]
        return uv

    def uv2xyz(self, uv, i=0):
        # scale to hemispheric uv
        uv = self.bbox[None, 0] + uv*self.sf[None, :]
        # translate to xyz with z view direction
        rxyz = translate.uv2xyz(uv)
        # rotate from this view space to world
        if np.asarray(i).size > 1:
            xyz = np.concatenate([self.view2world(r.reshape(1, -1), j)
                                  for r, j in zip(rxyz, i)])
        else:
            xyz = self.view2world(rxyz, i)
        return xyz

    def xyz2xy(self, xyz, i=0):
        rxyz = self.world2view(xyz, i)
        return translate.xyz2xy(rxyz)
