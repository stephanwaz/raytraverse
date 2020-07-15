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
    viewangle: float, optional
        if < 180, the horizontal and vertical view angle, if greater, view
        becomes 360,180
    """

    def __init__(self, dxyz=(0.0, 1.0, 0.0), viewangle=360.0):
        self._viewangle = viewangle
        # float: aspect ratio width/height
        self.aspect = 1
        self.dxyz = dxyz

    @property
    def dxyz(self):
        """(float, float, float) central view direction"""
        return self._dxyz

    @property
    def viewangle(self):
        """view angle"""
        return self._viewangle

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

    def pixelrays(self, res, i=0):
        pxy = (np.stack(np.mgrid[0:res, 0:res]).T + .5)
        return self.pixel2ray(pxy, res, i)

    # def pixelrays(self, res, i=0):
    #     pxy = (np.stack(np.mgrid[0:res, 0:res]).T + .5)/res
    #     if self.aspect == 2:
    #         rxyz, mask = translate.pxy2xyz(pxy, 180)
    #     else:
    #         rxyz, mask = translate.pxy2xyz(pxy, self.viewangle)
    #     xyz = self.view2world(rxyz.reshape(-1, 3), i).reshape(rxyz.shape)
    #     return xyz, mask

    def ray2pixel(self, xyz, res, i=0):
        xy = self.xyz2xy(xyz, i)
        print(np.percentile(xy, (0, 50, 100), 0))
        pxy = np.floor((xy/2 + .5) * res).astype(int)[:, -1::-1]
        pxy[:, 1] = res - pxy[:, 1]
        return pxy

    def pixel2ray(self, pxy, res, i=0):
        rxyz, mask = translate.pxy2xyz(pxy/res, self.viewangle/self.aspect)
        xyz = self.view2world(rxyz.reshape(-1, 3), i).reshape(rxyz.shape)
        return xyz, mask

    def pixel2omega(self, pxy, res):
        va = self.viewangle/self.aspect
        print(pxy/res, pxy)
        of = np.array(((.5, 0), (0, .5)))
        xa, _ = translate.pxy2xyz((pxy - of[0])/res, va)
        xb, _ = translate.pxy2xyz((pxy + of[0])/res, va)
        ya, _ = translate.pxy2xyz((pxy - of[1])/res, va)
        yb, _ = translate.pxy2xyz((pxy + of[1])/res, va)
        cp = np.cross(xb - xa, yb - ya)
        return np.linalg.norm(cp, axis=-1)

    def radians(self, vec, i=0):
        return np.arccos(np.einsum("i,ji->j", self.dxyz[i],
                                   vec.reshape(-1, vec.shape[-1])))

    def degrees(self, vec, i=0):
        return self.radians(vec, i) * 180/np.pi
