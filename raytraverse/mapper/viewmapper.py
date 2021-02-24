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
    dxyz: tuple, optional
        central view direction
    viewangle: float, optional
        if < 180, the horizontal and vertical view angle, if greater, view
        becomes 360,180
    """

    def __init__(self, dxyz=(0.0, 1.0, 0.0), viewangle=360.0, name='view',
                 mtxs=None, imtxs=None):
        # float: aspect ratio width/height
        self.aspect = 1
        self._viewangle = viewangle
        if viewangle > 180:
            self.aspect = 2
            self._viewangle = 180
        self._initmtx = (mtxs, imtxs)
        self.dxyz = dxyz
        self._initmtx = None
        self.area = 2*np.pi*(1 - np.cos(self.viewangle*np.pi*self.aspect/360))
        cl = translate.theta2chord(np.pi/2)/(np.pi/2)
        va = self.viewangle*np.pi/360
        clp = translate.theta2chord(va)/va
        self._chordfactor = cl / clp
        self.name = name

    def __getitem__(self, item):
        mtxs = [(self.ymtx[item], self.pmtx[item])]
        if self._ivm is not None:
            imtxs = [(self.ivm.ymtx[item], self.ivm.pmtx[item])]
        else:
            imtxs = None
        return ViewMapper(self.dxyz[item], self.viewangle, self.name,
                          mtxs=mtxs, imtxs=imtxs)

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

    @property
    def ivm(self):
        """viewmapper for opposite view direction (in case of 360 degree view"""
        return self._ivm

    @dxyz.setter
    def dxyz(self, xyz):
        """set view parameters"""
        self._dxyz = translate.norm(xyz)
        if self._initmtx[0] is None:
            self._ymtx, self._pmtx = zip(*[translate.rmtx_yp(x)
                                           for x in self.dxyz])
        else:
            self._ymtx, self._pmtx = zip(*self._initmtx[0])
        if self.aspect == 2:
            self._sf = np.array((1, 1))
            self._bbox = np.stack(((0, 0), (2, 1)))
            self._ivm = ViewMapper(-self.dxyz, 180, mtxs=self._initmtx[1])
        else:
            self._sf = np.array((self._viewangle/180, self._viewangle/180))
            self._bbox = np.stack((.5 - self._sf/2, .5 + self._sf/2))
            self._ivm = None

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

    def pixels(self, res):
        return np.stack(np.mgrid[0:res, 0:res], 2) + .5

    def pixelrays(self, res, i=0):
        pxy = self.pixels(res)
        if self.aspect == 2:
            a = self.pixel2ray(pxy, res, i)
            b = self.ivm.pixel2ray(pxy, res, i)
            return np.concatenate((a, b), 0)
        else:
            return self.pixel2ray(pxy, res, i)

    def ray2pixel(self, xyz, res, i=0, integer=True):
        xy = self.xyz2xy(xyz, i) * 180 / (self.viewangle * self._chordfactor)
        pxy = (xy/2 + .5) * res
        if integer:
            pxy = np.floor(pxy).astype(int)
        pxy[:, 0] = res - 1 - pxy[:, 0]
        return pxy

    def pixel2ray(self, pxy, res, i=0):
        rxyz = translate.pxy2xyz(pxy/res, self.viewangle)
        xyz = self.view2world(rxyz.reshape(-1, 3), i).reshape(rxyz.shape)
        return xyz

    def pixel2omega(self, pxy, res):
        of = np.array(((.5, 0), (0, .5)))
        xa = translate.pxy2xyz((pxy - of[0])/res, self.viewangle)
        xb = translate.pxy2xyz((pxy + of[0])/res, self.viewangle)
        ya = translate.pxy2xyz((pxy - of[1])/res, self.viewangle)
        yb = translate.pxy2xyz((pxy + of[1])/res, self.viewangle)
        cp = np.cross(xb - xa, yb - ya)
        return np.linalg.norm(cp, axis=-1)

    def ctheta(self, vec, i=0):
        vec = np.asarray(vec)
        return np.einsum("i,ji->j", self.dxyz[i],
                         vec.reshape(-1, vec.shape[-1]))

    def radians(self, vec, i=0):
        return np.arccos(self.ctheta(vec, i))

    def degrees(self, vec, i=0):
        return self.radians(vec, i) * 180/np.pi

    def in_view(self, vec, i=0, indices=True):
        ang = self.radians(vec, i)
        mask = ang < (self._chordfactor * self.viewangle * np.pi / 360)
        if indices:
            return np.unravel_index(np.arange(ang.size)[mask], vec.shape[:-1])
        else:
            return mask

    def init_img(self, res=512, pt=(0, 0, 0)):
        """Initialize an Image array with vectors and mask

        Parameters
        ----------
        res: int, optional
            image array resolution
        pt: tuple, optional
            view point for image header

        Returns
        -------
        img: np.array
            zero array of shape (res*self.aspect, res)
        vecs: np.array
            direction vectors corresponding to each pixel (img.size, 3)
        mask: np.array
            indices of flattened img that are in view
        mask2: np.array None
            if ViewMapper is 360 degree, include mask for opposite view to use::

                add_to_img(img, vecs[mask], mask)
                add_to_img(img[res:], vecs[res:][mask2], mask2
        header: str

        """
        img = np.zeros((res*self.aspect, res))
        vecs = self.pixelrays(res)
        mask = self.in_view(vecs[0:res])
        if self.aspect == 2:
            mask2 = self.ivm.in_view(vecs[res:])
        else:
            mask2 = None
        header = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3} -vp {4} {5} '
                  '{6}'.format(self.viewangle, *self.dxyz[0], *pt))
        return img, vecs, mask, mask2, header
