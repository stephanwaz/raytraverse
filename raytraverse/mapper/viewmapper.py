# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.mapper.mapper import Mapper
from raytraverse import translate


class ViewMapper(Mapper):
    """translate between world direction vectors and normalized UV space for a
    given view angle. pixel projection yields equiangular projection

    Parameters
    ----------
    dxyz: tuple, optional
        central view direction
    viewangle: float, optional
        if < 180, the horizontal and vertical view angle, if greater, view
        becomes 360,180
    """

    def __init__(self, dxyz=(0.0, 1.0, 0.0), viewangle=360.0, name='view'):
        self._viewangle = viewangle
        if viewangle > 180:
            aspect = 2
            self._viewangle = 180
            sf = (1, 1)
            bbox = np.stack(((0, 0), (2, 1)))
        else:
            aspect = 1
            sf = np.array((self._viewangle/180, self._viewangle/180))
            bbox = np.stack((.5 - sf/2, .5 + sf/2))
        super().__init__(dxyz=dxyz, sf=sf, bbox=bbox, aspect=aspect, name=name)

    @property
    def aspect(self):
        return self._aspect

    @aspect.setter
    def aspect(self, a):
        self.area = 2*np.pi*(1 - np.cos(self.viewangle*np.pi*a/360))
        self._aspect = a
        cl = translate.theta2chord(np.pi/2)/(np.pi/2)
        va = self.viewangle*np.pi/360
        clp = translate.theta2chord(va)/va
        self._chordfactor = cl/clp

    @property
    def dxyz(self):
        """(float, float, float) central view direction"""
        return self._dxyz

    @dxyz.setter
    def dxyz(self, xyz):
        """set view parameters"""
        self._dxyz = translate.norm1(np.asarray(xyz).ravel()[0:3])
        self._rmtx = translate.rmtx_yp(self.dxyz)
        if self.aspect == 2:
            self._ivm = ViewMapper(-self.dxyz, 180)
        else:
            self._ivm = None

    def xyz2uv(self, xyz):
        """transform from world xyz space to mapper UV space"""
        # rotate from world to view space
        rxyz = self.world2view(np.atleast_2d(xyz))
        # translate to uv
        uv = translate.xyz2uv(rxyz)
        # scale to view uv space
        uv = (uv - self.bbox[None, 0])/self._sf[None, :]
        return uv

    def uv2xyz(self, uv, stackorigin=False):
        """transform from mapper UV space to world xyz"""
        # scale to hemispheric uv
        uv = self.bbox[None, 0] + np.atleast_2d(uv)*self._sf[None, :]
        # translate to xyz with z view direction
        rxyz = translate.uv2xyz(uv)
        # rotate from this view space to world
        xyz = self.view2world(rxyz)
        if stackorigin:
            xyz = np.hstack((np.broadcast_to(self.origin, xyz.shape), xyz))
        return xyz

    def xyz2vxy(self, xyz):
        """transform from world xyz to view image space (2d)"""
        rxyz = self.world2view(np.atleast_2d(xyz))
        xy = translate.xyz2xy(rxyz) * 180 / (self.viewangle * self._chordfactor)
        return xy/2 + .5

    def vxy2xyz(self, xy, stackorigin=False):
        """transform from view image space (2d) to world xyz"""
        pxy = np.atleast_2d(xy)
        pxy -= .5
        pxy *= (self.viewangle * self._chordfactor) / 180
        d = np.sqrt(np.sum(np.square(pxy), -1))
        z = np.cos(np.pi*d)
        d = np.where(d <= 0, np.pi, np.sqrt(1 - z*z)/d)
        pxy *= d[..., None]
        xyz = np.concatenate((pxy, z[..., None]), -1)
        xyz = self.view2world(xyz.reshape(-1, 3)).reshape(xyz.shape)
        if stackorigin:
            xyz = np.hstack((np.broadcast_to(self.origin, xyz.shape), xyz))
        return xyz

    def _framesize(self, res):
        return res, res

    def pixelrays(self, res):
        """world xyz coordinates for pixels in view image space"""
        a = super().pixelrays(res)
        if self.aspect == 2:
            b = self.ivm.pixelrays(res)
            return np.concatenate((a, b), 0)
        else:
            return a

    def pixel2omega(self, pxy, res):
        """pixel solid angle"""
        of = np.array(((.5, 0), (0, .5)))
        xa = self.vxy2xyz((pxy - of[0])/res)
        xb = self.vxy2xyz((pxy + of[0])/res)
        ya = self.vxy2xyz((pxy - of[1])/res)
        yb = self.vxy2xyz((pxy + of[1])/res)
        cp = np.cross(xb - xa, yb - ya)
        return np.linalg.norm(cp, axis=-1)

    def in_view(self, vec, indices=True):
        """generate mask for vec that are in the field of view (up to
        180 degrees) if view aspect is 2, only tests against primary view
        direction"""
        ang = self.radians(vec)
        mask = ang < (self._chordfactor * self.viewangle * np.pi / 360)
        if indices:
            return np.unravel_index(np.arange(ang.size)[mask], vec.shape[:-1])
        else:
            return mask

    def header(self, pt=(0, 0, 0), **kwargs):
        return ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3} -vp {4} {5} '
                '{6}'.format(self.viewangle, *self.dxyz, *pt))

    def init_img(self, res=512, pt=(0, 0, 0), **kwargs):
        """Initialize an image array with vectors and mask

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
                add_to_img(img[res:], vecs[res:][mask2], mask2)
        header: str
        """
        img = np.zeros((res*self.aspect, res))
        vecs = self.pixelrays(res)
        if self.aspect == 2:
            mask = self.in_view(vecs[0:res])
            mask2 = self.ivm.in_view(vecs[res:])
        else:
            mask = self.in_view(vecs)
            mask2 = None
        header = self.header(pt, **kwargs)
        return img, vecs, mask, mask2, header

    @property
    def viewangle(self):
        """view angle"""
        return self._viewangle

    @property
    def ivm(self):
        """viewmapper for opposite view direction (in case of 360 degree view"""
        return self._ivm

    def ctheta(self, vec):
        """cos(theta) (dot product) between view direction and vec"""
        vec = np.asarray(vec)
        return np.einsum("i,ji->j", self.dxyz,
                         vec.reshape(-1, vec.shape[-1]))

    def radians(self, vec):
        """angle in radians betweeen vieew direction and vec"""
        return np.arccos(self.ctheta(vec))

    def degrees(self, vec):
        """angle in degrees betweeen vieew direction and vec"""
        return self.radians(vec) * 180/np.pi
