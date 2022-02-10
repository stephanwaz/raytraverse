# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.ndimage.filters import uniform_filter

from raytraverse import translate


class AngularMixin(object):
    """includes overrides of transformation functions for angular type mapper
    classes. Inherit before raytraverse.mapper.Mapper eg::

        NewMapper(AngularMixin, Mapper)

    initialization of NewMapper must include declarations of::

        self._viewangle = viewangle
        self._chordfactor = chordfactor
        self._ivm = ivm
    """

    _flipu = True
    _xsign = -1

    def xyz2uv(self, xyz):
        """transform from world xyz space to mapper UV space"""
        # rotate from world to view space
        rxyz = self.world2view(xyz)
        # translate to uv
        uv = translate.xyz2uv(rxyz, flipu=self._flipu)
        # scale to view uv space
        uv = (uv - self.bbox[None, 0])/self._sf[None, :]
        return uv

    def uv2xyz(self, uv, stackorigin=False):
        """transform from mapper UV space to world xyz"""
        # scale to hemispheric uv
        uv = self.bbox[None, 0] + np.atleast_2d(uv)*self._sf[None, :]
        # translate to xyz with z view direction
        rxyz = translate.uv2xyz(uv, xsign=self._xsign)
        # rotate from this view space to world
        xyz = self.view2world(rxyz)
        if stackorigin:
            xyz = np.hstack((np.broadcast_to(self.origin, xyz.shape), xyz))
        return xyz

    def xyz2vxy(self, xyz):
        """transform from world xyz to view image space (2d)"""
        rxyz = self.world2view(np.atleast_2d(xyz))
        xy = (translate.xyz2xy(rxyz, flip=self._flipu) * 180 /
              (self.viewangle * self._chordfactor))
        return xy/2 + .5

    def vxy2xyz(self, xy, stackorigin=False):
        """transform from view image space (2d) to world xyz"""
        pxy = np.atleast_2d(xy)
        pxy -= .5
        pxy *= (self.viewangle * self._chordfactor) / 180
        d = np.sqrt(np.sum(np.square(pxy), -1))
        z = np.cos(np.pi*d)
        nperr = np.seterr(all="ignore")
        d = np.where(d <= 0, np.pi, np.sqrt(1 - z*z)/d)
        np.seterr(**nperr)
        pxy *= d[..., None]
        xyz = np.concatenate((pxy, z[..., None]), -1)
        xyz = self.view2world(xyz.reshape(-1, 3)).reshape(xyz.shape)
        if stackorigin:
            xyz = np.hstack((np.broadcast_to(self.origin, xyz.shape), xyz))
        return xyz

    @staticmethod
    def framesize(res):
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

    def in_view(self, vec, indices=True, tol=0.0):
        """generate mask for vec that are in the field of view (up to
        180 degrees) if view aspect is 2, only tests against primary view
        direction"""
        ang = self.radians(vec) - tol
        mask = ang < (self._chordfactor * self.viewangle * np.pi / 360)
        if indices:
            return np.unravel_index(np.arange(ang.size)[mask], vec.shape[:-1])
        else:
            return mask

    def header(self, pt=(0, 0, 0), **kwargs):
        if np.allclose(self.dxyz, (0, 0, 1)):
            vup = "0 1 0"
        else:
            vup = "0 0 1"
        return ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3} -vp {4} {5} '
                '{6} -vu {7}'.format(self.viewangle, *self.dxyz, *pt, vup))

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

    def add_vecs_to_img(self, img, v, channels=(1, 0, 0), grow=0, fisheye=True):
        res = img.shape[-1]
        if fisheye:
            if self.aspect == 2:
                reverse = self.degrees(v) > 90
                pa = self.ivm.ray2pixel(v[reverse], res)
                pa[:, 0] += res
                pb = self.ray2pixel(v[np.logical_not(reverse)], res)
                xp = np.concatenate((pa[:, 0], pb[:, 0]))
                yp = np.concatenate((pa[:, 1], pb[:, 1]))
            else:
                pb = self.ray2pixel(v, res)
                xp = pb[:, 0]
                yp = pb[:, 1]
        else:
            pa = translate.uv2ij(self.xyz2uv(v), res)
            xp = res - 1 - pa[:, 0]
            yp = pa[:, 1]
        r = int(grow*2 + 1)
        if len(img.shape) == 2:
            try:
                channel = channels[0]
            except TypeError:
                channel = channels
            img[xp, yp] = channel
            if grow > 0:
                img = uniform_filter(img*r**2, r)
        else:
            for i in range(img.shape[0]):
                if channels[i] is not None:
                    img[i, xp, yp] = channels[i]
            if grow > 0:
                img = uniform_filter(img*r**2, (1, r, r))
        return img

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
