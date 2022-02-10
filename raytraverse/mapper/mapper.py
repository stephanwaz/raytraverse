# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import functools

import numpy as np
from scipy.ndimage.filters import uniform_filter

from raytraverse import translate, io


class Mapper(object):
    """translate between world and normalized UV space. do not use
    directly, instead use an inheriting class.

    Parameters
    ----------
    sf: tuple np.array, optional
        scale factor for each axis (array of length(2)
    bbox: tuple np.array, optional
        bounding box for mapper shape (2, 2)
    name: str, optional
        used for output file naming
    """

    def __init__(self, dxyz=(0.0, 0.0, 1.0), sf=(1, 1), bbox=((0, 0), (1, 1)),
                 aspect=None, name='mapper', origin=(0, 0, 0), jitterrate=1.0):
        self._sf = np.asarray(sf).flatten()
        self._bbox = np.asarray(bbox).reshape(2, 2)
        self.name = name
        self.aspect = aspect
        self.dxyz = dxyz
        self.origin = origin
        self.jitterrate = jitterrate

    @property
    def aspect(self):
        return self._aspect

    @aspect.setter
    def aspect(self, a):
        xd = self.bbox[1, 0] - self.bbox[0, 0]
        yd = self.bbox[1, 1] - self.bbox[0, 1]
        self.area = xd * yd
        if a is None:
            self._aspect = xd / yd
        else:
            self._aspect = a

    @property
    def dxyz(self):
        """(float, float, float) central view direction"""
        return self._dxyz

    @dxyz.setter
    def dxyz(self, xyz):
        """set view parameters"""
        self._dxyz = translate.norm1(np.asarray(xyz).ravel()[0:3])
        self._rmtx = (np.eye(3), np.eye(3))

    @property
    def bbox(self):
        """np.array of shape (2,2): bounding box of view"""
        return self._bbox

    def view2world(self, xyz):
        """rotate vectors from view direction to world Z"""
        xyz = np.atleast_2d(xyz)
        ymtx, pmtx = self._rmtx
        return (ymtx.T@(pmtx.T@xyz.T)).T

    def world2view(self, xyz):
        """rotate vectors from world Z to view direction"""
        xyz = np.atleast_2d(xyz)
        ymtx, pmtx = self._rmtx
        return (pmtx@(ymtx@xyz.T)).T

    def xyz2uv(self, xyz):
        """transform from world xyz space to mapper UV space"""
        ishape = xyz.shape
        vxy = self.world2view(np.reshape(xyz, (-1, 3)))[:, 0:2]
        uv = (vxy - self.bbox[None, 0])/self._sf[None, :]
        return uv.reshape(*ishape[:-1], 2)

    def uv2xyz(self, uv, stackorigin=False):
        """transform from mapper UV space to world xyz"""
        uv = self.bbox[None, 0] + np.reshape(uv, (-1, 2))*self._sf[None, :]
        xyz = self.view2world(np.hstack((uv, np.zeros(len(uv), 1))))
        if stackorigin:
            xyz = np.hstack((np.broadcast_to(self.origin, xyz.shape), xyz))
        return xyz

    def idx2uv(self, idx, shape, jitter=True):
        """
        Parameters
        ----------
        idx: flattened index
        shape:
            the shape to unravel into
        jitter: bool, optional
            randomly offset coordinates within grid

        Returns
        -------
        uv: np.array
            uv coordinates
        """
        si = np.stack(np.unravel_index(idx, shape))
        if jitter:
            rng = ((1 - self.jitterrate)/2, (1 + self.jitterrate)/2)
            offset = np.random.default_rng().uniform(*rng, si.shape).T
        else:
            offset = 0.5
        uv = (si.T + offset)/np.asarray(shape)
        return uv

    @staticmethod
    def uv2idx(uv, shape):
        ij = (uv * np.asarray(shape)).astype(int)
        return ij[:, 0] * shape[1] + ij[:, 1]

    def xyz2vxy(self, xyz):
        """transform from world xyz to view image space (2d)"""
        return self.xyz2uv(xyz)

    def vxy2xyz(self, xy, stackorigin=False):
        """transform from view image space (2d) to world xyz"""
        xy[..., 0] = 1 - xy[..., 0]
        return self.uv2xyz(xy, stackorigin=stackorigin)

    @functools.lru_cache(1)
    def framesize(self, res):
        if self.aspect < 1:
            yres = res
            xres = int(round(res*self.aspect))
        else:
            xres = res
            yres = int(round(res/self.aspect))
        return xres, yres

    def pixels(self, res):
        """generate pixel coordinates for image space"""
        xres, yres = self.framesize(res)
        return np.stack(np.mgrid[0:xres, 0:yres], 2) + .5

    def pixelrays(self, res):
        """world xyz coordinates for pixels in view image space"""
        pxy = self.pixels(res)
        return self.pixel2ray(pxy, res)

    def ray2pixel(self, xyz, res, integer=True):
        """world xyz to pixel coordinate"""
        try:
            xres, yres = res
        except TypeError:
            xres, yres = self.framesize(res)
        pxy = self.xyz2vxy(xyz) * np.array([[xres, yres]])
        if integer:
            pxy = np.floor(pxy).astype(int)
        pxy[:, 0] = xres - 1 - pxy[:, 0]
        return pxy

    def pixel2ray(self, pxy, res):
        """pixel coordinate to world xyz vector"""
        try:
            xres, yres = res
        except TypeError:
            xres, yres = self.framesize(res)
        return self.vxy2xyz(pxy/np.broadcast_to((xres, yres), pxy.shape))

    def pixel2omega(self, pxy, res):
        """pixel area"""
        xres, yres = self.framesize(res)
        return np.full(len(pxy), self.area / (xres * yres))

    def in_view(self, vec, indices=True):
        """generate mask for vec that are in the field of view"""
        uv = self.xyz2uv(vec)
        mask = np.logical_and(uv[:, 0] >= 0, uv[:, 0] <= 1,
                              uv[:, 1] >= 0, uv[:, 1] <= 1)
        if indices:
            return np.unravel_index(np.arange(vec.shape[0])[mask],
                                    vec.shape[:-1])
        else:
            return mask

    def header(self, **kwargs):
        return 'VIEW= -vtl -vv 1 -vh 1'

    def init_img(self, res=512, **kwargs):
        """Initialize an image array with vectors and mask

        Parameters
        ----------
        res: int, optional
            image array resolution
        kwargs:
            passed to self.header

        Returns
        -------
        img: np.array
            zero array of shape (res, res)
        vecs: np.array
            direction vectors corresponding to each pixel (img.size, 3)
        mask: np.array
            indices of flattened img that are in view
        mask2: np.array None
            if ViewMapper has inverse, mask for opposite view, usage::

                add_to_img(img, vecs[mask], mask)
                add_to_img(img[res:], vecs[res:][mask2], mask2
        header: str
        """
        img = np.zeros(self.framesize(res))
        vecs = self.pixelrays(res)
        mask = self.in_view(vecs)
        mask2 = None
        header = self.header(**kwargs)
        return img, vecs, mask, mask2, header

    def add_vecs_to_img(self, img, v, channels=(1, 0, 0), grow=0, **kwargs):
        pxy = self.ray2pixel(v, img.shape[-2:])
        xp = pxy[:, 0]
        yp = pxy[:, 1]
        r = int(grow*2 + 1)
        if len(img.shape) == 2:
            try:
                channel = channels[0]
            except TypeError:
                channel = channels
            img[xp, yp] = channel
            if grow > 1:
                img = uniform_filter(img*r**2, r)
        else:
            imgv = np.moveaxis(img, 0, 2)
            imgv[xp, yp] = channels
            if grow > 1:
                img = uniform_filter(img*r**2, (1, r, r))
        return img

    def plot(self, xyz, outf, res=1000, grow=1, **kwargs):
        img = np.zeros(self.framesize(res))
        img = self.add_vecs_to_img(img, xyz, grow=grow, **kwargs)
        io.array2hdr(img, outf, [self.header()])
