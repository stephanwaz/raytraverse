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
    """translate between view and normalized UV space

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
        self.dxyz = dxyz

    @property
    def dxyz(self):
        """(float, float, float) central view direction (must be horizontal)"""
        return self._dxyz[0]

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
        if np.allclose(self.dxyz, (0, 1, 0)):
            self._up = (0, 0, 1)
        else:
            self._up = (0, 1, 0)
        if self._viewangle > 180:
            self._sf = np.array((1, 1))
            self._bbox = np.stack(((0, 0), (2, 1)))
        else:
            self._sf = np.array((self._viewangle/180, self._viewangle/180))
            self._bbox = np.stack((.5 - self._sf/2, .5 + self._sf/2))

    def xyz2uv(self, xyz):
        rxyz = translate.rotate(xyz, self.dxyz, (0, 0, 1), (0, 1, 0))
        uv = translate.xyz2uv(rxyz)
        uv = (uv - self.bbox[None, 0]) / self.sf[None, :]
        return uv

    def uv2xyz(self, uv):
        uv = self.bbox[None, 0] + uv * self.sf[None, :]
        rxyz = translate.uv2xyz(uv)
        return translate.rotate(rxyz, (0, 0, 1), self.dxyz, self._up)

