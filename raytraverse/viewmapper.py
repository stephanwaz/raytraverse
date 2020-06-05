# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import re

import numpy as np
from matplotlib.path import Path

from raytraverse import translate


class ViewMapper(object):
    """translate between view and normalized UV space

    Parameters
    ----------
    dxy: (float, float), optional
        central view direction (must be horizontal)
    vh: int, optional
        horizontal view size (2-360)
    vv: int, optional
        vertical view size (2-180)
    """

    def __init__(self, dxy=(1.0, 0.0), vh=360, vv=180, axes=(0, 2, 1)):
        #: (int, int, int): transform axes for translate.xyz2uv
        self.axes = axes
        self._vh = vh
        self._vv = vv
        self.dxyz = dxy

    @property
    def dxyz(self):
        """(float, float, float) central view direction (must be horizontal)"""
        return self._dxyz[0]

    @property
    def duv(self):
        """(float, float) central view direction UV coordinates"""
        return self._duv[0]

    @property
    def bbox(self):
        """np.array of shape (2,2): bounding box of view"""
        return self._bbox

    @property
    def sf(self):
        """bbox scale factor"""
        return self._sf

    @dxyz.setter
    def dxyz(self, xy):
        """set view parameters"""
        self._dxyz = translate.norm(xy + (0.0, ))
        self._duv = translate.xyz2uv(self._dxyz, axes=self.axes)
        self._sf = np.array((self._vh/180, self._vv/180))
        self._bbox = np.stack((self.duv-self._sf/2, self.duv+self._sf/2))

    def uv2view(self, uv):
        """convert UV --> world

        Parameters
        ----------
        uv: np.array
            normalized UV coordinates of shape (N, 2)

        Returns
        -------
        view: np.array
            world xyz coordinates of shape (N, 3)
        """
        return translate.uv2xyz(self.bbox[None, 0] + uv * self.sf[None, :], axes=self.axes)

    def view2uv(self, xyz):
        """convert world --> UV

        Parameters
        ----------
        xyz: np.array
            world xyz coordinates, shape (N, 3)

        Returns
        -------
        uv: np.array
            normalized UV coordinates of shape (N, 2)
        """
        uv = (translate.xyz2uv(xyz, axes=self.axes) - self.bbox[None, 0]) / self.sf[None, :]
        return uv
