# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.mapper.mapper import Mapper
from raytraverse.mapper.angularmixin import AngularMixin
from raytraverse import translate


class ViewMapper(AngularMixin, Mapper):
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

    def __init__(self, dxyz=(0.0, 1.0, 0.0), viewangle=360.0, name='view',
                 origin=(0, 0, 0), jitterrate=0.9):
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
        super().__init__(dxyz=dxyz, sf=sf, bbox=bbox, aspect=aspect, name=name,
                         origin=origin, jitterrate=jitterrate)

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

