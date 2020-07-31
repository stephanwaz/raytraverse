# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""helper functions and classes"""


import numpy as np
from scipy.spatial import SphericalVoronoi, _voronoi


class SVoronoi(SphericalVoronoi):
    """this is a temporary fix for an apperent bug in SphericalVoronoi"""
    def sort_vertices_of_regions(self):
        if self._dim != 3:
            raise TypeError("Only supported for three-dimensional point sets")
        reg = [r for r in self.regions if len(r) > 0]
        _voronoi.sort_vertices_of_regions(self._simplices, reg)


class ArrayDict(dict):
    """tuple indexed dictionary indexable by np.array"""
    def __init__(self, d, tsize=2):
        self.tsize = tsize
        super(ArrayDict, self).__init__(d)

    def __getitem__(self, item):
        return np.vstack([super(ArrayDict, self).__getitem__(tuple(i)) for i in
                          np.reshape(item, (-1, self.tsize))])


class MemArraySeq(object):
    @staticmethod
    def _map(i):
        return np.memmap(i[0], dtype=i[1], mode=i[2], offset=i[3], shape=i[4])

    def __getitem__(self, item):
        return self._map(super().__getitem__(item))


class MemArrayList(MemArraySeq, tuple):
    def __new__(cls, arg):
        out = super().__new__(cls, arg)
        out.full_array = out.constructors()
        return out

    def __iter__(self):
        return (self._map(item) for item in super().__iter__())

    def constructors(self):
        return (item for item in super().__iter__())

    @property
    def full_array(self):
        return self._map(self.full_constructor)

    @full_array.setter
    def full_array(self, constructors):
        fulli = None
        shape = 0
        strides = [0]
        for i in constructors:
            if fulli is None:
                fulli = list(i)
            shape += i[4][0]
            strides.append(shape)
        fulli[4] = (shape, fulli[4][1])
        self.full_constructor = tuple(fulli)
        self.index_strides = tuple(strides)


class MemArrayDict(MemArraySeq, dict):

    def values(self):
        return (self._map(item) for item in super().values())

    def constructors(self):
        return (item for item in super().values())

    def full_array(self):
        return self._map(self.full_constructor())

    def full_constructor(self):
        fulli = None
        shape = 0
        for i in self.constructors():
            if fulli is None:
                fulli = list(i)
            shape += i[4][0]
        fulli[4] = (shape, fulli[4][1])
        return tuple(fulli)

    def index_strides(self):
        shape = 0
        strides = [0]
        for i in self.constructors():
            shape += i[4][0]
            strides.append(shape)
        return tuple(strides)
