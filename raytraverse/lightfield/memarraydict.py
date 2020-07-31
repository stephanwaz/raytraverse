# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np


class MemArrayDict(dict):
    """a dictionary like object that holds arguments for numpy.memmap, the
    getter returns a view to the array"""
    @staticmethod
    def _map(i):
        return np.memmap(i[0], dtype=i[1], mode=i[2], offset=i[3], shape=i[4])

    def __getitem__(self, item):
        return self._map(super().__getitem__(item))

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
