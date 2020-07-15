# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""helper functions and classes"""

import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree


class ArrayDict(dict):
    """tuple indexed dictionary indexable by np.array"""
    def __init__(self, d, tsize=2):
        self.tsize = tsize
        super(ArrayDict, self).__init__(d)

    def __getitem__(self, item):
        return np.vstack([super(ArrayDict, self).__getitem__(tuple(i)) for i in
                          np.reshape(item, (-1, self.tsize))])


def sunfield_load_item(vlamb, vlsun, i, j, maxspec):
    """function for pool processing sunfield results"""
    # remove accidental direct hits
    notspa = vlamb[:, i + 3] < maxspec
    notsps = vlsun[:, 3] < maxspec
    vecs = np.vstack((vlamb[notspa, 0:3],
                      vlsun[notsps, 0:3]))
    lums = np.hstack((vlamb[notspa, i + 3],
                      vlsun[notsps, 3]))[:, None]
    omega = SphericalVoronoi(vecs).calculate_areas()[:, None]
    vlo = np.hstack((vecs, lums, omega))
    d_kd = cKDTree(vecs)
    return (j, i), vlo, d_kd
