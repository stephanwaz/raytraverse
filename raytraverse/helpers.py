# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""helper functions and classes"""
import os
from datetime import datetime, timezone
import subprocess

import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree
import raytraverse


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


def oconvline(scene):
    octe = f"{scene.outdir}/scene.oct"
    hdr = subprocess.run(f'getinfo {octe}'.split(), capture_output=True,
                         text=True)
    hdr = [i.strip() for i in hdr.stdout.split('\n')]
    return [i for i in hdr if i[0:5] == 'oconv']


def header(scene):
    hdr = []
    hdr += oconvline(scene)
    tf = "%Y:%m:%d %H:%M:%S"
    hdr.append("CAPDATE= " + datetime.now().strftime(tf))
    hdr.append("GMT= " + datetime.now(timezone.utc).strftime(tf))
    radversion = subprocess.run('rpict -version'.split(), capture_output=True,
                                text=True)
    hdr.append(f"SOFTWARE= {radversion.stdout}")
    lastmod = os.path.getmtime(os.path.dirname(raytraverse.__file__))
    tf = "%a %b %d %H:%M:%S %Z %Y"
    lm = datetime.fromtimestamp(lastmod, timezone.utc).strftime(tf)
    hdr.append(f"SOFTWARE= RAYTRAVERSE {raytraverse.__version__} lastmod {lm}")
    hdr.append("LOCATION= lat: {} lon: {} tz: {}".format(*scene.loc))
    return hdr
