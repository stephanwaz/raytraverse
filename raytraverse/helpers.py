# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""helper functions and classes"""
import collections
import os
from abc import ABC
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import subprocess

import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree, _voronoi
import raytraverse
from raytraverse import quickplot


class SVoronoi(SphericalVoronoi):
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


class MemArrayList(tuple):

    def __new__(cls, arg):
        out = super().__new__(cls, arg)
        out.full_array = out.constructors()
        return out

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

    @staticmethod
    def _map(i):
        return np.memmap(i[0], dtype=i[1], mode=i[2], offset=i[3], shape=i[4])

    def __getitem__(self, item):
        return self._map(super().__getitem__(item))

    def __iter__(self):
        return (self._map(item) for item in super().__iter__())

    def constructors(self):
        return (item for item in super().__iter__())


class MemArrayDict(dict):
    @staticmethod
    def _map(i):
        return np.memmap(i[0], dtype=i[1], mode=i[2], offset=i[3], shape=i[4])

    def __getitem__(self, item):
        return self._map(super().__getitem__(item))

    def __iter__(self):
        return (self._map(item) for item in super().__iter__())

    def constructors(self):
        return (item for item in super().__iter__())


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
    try:
        hdr.append("LOCATION= lat: {} lon: {} tz: {}".format(*scene.loc))
    except TypeError:
        pass
    return hdr


def draw_from_pdf(pdf, threshold):
    np.set_printoptions(5, suppress=True)
    nsampc = int(np.sum(pdf > threshold))
    if nsampc == 0:
        return None
    clip = pdf > threshold/2
    pnorm = pdf[clip]/np.sum(pdf[clip])
    candidates = np.arange(pdf.size, dtype=np.uint32)[clip]
    return np.random.default_rng().choice(candidates, nsampc, replace=False,
                                          p=pnorm)


def mk_vector_ball(v):
    d_kd = cKDTree(v)
    omega = SVoronoi(v).calculate_areas()[:, None]
    return d_kd, omega


def skybin_pdf(outf, idxs, constructor, sb, shape, maxspec=0.3):
    lums = np.memmap(*constructor, order='F')[sb]
    # lums[np.greater(lums, maxspec)] = 0
    print('set')
    f = open(outf, 'wb')
    np.save(f, lums[idxs].reshape(shape))
    f.close()
    print('saved')
    return outf


def skybin_idx(skyfield, shape, interp=1):
    si = np.stack(np.unravel_index(np.arange(np.product(shape)), shape))
    uv = (si.T + .5)/shape[1]
    grid = skyfield.scene.view.uv2xyz(uv)
    futures = []
    strides = skyfield.lum.index_strides
    with ProcessPoolExecutor() as exc:
        for pt in range(len(skyfield.vec)):
            futures.append(exc.submit(skyfield.d_kd[pt].query, grid, interp))
    idxs = []
    errs = []
    for fu, stride in zip(futures, strides):
        r = fu.result()
        idxs.append(r[1] + stride)
        errs.append(r[0])
    return np.vstack(idxs), np.vstack(errs)
