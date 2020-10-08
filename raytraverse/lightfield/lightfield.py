# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import pickle
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from raytraverse import translate
from raytraverse.craytraverse import interpolate_kdquery


def _interpolate(outv, d, i, src_lum, src_vec, oob, err=0.00436):
    idx = i[i < oob]
    if d[0] <= err:
        return src_lum[idx[0]]
    ivecs = translate.norm(src_vec[idx] - outv)
    indi = np.copy(idx)
    indji = np.arange(len(idx))
    indo = [indi[0]]
    indjo = [indji[0]]
    for i in range(2):
        mask = np.einsum("i,ji->j", ivecs[0], ivecs) < np.cos(np.pi/(2+2*i))
        ivecs = ivecs[mask]
        if ivecs.shape[0] == 0:
            break
        indi = indi[mask]
        indji = indji[mask]
        indo.append(indi[0])
        indjo.append(indji[0])
    if len(indo) == 1:
        return src_lum[indo[0]]
    dt = 1/d[indjo]**2
    n = np.sum(dt)
    return np.einsum("j,ji->i", dt/n, src_lum[indo])


def interpolate_query(src_lum, src_vec, src_kd, dest_vec, k=8,
                      err=0.00436, up=0.347296):
    """query a kd_tree and interpolate corresponding values. used to
    merge to kd_trees with vector and luminance

    Parameters
    ----------
    src_lum: np.array
        luminance values for src_kd, shape (src_vec[0], srcn)
    src_vec: np.array
        vectors of src_kd, shape (src_kd.n, 3)
    src_kd: scipy.spatial.cKDTree
    dest_vec: np.array
        destination vectors to interpolate to, shape (N, 3)
    k: int
        initial query size
    err: float
        chord length under which value is taken without interpolation
        default is .25 degrees = translate.theta2chord(.25*pi/180)
    up: float
        chord length of maximum search radius for neighbors
        default is 10 degrees  = translate.theta2chord(20*pi/180)

    Returns
    -------
    np.array
        shape of (dest_vec.shape[0], src_lum.shape[1])
    """
    errs, idxs = src_kd.query(dest_vec, k=k, distance_upper_bound=up)
    if k == 1:
        return src_lum[idxs]
    arrout = interpolate_kdquery(dest_vec, errs, idxs, src_vec, src_lum, err=err)
    # arrout = np.zeros((dest_vec.shape[0], src_lum.shape[1]))
    # for j, (outv, d, i) in enumerate(zip(dest_vec, errs, idxs)):
    #     arrout[j] = _interpolate(outv, d, i, src_lum, src_vec, src_kd.n,
    #                              err=err)
    return arrout


class LightField(object):
    """container for accessing sampled data

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    rebuild: bool, optional
        build kd-tree even if one exists
    prefix: str, optional
        prefix of data files to map
    """

    def __init__(self, scene, rebuild=False, prefix='sky', srcn=1, rmraw=False):
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
        self.srcn = srcn
        #: str: prefix of data files from sampler (stype)
        self.prefix = prefix
        self._vec = None
        self._lum = None
        self._omega = None
        self._rmraw = rmraw
        self.scene = scene
        self._rawfiles = self.raw_files()

    def __del__(self):
        try:
            if self._rmraw:
                for rf in self._rawfiles:
                    try:
                        os.remove(rf)
                    except IOError:
                        pass
        except AttributeError:
            pass

    def raw_files(self):
        return []

    @property
    def vec(self):
        """direction vector (3,)"""
        return self._vec

    @property
    def lum(self):
        """luminance (srcn,)"""
        return self._lum

    @property
    def omega(self):
        """solid angle (1,)"""
        return self._omega

    def outfile(self, idx):
        istr = "_".join([f"{i:04d}" for i in np.asarray(idx).reshape(-1)])
        return f"{self.scene.outdir}_{self.prefix}_{istr}"

    def items(self):
        return range(self.scene.area.npts)


