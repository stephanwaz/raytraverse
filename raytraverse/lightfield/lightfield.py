# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.spatial import cKDTree


class LightField(object):
    """collection of light data with KDtree structure for spatial query

    Parameters
    ----------
    scene: raytraverse.scene.BaseScene
    vecs: np.array str
        the vectors used to organizing the child data as array or file shape
        (N,3) or (N,4) if 3, indexed from 0
    pm: raytraverse.mapper.PlanMapper
    src: str
        name of source group.
    """

    def __init__(self, scene, vecs, pm, src):
        self._datadir = f"{scene.outdir}/{pm.name}"
        self.scene = scene
        self.pm = pm
        self.src = src
        self._kd = None
        self._omega = None
        self._data = None
        self.vecs = vecs

    @property
    def vecs(self):
        """indexing vectors (such as position, sun positions, etc.)"""
        return self._vecs

    @property
    def samplelevel(self):
        """the level at which the vec was sampled (all zero if not provided
        upon initialization"""
        return self._samplelevel

    @vecs.setter
    def vecs(self, pt):
        pts, idx, samplelevel = self._load_vecs(pt)
        self._vecs = pts
        self.data = idx
        self._kd = None
        self._samplelevel = samplelevel
        self.omega = None

    @property
    def data(self):
        """light data"""
        return self._data

    @data.setter
    def data(self, idx):
        """virtual setter, override"""
        self._data = idx

    @property
    def kd(self):
        """kdtree for spatial queries built on demand"""
        if self._kd is None:
            self._kd = cKDTree(self.vecs)
        return self._kd

    @property
    def omega(self):
        """solid angle or area"""
        return self._omega

    @omega.setter
    def omega(self, oga):
        self._omega = oga

    def query(self, vecs):
        """return the index and distance of the nearest point to each of points

        Parameters
        ----------
        vecs: np.array
            shape (N, 3) vectors to query.

        Returns
        -------
        i: np.array
            integer indices of closest ray to each query
        d: np.array
            distance from query to point in spacemapper.
        """
        d, i = self.kd.query(vecs)
        return i, d

    def evaluate(self, *args, **kwargs):
        pass

    @staticmethod
    def _load_vecs(pt):
        try:
            pts = np.atleast_2d(np.loadtxt(pt))
        except TypeError:
            pts = pt
        if pts.shape[-1] == 3:
            idx = np.arange(pts.shape[0])
            samplelevel = np.zeros(pts.shape[0], dtype=int)
        elif pts.shape[-1] == 4:
            idx = pts[:, 0].astype(int)
            samplelevel = np.zeros(pts.shape[0], dtype=int)
            pts = pts[:, 1:]
        elif pts.shape[-1] == 5:
            samplelevel = pts[:, 0].astype(int)
            idx = pts[:, 1].astype(int)
            pts = pts[:, 2:]
        else:
            raise ValueError(f"vector array must have shape (N, [3, 4, or 5]) "
                             f"not {pts.shape}")
        return pts, idx, samplelevel
