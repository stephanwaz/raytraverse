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
        if self._rmraw:
            for rf in self._rawfiles:
                try:
                    os.remove(rf)
                except IOError:
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
        return range(np.product(self.scene.ptshape))


