# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import functools
import os
from clasp.script_tools import try_mkdir


class LightField(object):
    """container for accessing sampled data

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry
    rebuild: bool, optional
        build kd-tree even if one exists
    src: str, optional
        prefix of data files to map
    """

    def __init__(self, scene, rebuild=False, src='sky', position=0, srcn=1,
                 rmraw=False, fvrays=0.0, calcomega=True):
        #: float: threshold for filtering direct view rays
        self._fvrays = fvrays
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
        self.srcn = srcn
        #: str: prefix of data files from sampler (stype)
        self.src = src
        self.position = position
        self._vec = None
        self._lum = None
        self._omega = None
        self._rmraw = rmraw
        self.scene = scene
        self.calcomega = calcomega
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

    @property
    @functools.lru_cache(1)
    def outfile(self):
        outdir = f"{self.scene.outdir}/{self.src}"
        try_mkdir(outdir)
        return f"{outdir}/{self.position:06d}.rytree"
