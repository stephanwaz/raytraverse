# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from io import StringIO

import numpy as np

from raytraverse import io
from clasp.script_tools import pipeline


class SpRenderer(object):
    """sub-process renderer for calling external executables"""

    args = None
    scene = None
    name = "rtrace"
    defaultargs = ""
    _args = None
    nproc = None

    def __init__(self, rayargs=None, scene=None, nproc=None, default_args=True):
        if default_args:
            if rayargs is None:
                rayargs = self.get_default_args()
            else:
                rayargs = f"{self.get_default_args()} {rayargs}"
        if rayargs is not None and scene is not None:
            self.set_args(rayargs, nproc)
            self.load_scene(scene)

    @classmethod
    def __call__(cls, rays):
        s = StringIO()
        np.savetxt(s, rays)
        out = pipeline([f"{cls.name} -n {cls.nproc} {cls.args} {cls.scene}"],
                       inp=s.getvalue())
        return out

    def run(self, *args, **kwargs):
        """alias for call, for consistency with SamplerPt classes for nested
        dimensions of evaluation"""
        return self(args[0])

    @classmethod
    def get_default_args(cls):
        return cls.defaultargs

    @classmethod
    def reset(cls):
        """reset engine instance and unset associated attributees"""
        cls.scene = None
        cls.args = None
        cls._args = None

    @classmethod
    def set_args(cls, args, nproc=None):
        """prepare arguments to call engine instance initialization

        Parameters
        ----------
        args: str
            rendering options
        nproc: int, optional
            cpu limit

        """
        if nproc is None:
            nproc = cls.nproc
        nproc = io.get_nproc(nproc)
        if "-ab 0" in args:
            nproc = 1
        cls.nproc = nproc
        cls.args = args

    @classmethod
    def load_scene(cls, scene):
        """load octree file to engine instance

        Parameters
        ----------
        scene: str
            path to octree file

        Raises
        ------
        ValueError:
            can only be called after set_args, otherwise engine instance
            will abort.
        """
        cls.scene = scene
