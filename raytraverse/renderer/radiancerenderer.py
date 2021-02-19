# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import shlex

from raytraverse import io
from raytraverse.renderer.renderer import Renderer
from raytraverse.crenderer import cRtrace


class RadianceRenderer(Renderer):
    """Virtual class for wrapping c++ Radiance renderer executable classes

    Do not use directly, either subclass or use existing: Rtrace, Rcontrib
    """

    name = "radiance_virtual"
    #: raytraverse.crenderer.cRtrace
    engine = cRtrace
    srcn = 1
    defaultargs = ""
    args = ""

    def __init__(self, rayargs=None, scene=None, nproc=None, default_args=True):
        type(self).instance = self.engine.get_instance()
        if default_args:
            if rayargs is None:
                rayargs = self.get_default_args()
            else:
                rayargs = f"{self.get_default_args()} {rayargs}"
        if rayargs is not None and scene is not None:
            self.set_args(rayargs, nproc)
            self.load_scene(scene)

    @classmethod
    def get_default_args(cls):
        return cls.defaultargs

    @classmethod
    def reset(cls):
        """reset engine instance and unset associated attributees"""
        cls.instance.reset()
        cls.scene = None
        cls.args = None

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
        nproc = io.get_nproc(nproc)
        cls.args = shlex.split(f"{cls.name} -n {nproc} {args}")
        nproc = cls.instance.initialize(cls.args)
        if nproc < 0:
            raise ValueError(f"Could not initialize {cls.__name__} with "
                             f"arguments: '{' '.join(cls.args)}'")

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
        if cls.args is None:
            raise ValueError(f'{cls.__name__} instance args must be '
                             'initialized before scene is loaded')
        cls.scene = scene
        cls.instance.load_scene(scene)
