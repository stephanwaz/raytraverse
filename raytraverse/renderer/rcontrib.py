# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import numpy as np
import clasp.script_tools as cst

from raytraverse import io
from raytraverse.renderer.radiancerenderer import RadianceRenderer
from raytraverse.crenderer import cRcontrib
from raytraverse.formatter import RadianceFormatter as Fmt


class Rcontrib(RadianceRenderer):
    """wrapper for c++ crenderer.cRcontrib singleton class"""
    name = 'rcontrib'
    engine = cRcontrib
    ground = True
    side = 18
    srcn = 325
    modname = "skyglow"

    def __init__(self, rayargs=None, scene=None, nproc=None,
                 skyres=10.0, modname='skyglow', ground=True,
                 default_args=True):
        scene = self.setup(scene, ground, modname, skyres)
        super().__init__(rayargs, scene, nproc=nproc,
                         default_args=default_args)

    @classmethod
    def setup(cls, scene=None, ground=True, modname="skyglow", skyres=10.0):
        cls.ground = ground
        if scene is not None:
            srcdef = Fmt.get_skydef((1, 1, 1), ground=True, name=modname)
            scene = Fmt.add_source(scene, srcdef)
        cls.side = int(np.floor(90/skyres)*2)
        cls.srcn = cls.side**2 + ground
        cls.modname = modname
        return scene

    @classmethod
    def get_default_args(cls):
        return f"-av 0 0 0 -ab 7 -ad 10 -c {10*cls.srcn} -as 0 -lw 1e-5 -Z+"

    @classmethod
    def set_args(cls, args, nproc=None):
        args = (f" -V+ {args} -e 'side:{cls.side}' -f scbins.cal "
                f"-b bin -bn {cls.srcn} -m {cls.modname}")
        super().set_args(args, nproc)
