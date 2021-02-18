# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse.renderer.radiancerenderer import RadianceRenderer
from raytraverse.crenderer import cRtrace
from raytraverse import io
import numpy as np


class Rtrace(RadianceRenderer):
    """singleton wrapper for c++ crenderer.cRtrace singleton class"""
    name = 'rtrace'
    engine = cRtrace
    defaultargs = "-av 0 0 0 -aa 0 -ab 7 -ad 128 -as 0 -c 10 -as 0 -lw 1e-5 -oZ"
    directargs = "-av 0 0 0 -ab 0 -oZ -lr 0"
    usedirect = False

    def __init__(self, rayargs=None, scene=None, nproc=None,
                 default_args=True, direct=False):
        type(self).usedirect = direct
        default_args = default_args or direct
        super().__init__(rayargs, scene, nproc, default_args=default_args)

    @classmethod
    def get_default_args(cls):
        if cls.usedirect:
            return cls.directargs
        else:
            return cls.defaultargs

    @classmethod
    def update_ospec(cls, vs):
        outcnt = cls.instance.update_ospec(vs)
        if outcnt < 0:
            raise ValueError(f"Could not update {cls.__name__} with "
                             f"outputs: '{vs}'")

    @classmethod
    def load_source(cls, srcname, freesrc=-1):
        cls.instance.load_source(srcname, freesrc)
