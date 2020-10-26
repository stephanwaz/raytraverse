# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse.renderer.radiancerenderer import RadianceRenderer
from raytraverse.crenderer import cRcontrib

if cRcontrib.version == "PyVirtual":
    from raytraverse.renderer.sprenderer import SPRcontrib
    Rcontrib = SPRcontrib
else:
    class Rcontrib(RadianceRenderer):
        """singleton wrapper for c++ crenderer.cRcontrib singleton class"""
        Engine = cRcontrib
        name = 'rcontrib'
        arg_prefix = "-o !cat"
