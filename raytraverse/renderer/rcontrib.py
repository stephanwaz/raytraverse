# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import shutil

from raytraverse.renderer.radiancerenderer import RadianceRenderer
from raytraverse.crenderer import cRcontrib
import raytraverse

path = os.path.dirname(raytraverse.__file__)
rcontrib_capture_file = f'{path}/tmp_rcontrib_capture'


class Rcontrib(RadianceRenderer):
    """singleton wrapper for c++ crenderer.cRcontrib singleton class"""
    Engine = cRcontrib
    name = 'rcontrib'
    arg_prefix = "-o !cat"
