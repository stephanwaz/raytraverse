# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse.renderer.radiancerenderer import RadianceRenderer


class Rcontrib(RadianceRenderer):
    """singleton wrapper for c++
    singleton class, note that all instances of this class will point to same
    c++ instance"""
    from raytraverse.crenderer import cRcontrib as Engine
