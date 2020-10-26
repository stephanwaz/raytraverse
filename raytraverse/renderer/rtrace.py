# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse.renderer.radiancerenderer import RadianceRenderer
from raytraverse.crenderer import cRtrace

if cRtrace.version == "PyVirtual":
    from raytraverse.renderer.sprenderer import SPRtrace
    Rtrace = SPRtrace
else:
    class Rtrace(RadianceRenderer):
        """singleton wrapper for c++ crenderer.cRtrace singleton class"""
        Engine = cRtrace
        name = 'rtrace'

        @classmethod
        def update_ospec(cls, vs, of='a'):
            if not cls.initialized:
                raise ValueError('Rtrace instance not initialized')
            cls.instance.update_ospec(vs, of)

        @classmethod
        def load_source(cls, srcname, freesrc=-1):
            cls.instance.load_source(srcname, freesrc)
