# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

class _VirtEngine:

    @classmethod
    def __call__(cls, rays):
        return None


class Renderer(object):
    """virtual singleton renderer class.
    the Renderer is implemented as a singleton as specific subclasses (rtrace,
    rcontrib) have many global variables set at import time. This ensures the
    python object is connected to the current state of the engine c++-class.

    All renderer classes are callable with with a numpy array of shape (N,6)
    representing the origin and direction of ray samples to calculate.
    """

    args = None
    _pyinstance = None
    instance = _VirtEngine()
    scene = None

    def __new__(cls, rayargs=None, scene=None, nproc=None, **kwargs):
        if cls._pyinstance is None:
            cls._pyinstance = object.__new__(cls)
        return cls._pyinstance

    @classmethod
    def __call__(cls, rays):
        return cls.instance(rays)
