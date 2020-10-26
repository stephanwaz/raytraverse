# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
class Renderer(object):
    """virtual renderer class"""

    initialized = False
    instance = None
    _pyinstance = None
    Engine = None
    name = None
    header = ""
    arg_prefix = ''
    scene = None

    def __new__(cls, rayargs=None, scene=None, nproc=None, **kwargs):
        if cls._pyinstance is None:
            cls._pyinstance = object.__new__(cls)
        return cls._pyinstance

    def __init__(self, rayargs=None, scene=None, nproc=None, **kwargs):
        self.initialize(rayargs, scene, nproc, **kwargs)

    @classmethod
    def initialize(cls, args, scene, nproc=None, **kwargs):
        cls.scene = scene
        pass

    @classmethod
    def call(cls, rayfile, store=True, outf=None):
        return None

    @classmethod
    def reset(cls):
        cls.initialized = False
        cls.header = ""

    @classmethod
    def reset_instance(cls):
        cls.instance = None
        cls.initialized = False
        cls.header = ""

