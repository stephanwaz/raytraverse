# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse import io


class RadianceRenderer(object):
    """Virtual class for wrapping c++ Radiance renderer executable classes"""

    initialized = False
    instance = None
    _pyinstance = None
    Engine = None

    def __new__(cls, rayargs=None):
        cls.instance = cls.Engine.get_instance()
        if cls._pyinstance is None:
            cls._pyinstance = object.__new__(cls)
        return cls._pyinstance

    def __init__(self, rayargs=None):
        self.initialize(rayargs)

    @classmethod
    def initialize(cls, args):
        if cls.instance is None:
            cls.instance = cls.Engine.get_instance()
        if args is not None and not cls.initialized:
            cls.initialized = args
            cls.instance.initialize(cls.initialized)

    @classmethod
    def call(cls, rayfile, returnbytes=False):
        if not cls.initialized:
            raise ValueError(f'{cls.__name__} instance not initialized')
        with io.CaptureStdOut(returnbytes) as capture:
            try:
                cls.instance.call(rayfile)
            except SystemExit as ex:
                print(ex)
        return capture.stdout

    @classmethod
    def reset(cls, args=None):
        cls.instance.reset()
        cls.initialized = False
        cls.initialize(args)

    @classmethod
    def reset_instance(cls):
        cls.instance.reset_instance()
        cls.instance = None
        cls.initialized = False

