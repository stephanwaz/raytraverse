# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import shlex
from threading import Thread
import sys

import psutil


from raytraverse import io


class RadianceRenderer(object):
    """Virtual class for wrapping c++ Radiance renderer executable classes"""

    initialized = False
    instance = None
    _pyinstance = None
    Engine = None
    name = None
    header = ""
    returnbytes = False
    arg_prefix = ''

    def __new__(cls, rayargs=None, scene=None, nproc=None, iot="ff"):
        cls.instance = cls.Engine.get_instance()
        if cls._pyinstance is None:
            cls._pyinstance = object.__new__(cls)
        return cls._pyinstance

    def __init__(self, rayargs=None, scene=None, nproc=None, iot="ff"):
        self.initialize(rayargs, scene, nproc, iot)

    @classmethod
    def _set_args(cls, args, iot, nproc):
        return shlex.split(f"{cls.name} -f{iot} -h- -n {nproc} {cls.arg_prefix}"
                           f" {args}")

    @classmethod
    def initialize(cls, args, scene, nproc=None, iot="ff"):
        if cls.instance is None:
            cls.instance = cls.Engine.get_instance()
        if args is not None and not cls.initialized:
            if nproc is None:
                nproc = os.cpu_count()
            cls.initialized = cls._set_args(args, iot, nproc)
            cls.instance.initialize(cls.initialized)
            cls.header = ""
            cls.instance.load_scene(scene)
        cls.returnbytes = iot[-1] != "a"

    @classmethod
    def call(cls, rayfile, store=True, outf=None):
        if not cls.initialized:
            raise ValueError(f'{cls.__name__} instance not initialized')
        with io.CaptureStdOut(cls.returnbytes, store, outf) as capture:
            cls.instance.call(rayfile)
        return capture.stdout

    @classmethod
    def reset(cls):
        cls.instance.reset()
        cls.initialized = False
        cls.header = ""

    @classmethod
    def reset_instance(cls):
        cls.instance.reset_instance()
        cls.instance = None
        cls.initialized = False
        cls.header = ""

