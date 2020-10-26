# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import shlex
from raytraverse import io
from raytraverse.renderer.renderer import Renderer


class RadianceRenderer(Renderer):
    """Virtual class for wrapping c++ Radiance renderer executable classes"""

    returnbytes = False

    def __new__(cls, rayargs=None, scene=None, nproc=None, iot="ff"):
        cls.instance = cls.Engine.get_instance()
        return super().__new__(cls, rayargs=rayargs, scene=scene, nproc=nproc,
                               iot=iot)

    @classmethod
    def update_param(cls, args, nproc=None, iot="ff"):
        cls.returnbytes = iot[-1] != "a"
        nproc = io.get_nproc(nproc)
        cls.initialized = cls._set_args(args, iot, nproc)
        cls.instance.initialize(cls.initialized)

    @classmethod
    def initialize(cls, args, scene, nproc=None, iot="ff"):
        if cls.instance is None:
            cls.instance = cls.Engine.get_instance()
        if args is not None:
            firstload = not cls.initialized
            cls.update_param(args, nproc, iot)
            if firstload:
                cls.instance.load_scene(scene)
                # TODO: populate header
                cls.header = ""

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
        super().reset()

    @classmethod
    def reset_instance(cls):
        cls.instance.reset_instance()
        super().reset_instance()

    @classmethod
    def _set_args(cls, args, iot, nproc):
        return shlex.split(f"{cls.name} -f{iot} -n {nproc} {cls.arg_prefix}"
                           f" {args} -av 0 0 0")
