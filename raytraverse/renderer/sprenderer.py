# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import shlex
import re
from subprocess import Popen, PIPE
from raytraverse import io
from raytraverse.renderer.renderer import Renderer


class SPRenderer(Renderer):
    """Subprocess renderer class"""
    cleanup = "rcalc -if3 -of -e $1=.265074126*$1+.670114631*$2+.064811243*$3"
    filter_bad_args = [(r"-Z\S*", ""), (r"-oZ", '-ov')]

    @classmethod
    def initialize(cls, args, scene, nproc=None, iot="ff"):
        super().initialize(args, scene)
        if nproc is None:
            nproc = os.cpu_count()
        if args is not None:
            if iot[-1] == 'a':
                raise ValueError(f'{cls.__name__} must output binary format')
            cls.cleanup = (f"rcalc -i{iot[-1]}3 -o{iot[-1]}"
                           " -e $1=.265074126*$1+.670114631*$2+.064811243*$3")
            for badarg, repl in cls.filter_bad_args:
                args = re.sub(badarg, repl, args)
            cls.initialized = cls._set_args(args + " -h- " + scene, iot, nproc)
            # TODO: populate header
            cls.header = ""

    @classmethod
    def call(cls, rayfile, store=True, outf=None, vecs2stdin=True):
        if not cls.initialized:
            raise ValueError(f'{cls.__name__} instance not initialized')
        if cls.Engine is None:
            raise NotImplementedError(f'{cls.__name__} is a virtual class')
        with io.CaptureStdOut(True, store, outf) as capture:
            if vecs2stdin:
                p = Popen(cls.initialized, stdout=PIPE,
                          stdin=open(rayfile, 'rb'))
            else:
                p = Popen(cls.initialized + [rayfile], stdout=PIPE)
            q = Popen(shlex.split(cls.cleanup), stdin=p.stdout)
            q.communicate()
        return capture.stdout

    @classmethod
    def _set_args(cls, args, iot, nproc):
        print((f"{cls.name} -f{iot} -n {nproc} {cls.arg_prefix}"
                           f" {args}"))
        return shlex.split(f"{cls.name} -f{iot} -n {nproc} {cls.arg_prefix}"
                           f" {args}")


class SPRtrace(SPRenderer):
    Engine = 'rtrace'
    name = 'rtrace'

    @classmethod
    def load_source(cls, srcname, **kwargs):
        pass


class SPRcontrib(SPRenderer):
    Engine = 'rcontrib'
    name = 'rcontrib'

