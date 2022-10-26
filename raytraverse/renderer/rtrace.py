# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import tempfile

import numpy as np

from raytraverse import translate, io
from craytraverse.renderer import Rtrace as pRtrace
from craytraverse.renderer.rtrace import rtrace_instance


class Rtrace(pRtrace):
    """singleton wrapper for c++ raytraverse.crenderer.cRtrace class

    this class sets default arguments, helps with initialization and setting
    cpu limits of the cRtrace instance. see raytraverse.crenderer.cRtrace
    for more details.

    Parameters
    ----------
    rayargs: str, optional
        argument string (options and flags only) raises ValueError if arguments
        are not recognized by cRtrace.
    scene: str, optional
        path to octree
    nproc: int, optional
        if None, sets nproc to cpu count, or the RAYTRAVERSE_PROC_CAP
        environment variable
    default_args: bool, optional
        if True, prepend default args to rayargs parameter
    direct: bool, optional
        if True use Rtrace.directargs in place of default (also if True, sets
        default_args to True.

    Examples
    --------

    Basic Initialization and call::

        r = renderer.Rtrace(args, scene)
        ans = r(vecs)
        # ans.shape -> (vecs.shape[0], 1)

    If rayargs include cache files (ambient cache or photon map) be careful
    with updating sources. If you are going to swap sources, update the
    arguments as well with the new paths::

        r = renderer.Rtrace(args, scene)
        r.set_args(args.replace("temp.amb", "temp2.amb"))
        r.load_source(srcdef)

    Note that if you are using ambient caching, you must give an ambient file,
    because without a file ambient values are not shared across processes or
    successive calls to the instance.
    """
    instance = rtrace_instance
    defaultargs = (f"-u+ -ab 16 -av 0 0 0 -aa 0 -as 0 -dc 1 -dt 0 -lr -14 -ad "
                   f"1000 -lw 0.00004 -st 0 -ss 16 -w-")
    directargs = "-w- -av 0 0 0 -ab 0 -lr 1 -n 1 -st 0 -ss 16 -lw 0.00004"
    usedirect = False
    nproc = None
    ospec = "Z"

    def __init__(self, rayargs=None, scene=None, nproc=None,
                 default_args=True, direct=False):
        type(self).usedirect = direct
        default_args = default_args or direct
        if direct:
            nproc = 1
        if default_args:
            if rayargs is None:
                rayargs = self.get_default_args()
            else:
                rayargs = f"{self.get_default_args()} {rayargs}"
        super().__init__(rayargs, scene, nproc)

    @classmethod
    def set_args(cls, args, nproc=None):
        """prepare arguments to call engine instance initialization

        Parameters
        ----------
        args: str
            rendering options
        nproc: int, optional
            cpu limit

        """
        if nproc is None:
            nproc = cls.nproc
        nproc = io.get_nproc(nproc)
        if "-ab 0" in args:
            nproc = 1
        super().set_args(args, nproc)

    @classmethod
    def get_default_args(cls):
        """return default arguments of the class"""
        if cls.usedirect:
            return cls.directargs
        else:
            return cls.defaultargs

    @classmethod
    def load_solar_source(cls, scene, sun, ambfile=None, intens=1):
        # load new source
        fd, srcdef = tempfile.mkstemp(dir=f"./{scene.outdir}/",
                                      prefix='tmp_src')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(scene.formatter.get_sundef(sun, (intens, intens,
                                                         intens)))
            cls.load_source(srcdef, ambfile=ambfile)
        finally:
            os.remove(srcdef)

    @classmethod
    def reset(cls):
        """reset engine instance and unset associated attributes"""
        cls.ospec = "Z"
        super().reset()

    @classmethod
    def get_sources(cls):
        """returns source information

        Returns
        -------
        sources: np.array
            x,y,z,v,a
            distant: direction, view angle, solid angle
            not distant: location, max radius, area
        distant: np.arrary
            booleans, true if source type is distant
        """
        srcs, distant = cls.instance.get_sources()
        srcs[distant, 3] = translate.chord2theta(srcs[distant, 3]) * 360/np.pi
        return srcs, distant
