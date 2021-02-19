# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.renderer.radiancerenderer import RadianceRenderer
from raytraverse.crenderer import cRcontrib
from raytraverse.formatter import RadianceFormatter as Fmt


class Rcontrib(RadianceRenderer):
    """singleton wrapper for c++ raytrraverse.crenderer.cRcontrib class

    this class sets default arguments, helps with initialization and setting
    cpu limits of the cRcontrib instance. see raytrraverse.crenderer.cRcontrib
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
    skyres: float, optional
        approximate resolution for skypatch subdivision (in degrees). Patches
        will have (rounded) size skyres x skyres. So if skyres=10, each patch
        will be 100 sq. degrees (0.03046174197 steradians) and there will be
        18 * 18 = 324 sky patches.
    modname: str, optional
        passed the -m option of cRcontrib initialization
    ground: bool, optional
        if True include a ground source (included as a final bin)
    default_args: bool, optional
        if True, prepend default args to rayargs parameter

    Examples
    --------

    Basic Initialization and call::

        r = renderer.Rcontrib(args, scene)
        ans = r(vecs)
        # ans.shape -> (vecs.shape[0], 325)
    """
    name = 'rcontrib'
    #: raytraverse.crenderer.cRcontrib
    engine = cRcontrib
    ground = True
    side = 18
    srcn = 325
    modname = "skyglow"

    def __init__(self, rayargs=None, scene=None, nproc=None,
                 skyres=10.0, modname='skyglow', ground=True,
                 default_args=True):
        scene = self.setup(scene, ground, modname, skyres)
        super().__init__(rayargs, scene, nproc=nproc,
                         default_args=default_args)

    @classmethod
    def setup(cls, scene=None, ground=True, modname="skyglow", skyres=10.0):
        """set class attributes for proper argument initialization

        Parameters
        ----------
        scene: str, optional
            path to octree
        ground: bool, optional
            if True include a ground source (included as a final bin)
        modname: str, optional
            passed the -m option of cRcontrib initialization
        skyres: float, optional
            approximate resolution for skypatch subdivision (in degrees). Patches
            will have (rounded) size skyres x skyres. So if skyres=10, each patch
            will be 100 sq. degrees (0.03046174197 steradians) and there will be
            18 * 18 = 324 sky patches.

        Returns
        -------
        scene: str
            path to scene with added sky definition

        """
        cls.ground = ground
        if scene is not None:
            srcdef = Fmt.get_skydef((1, 1, 1), ground=True, name=modname)
            scene = Fmt.add_source(scene, srcdef)
        cls.side = int(np.floor(90/skyres)*2)
        cls.srcn = cls.side**2 + ground
        cls.modname = modname
        return scene

    @classmethod
    def get_default_args(cls):
        """construct default arguments"""
        return f"-av 0 0 0 -ab 7 -ad 10 -c {10*cls.srcn} -as 0 -lw 1e-5"

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
        args = (f" -V+ {args} -e 'side:{cls.side}' -f scbins.cal "
                f"-b bin -bn {cls.srcn} -m {cls.modname}")
        super().set_args(args, nproc)
